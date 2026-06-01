"""
Pi05 (π0.5) wrapper for ACR-VLA.

Provides:
  1. obs_features extraction (mean-pooled VLM hidden states, 2048D)
  2. K-sample flow-matching generation with sequential denoising
  3. LIBERO data preprocessing (image, language, state → PI05 inputs)

Preprocessing follows the official lerobot PI05 pipeline:
  - Images: resize_with_pad to 224×224 → float [0,1]
  - State: MEAN_STD normalize → discretize 256 bins → embed in text prompt
  - Action output: MEAN_STD unnormalize with mean/std stats

Key differences from SmolVLA wrapper:
  - VLM backbone: PaliGemma 2B (hidden_size=2048) vs Idefics3 (960D)
  - State: embedded in text prompt (not as separate tensor)
  - Normalization: MEAN_STD
  - KV cache: HuggingFace DynamicCache (deep-copied per denoise step internally)
"""
import copy
import math
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from safetensors.torch import load_file

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    make_att_2d_masks,
    resize_with_pad_torch,
)

try:
    from utils import FULL_ACTION_DIM, CORRECTION_DIM
except ImportError:
    from .utils import FULL_ACTION_DIM, CORRECTION_DIM

NORM_EPS = 1e-8


def split_action(action_7d):
    """Split LIBERO 7D action into 6D pose and 1D gripper."""
    return action_7d[..., :CORRECTION_DIM], action_7d[..., CORRECTION_DIM:FULL_ACTION_DIM]


class Pi05Wrapper:
    """Frozen π0.5 for obs_features and K-sample generation."""

    VLM_HIDDEN_SIZE = 2048

    def __init__(self, pretrained_path: str = "./models/pi05",
                 device: str = "cuda", n_action_steps: int = 10, dtype: str = "auto"):
        self.device = torch.device(device)
        self.n_action_steps = n_action_steps
        # Open-loop action chunk queue. PI05 emits chunk_size=50 actions per
        # inference call; replaying them open-loop (1 inference per 50 env
        # steps) is the official lerobot contract. Re-running inference each
        # env step and only consuming action[0] starves multi-step behaviors
        # like the gripper-close at chunk step ~20, which produces 0% SR.
        self._action_queue = deque()
        self.policy = PI05Policy.from_pretrained(pretrained_path, strict=False)
        self._fix_vision_tower_keys(pretrained_path)
        self.policy.to(self.device)
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if dtype in _dtype_map:
            self.policy = self.policy.to(dtype=_dtype_map[dtype])
            print(f"[pi05] Model cast to {dtype}")
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        self.model = self.policy.model  # PI05Pytorch
        self.config = self.policy.config
        self.vlm_hidden_size = self.VLM_HIDDEN_SIZE

        # Tokenizer (PaliGemma / Gemma — same SentencePiece vocab)
        from transformers import AutoTokenizer
        _tok_candidates = [
            "google/paligemma-3b-pt-224",
            "./models/gemma-2b-tokenizer",
            "unsloth/gemma-2b",
        ]
        self.tokenizer = None
        for _name in _tok_candidates:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    _name, local_files_only=(_name == _tok_candidates[0])
                )
                break
            except Exception:
                continue
        if self.tokenizer is None:
            raise RuntimeError(f"Cannot load tokenizer from any of {_tok_candidates}")
        self.tokenizer.padding_side = "right"

        self._to_tensor = T.ToTensor()

        # Load normalizer stats (MEAN_STD)
        model_dir = Path(pretrained_path)
        self._load_normalizer_stats(model_dir)


    def _fix_vision_tower_keys(self, pretrained_path: str):
        """Fix vision tower key mismatch between saved weights and model architecture.
        
        Saved weights use: .vision_tower.vision_model.X
        Model expects:     .vision_tower.X  (transformers >= 5.x)
        """
        sd = load_file(str(Path(pretrained_path) / "model.safetensors"))
        remap = {}
        for k, v in sd.items():
            if "vision_tower.vision_model." in k:
                new_k = k.replace("vision_tower.vision_model.", "vision_tower.")
                if not new_k.startswith("model."):
                    new_k = f"model.{new_k}"
                remap[new_k] = v
        if remap:
            missing, unexpected = self.policy.load_state_dict(remap, strict=False)
            loaded = len(remap) - len(unexpected)
            print(f"[pi05] Fixed {loaded} vision tower keys")
            if missing:
                print(f"[pi05] Vision tower missing_keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                print(f"[pi05] Vision tower unexpected_keys ({len(unexpected)}): {unexpected[:5]}")

    def _load_normalizer_stats(self, model_dir: Path):
        """Load MEAN_STD normalization stats (mean, std) from model directory."""
        pre_files = sorted(model_dir.glob("*normalizer*processor*.safetensors"))
        post_files = sorted(model_dir.glob("*unnormalizer*processor*.safetensors"))

        if pre_files:
            pre_stats = load_file(str(pre_files[0]))
            self.state_mean = pre_stats.get(
                "observation.state.mean", torch.zeros(32)
            ).to(self.device)
            self.state_std = pre_stats.get(
                "observation.state.std", torch.ones(32)
            ).to(self.device)
        else:
            self.state_mean = torch.zeros(32, device=self.device)
            self.state_std = torch.ones(32, device=self.device)

        if post_files:
            post_stats = load_file(str(post_files[0]))
            self.action_mean = post_stats.get(
                "action.mean", torch.zeros(32)
            ).to(self.device)
            self.action_std = post_stats.get(
                "action.std", torch.ones(32)
            ).to(self.device)
        else:
            self.action_mean = torch.zeros(32, device=self.device)
            self.action_std = torch.ones(32, device=self.device)

    def _preprocess_image(self, img_np: np.ndarray) -> torch.Tensor:
        """HWC uint8 numpy → (1, C, H, W) float tensor ready for PI05."""
        img = self._to_tensor(img_np)  # CHW float [0,1]
        img = img.unsqueeze(0).to(self.device)
        img = torch.flip(img, dims=[2, 3])
        h, w = self.config.image_resolution
        img = resize_with_pad_torch(img, h, w)
        img = img * 2.0 - 1.0  # [0,1] -> [-1,1] for SigLIP
        return img

    def _normalize_state(self, state_raw: torch.Tensor) -> torch.Tensor:
        """MEAN_STD normalize state."""
        D = state_raw.shape[-1]
        mean = self.state_mean[:D]
        std = self.state_std[:D]
        denom = std + NORM_EPS
        return (state_raw - mean) / denom

    def _unnormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """MEAN_STD unnormalize action to original scale."""
        mean = self.action_mean[:FULL_ACTION_DIM]
        std = self.action_std[:FULL_ACTION_DIM]
        return action_norm[..., :FULL_ACTION_DIM] * std + mean

    @torch.no_grad()
    def preprocess_obs(self, agentview_rgb: np.ndarray,
                       task_name: str,
                       robot_state: np.ndarray,
                       eye_in_hand_rgb: np.ndarray | None = None):
        """
        Convert raw LIBERO observation into PI05-ready tensors.

        Args:
            agentview_rgb: (H, W, 3) uint8
            task_name: task description string
            robot_state: (8,) float — eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
            eye_in_hand_rgb: optional (H, W, 3) uint8
        Returns:
            images: list of (1, C, H, W) float tensors
            img_masks: list of (1,) bool tensors
            lang_tokens: (1, L) long tensor
            lang_masks: (1, L) bool tensor
        """
        dev = self.device

        imgs = [self._preprocess_image(agentview_rgb)]
        img_masks = [torch.ones(1, dtype=torch.bool, device=dev)]
        if eye_in_hand_rgb is not None:
            imgs.append(self._preprocess_image(eye_in_hand_rgb))
            img_masks.append(torch.ones(1, dtype=torch.bool, device=dev))

        # Add empty camera (model trained with empty_cameras=1)
        n_empty = getattr(self.config, 'empty_cameras', 0)
        for _ in range(n_empty):
            empty_img = torch.ones_like(imgs[0]) * -1
            imgs.append(empty_img)
            img_masks.append(torch.zeros(1, dtype=torch.bool, device=dev))

        # State: MEAN_STD normalize → discretize 256 bins → text prompt
        state_t = torch.tensor(robot_state, dtype=torch.float32, device=dev)
        state_norm = self._normalize_state(state_t)
        state_np = state_norm.cpu().numpy()
        discretized = np.digitize(
            state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1

        state_str = " ".join(map(str, discretized))
        cleaned_text = task_name.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
        )
        lang_tokens = tokens["input_ids"].to(dev)
        lang_masks = tokens["attention_mask"].to(dev).bool()

        return imgs, img_masks, lang_tokens, lang_masks

    @torch.no_grad()
    def encode_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """
        Encode observation prefix through VLM, return obs_features and KV cache.

        Returns:
            obs_features: (B, 2048) — mean-pooled VLM hidden states
            past_key_values: DynamicCache for denoising
            prefix_pad_masks: (B, seq_len) for denoising attention
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )

        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

        [prefix_hidden, _], past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Mean-pool VLM hidden states → obs_features (2048D)
        mask = prefix_pad_masks.unsqueeze(-1).float()
        obs_features = (prefix_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)

        return obs_features, past_key_values, prefix_pad_masks

    @torch.no_grad()
    def generate_k_samples(self, past_key_values, prefix_pad_masks,
                           K: int = 10) -> torch.Tensor:
        """
        Generate K action samples using cached prefix KV.

        Each sample uses independent random noise. The KV cache is
        deep-copied inside PI05's denoise_step, preserving the original.

        Returns:
            samples: (B, K, chunk_size, FULL_ACTION_DIM) — unnormalized actions
        """
        B = prefix_pad_masks.shape[0]
        chunk_size = self.config.chunk_size
        max_act_dim = self.config.max_action_dim
        num_steps = self.config.num_inference_steps
        dt = -1.0 / num_steps

        all_samples = []
        for _ in range(K):
            noise = self.model.sample_noise(
                (B, chunk_size, max_act_dim), self.device
            )
            x_t = noise

            for step in range(num_steps):
                time_val = 1.0 + step * dt
                time_tensor = torch.full(
                    (B,), time_val, dtype=torch.float32, device=self.device
                )
                v_t = self.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=time_tensor,
                )
                x_t = x_t + dt * v_t

            raw_action = x_t[:, :, :FULL_ACTION_DIM]
            unnorm_action = self._unnormalize_action(raw_action)
            all_samples.append(unnorm_action)

        return torch.stack(all_samples, dim=1)  # (B, K, chunk_size, FULL_ACTION_DIM)

    @torch.no_grad()
    def get_obs_features_and_samples(self, agentview_rgb, task_name, robot_state,
                                     eye_in_hand_rgb=None, K=10):
        """
        End-to-end: raw obs → obs_features (2048D) + K action samples (7D).

        Returns:
            obs_features: (1, 2048)
            samples: (1, K, chunk_size, FULL_ACTION_DIM)
        """
        imgs, img_masks, lt, lm = self.preprocess_obs(
            agentview_rgb, task_name, robot_state, eye_in_hand_rgb
        )
        obs_feat, kv_cache, pad_masks = self.encode_prefix(
            imgs, img_masks, lt, lm
        )
        samples = self.generate_k_samples(kv_cache, pad_masks, K=K)
        return obs_feat, samples

    @torch.no_grad()
    def generate_adaptive_k_samples(self, past_key_values, prefix_pad_masks,
                                     threshold, k_init=2, k_full=5):
        """Adaptive K: generate k_init samples, expand to k_full if L2 disagreement >= threshold."""
        init_samples = self.generate_k_samples(
            past_key_values, prefix_pad_masks, K=k_init
        )
        B = init_samples.shape[0]

        if k_init == 2:
            diff = init_samples[:, 0, :, :FULL_ACTION_DIM] - init_samples[:, 1, :, :FULL_ACTION_DIM]
            per_step_l2 = diff.norm(dim=-1)
            mean_l2 = per_step_l2.mean(dim=-1)
        else:
            dists = []
            for i in range(k_init):
                for j in range(i + 1, k_init):
                    diff = init_samples[:, i, :, :FULL_ACTION_DIM] - init_samples[:, j, :, :FULL_ACTION_DIM]
                    dists.append(diff.norm(dim=-1).mean(dim=-1))
            mean_l2 = torch.stack(dists).mean(dim=0)

        needs_full = (mean_l2 >= threshold).any().item()
        l2_val = mean_l2.item() if B == 1 else mean_l2.cpu().tolist()

        if not needs_full:
            return init_samples, {
                'path': 'fast', 'actual_k': k_init, 'l2_distance': l2_val,
            }

        extra_samples = self.generate_k_samples(
            past_key_values, prefix_pad_masks, K=k_full - k_init
        )
        all_samples = torch.cat([init_samples, extra_samples], dim=1)
        return all_samples, {
            'path': 'full', 'actual_k': k_full, 'l2_distance': l2_val,
        }

    @torch.no_grad()
    def get_obs_features_and_samples_adaptive(self, agentview_rgb, task_name,
                                               robot_state, eye_in_hand_rgb=None,
                                               threshold=0.1, k_init=2, k_full=5):
        """End-to-end adaptive K: obs → features + adaptive samples + metadata."""
        imgs, img_masks, lt, lm = self.preprocess_obs(
            agentview_rgb, task_name, robot_state, eye_in_hand_rgb
        )
        obs_feat, kv_cache, pad_masks = self.encode_prefix(
            imgs, img_masks, lt, lm
        )
        samples, metadata = self.generate_adaptive_k_samples(
            kv_cache, pad_masks, threshold, k_init, k_full
        )
        return obs_feat, samples, metadata

    def reset_action_queue(self):
        """Clear cached chunk. Call at the start of every rollout (env.reset)."""
        self._action_queue.clear()

    @torch.no_grad()
    def predict_action(self, agentview_rgb, task_name, robot_state,
                       eye_in_hand_rgb=None, K: int = 1):
        """
        Return one (FULL_ACTION_DIM,) numpy action per call, replaying the
        cached chunk before re-inferring. When the queue is empty, runs one
        PI05 inference to obtain a chunk_size-step chunk (mean over K samples
        if K>1) and pushes every step into the queue.
        """
        if not self._action_queue:
            _obs_feat, samples = self.get_obs_features_and_samples(
                agentview_rgb=agentview_rgb,
                task_name=task_name,
                robot_state=robot_state,
                eye_in_hand_rgb=eye_in_hand_rgb,
                K=K,
            )
            # samples: (1, K, chunk_size, FULL_ACTION_DIM) -- average over K
            chunk = samples[0].mean(dim=0)
            chunk_np = chunk.detach().cpu().numpy()
            for i in range(min(self.n_action_steps, chunk_np.shape[0])):
                self._action_queue.append(chunk_np[i])
        return self._action_queue.popleft()
