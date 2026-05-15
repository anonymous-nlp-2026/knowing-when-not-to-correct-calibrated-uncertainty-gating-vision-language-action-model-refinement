"""ACR-VLA end-to-end pipeline dimension verification."""
import time
import torch
import numpy as np

torch.set_grad_enabled(False)
device = torch.device("cuda:0")

print("=" * 60)
print("STEP 1: Loading SmolVLA")
print("=" * 60)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

t0 = time.time()
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.eval()
policy.to(device)
load_time = time.time() - t0

n_params = sum(p.numel() for p in policy.parameters()) / 1e6
vram_gb = torch.cuda.memory_allocated() / 1024**3

print(f"Params: {n_params:.1f}M")
print(f"VRAM: {vram_gb:.2f}GB")
print(f"Load time: {load_time:.1f}s")
print(f"chunk_size: {policy.config.chunk_size}")
print(f"num_steps (flow matching): {policy.config.num_steps}")
print(f"max_action_dim: {policy.config.max_action_dim}")
print(f"max_state_dim: {policy.config.max_state_dim}")

print("\n" + "=" * 60)
print("STEP 2: Architecture dimensions")
print("=" * 60)

vlm_hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
expert_hidden = policy.model.vlm_with_expert.expert_hidden_size
num_vlm_layers = policy.model.vlm_with_expert.num_vlm_layers
num_expert_layers = policy.model.vlm_with_expert.num_expert_layers

print(f"VLM text hidden_size: {vlm_hidden}")
print(f"Expert hidden_size: {expert_hidden}")
print(f"VLM layers: {num_vlm_layers}")
print(f"Expert layers: {num_expert_layers}")
print(f"action_in_proj: {policy.model.action_in_proj.in_features} -> {policy.model.action_in_proj.out_features}")
print(f"action_out_proj: {policy.model.action_out_proj.in_features} -> {policy.model.action_out_proj.out_features}")
print(f"state_proj: {policy.model.state_proj.in_features} -> {policy.model.state_proj.out_features}")

print(f"\nInput features:")
for k, v in policy.config.input_features.items():
    print(f"  {k}: type={v.type}, shape={v.shape}")
print(f"Output features:")
for k, v in policy.config.output_features.items():
    print(f"  {k}: type={v.type}, shape={v.shape}")

action_dim = policy.config.output_features["action"].shape[0]
print(f"\n>>> action_dim = {action_dim}")

print("\n" + "=" * 60)
print("STEP 3: Single inference (K=1)")
print("=" * 60)

batch = {}
for k, v in policy.config.input_features.items():
    if v.type.name == "VISUAL":
        batch[k] = torch.randn(1, *v.shape, device=device)
    elif v.type.name == "STATE":
        batch[k] = torch.randn(1, *v.shape, device=device)

tokenizer = policy.model.vlm_with_expert.processor.tokenizer
dummy_text = "pick up the red cube\n"
tokens = tokenizer(
    dummy_text,
    return_tensors="pt",
    padding="max_length",
    max_length=policy.config.tokenizer_max_length,
    truncation=True,
)
batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to(device)
batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool().to(device)

print("Input batch shapes:")
for k, v in batch.items():
    print(f"  {k}: {v.shape} ({v.dtype})")

# Warmup
policy.reset()
_ = policy._get_action_chunk(batch)
torch.cuda.synchronize()

# Timed K=1 inference
torch.cuda.synchronize()
t0 = time.time()
actions_single = policy._get_action_chunk(batch)
torch.cuda.synchronize()
t1 = time.time()

print(f"\nSingle inference output: {actions_single.shape}")
print(f"  = (batch={actions_single.shape[0]}, chunk_size={actions_single.shape[1]}, action_dim={actions_single.shape[2]})")
print(f"Time (K=1): {(t1-t0)*1000:.1f}ms")

print("\n" + "=" * 60)
print("STEP 4: K=10 sampling (with KV cache reuse)")
print("=" * 60)

K = 10
H_eff = 5
D = action_dim

images, img_masks = policy.prepare_images(batch)
state = policy.prepare_state(batch)
lang_tokens = batch[OBS_LANGUAGE_TOKENS]
lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

# Compute prefix KV cache once
prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
    images, img_masks, lang_tokens, lang_masks, state=state
)
prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

_, past_key_values = policy.model.vlm_with_expert.forward(
    attention_mask=prefix_att_2d_masks,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=policy.config.use_cache,
    fill_kv_cache=True,
)

print(f"Prefix computed. KV cache entries: {len(past_key_values)}")
print(f"Prefix sequence length: {prefix_pad_masks.shape[1]}")
print(f"prefix_embs shape: {prefix_embs.shape}")
print(f"  -> obs_features dim (VLM hidden): {prefix_embs.shape[-1]}")

# K denoising passes
num_steps = policy.config.num_steps
dt = -1.0 / num_steps
all_samples = []

torch.cuda.synchronize()
t_start = time.time()

for k_idx in range(K):
    actions_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    noise = policy.model.sample_noise(actions_shape, device)
    x_t = noise

    for step in range(num_steps):
        t_val = 1.0 + step * dt
        time_tensor = torch.tensor(t_val, dtype=torch.float32, device=device).expand(1)
        v_t = policy.model.denoise_step(
            x_t=x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time_tensor,
        )
        x_t = x_t + dt * v_t

    x_t = x_t[:, :, :action_dim]
    all_samples.append(x_t)

torch.cuda.synchronize()
t_end = time.time()

samples_tensor = torch.cat(all_samples, dim=0)  # (K, chunk_size, action_dim)
print(f"\nK={K} samples tensor: {samples_tensor.shape}")
print(f"  = (K={samples_tensor.shape[0]}, chunk_size={samples_tensor.shape[1]}, action_dim={samples_tensor.shape[2]})")
print(f"Time (K={K} with cache reuse): {(t_end-t_start)*1000:.1f}ms")
print(f"Time per sample: {(t_end-t_start)/K*1000:.1f}ms")

# Also time K=10 without cache reuse for comparison
torch.cuda.synchronize()
t_no_cache_start = time.time()
for k_idx in range(K):
    _ = policy._get_action_chunk(batch)
torch.cuda.synchronize()
t_no_cache_end = time.time()
print(f"Time (K={K} without cache reuse): {(t_no_cache_end-t_no_cache_start)*1000:.1f}ms")

samples_trimmed = samples_tensor[:, :H_eff, :]  # (K, H_eff, D)
print(f"\nTrimmed to H_eff={H_eff}: {samples_trimmed.shape}")

print("\n" + "=" * 60)
print("STEP 5: CGS feature computation & dimension verification")
print("=" * 60)

from sklearn.covariance import LedoitWolf

samples_np = samples_trimmed.cpu().float().numpy()  # (K, H_eff, D)

per_dof_stds_list = []
conformal_radii = []
log_volumes = []

for h in range(H_eff):
    step_samples = samples_np[:, h, :]  # (K, D)
    lw = LedoitWolf().fit(step_samples)
    cov = lw.covariance_  # (D, D)

    stds = np.sqrt(np.diag(cov))  # (D,)
    per_dof_stds_list.append(stds)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals_pos = np.maximum(eigvals, 1e-30)
    log_vol = 0.5 * np.sum(np.log(eigvals_pos))
    log_volumes.append(log_vol)

    mean = step_samples.mean(axis=0)
    precision = lw.precision_
    diffs = step_samples - mean
    mahal_dists = np.sqrt(np.sum(diffs @ precision * diffs, axis=1))
    conformal_radii.append(np.quantile(mahal_dists, 0.9))

per_dof_stds = np.concatenate(per_dof_stds_list)
conformal_radii = np.array(conformal_radii)
log_volumes = np.array(log_volumes)

print(f"per_dof_stds: {per_dof_stds.shape} (expected (30,))")
print(f"conformal_radii: {conformal_radii.shape} (expected (5,))")
print(f"log_volumes: {log_volumes.shape} (expected (5,))")

cgs_features = np.concatenate([per_dof_stds, conformal_radii, log_volumes])
print(f"CGS features total: {cgs_features.shape[0]}D (expected 40)")

mean_action = samples_trimmed.mean(dim=0).flatten().cpu().numpy()
print(f"mean_action: {mean_action.shape} (expected (30,))")

print("\n" + "=" * 60)
print("STEP 6: CRM input dimension & obs_features")
print("=" * 60)

obs_features_dim = vlm_hidden
cgs_dim = cgs_features.shape[0]
mean_action_dim = mean_action.shape[0]
crm_action_input_dim = cgs_dim + mean_action_dim

print(f"obs_features (VLM hidden): {obs_features_dim}D")
print(f"  NOTE: prefix_embs is (1, seq_len, {obs_features_dim})")
print(f"  For CRM, pool over seq_len to get ({obs_features_dim},) per sample")
print(f"CGS uncertainty features: {cgs_dim}D")
print(f"  - per_dof_stds: {per_dof_stds.shape[0]}D")
print(f"  - conformal_radii: {conformal_radii.shape[0]}D")
print(f"  - log_volumes: {log_volumes.shape[0]}D")
print(f"Mean action: {mean_action_dim}D")
print(f"CRM action+uncertainty input: {crm_action_input_dim}D (expected 70)")
print(f"CRM total input: {obs_features_dim} + {crm_action_input_dim} = {obs_features_dim + crm_action_input_dim}D")
print(f"CRM output: {mean_action_dim}D (refined action)")

print("\n" + "=" * 60)
print("STEP 7: Full pipeline shape flow")
print("=" * 60)
print(f"""
INPUT:
  image x3:      (1, 3, 256, 256) -> resized to {policy.config.resize_imgs_with_padding}
  instruction:   str -> tokenized to (1, {policy.config.tokenizer_max_length})
  state:         (1, 6) -> padded to (1, {policy.config.max_state_dim})

SmolVLA INTERNALS:
  VLM prefix:     (1, seq_len={prefix_pad_masks.shape[1]}, {vlm_hidden})
  KV cache:       {len(past_key_values)} entries (VLM {num_vlm_layers}L + Expert {num_expert_layers}L)
  Flow matching:  {policy.config.num_steps} denoising steps per sample
  Raw output:     (1, {policy.config.chunk_size}, {policy.config.max_action_dim}) -> unpadded to (1, {policy.config.chunk_size}, {action_dim})

K={K} SAMPLING:
  Prefix KV cache computed ONCE
  K={K} denoising passes with different noise
  samples:        ({K}, {policy.config.chunk_size}, {action_dim})
  trimmed H_eff:  ({K}, {H_eff}, {D})

CGS FEATURES (per observation):
  per_dof_stds:    ({H_eff}*{D},) = ({H_eff*D},)
  conformal_radii: ({H_eff},)
  log_volumes:     ({H_eff},)
  total CGS:       {cgs_dim}D

CRM INPUT:
  obs_features:    ({obs_features_dim},)  [pooled VLM prefix]
  mean_action:     ({mean_action_dim},)   [{H_eff}x{D} flattened]
  CGS features:    ({cgs_dim},)           [uncertainty]
  total CRM input: {obs_features_dim + crm_action_input_dim}D

CRM OUTPUT:
  refined_action:  ({mean_action_dim},) = ({H_eff} steps x {D} DoF)
""")

print("=" * 60)
print("DIMENSION CHECKS")
print("=" * 60)

checks = [
    ("action_dim", action_dim, 6),
    ("chunk_size", policy.config.chunk_size, 50),
    ("flow_matching_steps", policy.config.num_steps, 10),
    ("K samples shape[0]", samples_tensor.shape[0], K),
    ("trimmed shape", tuple(samples_trimmed.shape), (K, H_eff, D)),
    ("per_dof_stds dim", per_dof_stds.shape[0], H_eff * D),
    ("conformal_radii dim", conformal_radii.shape[0], H_eff),
    ("log_volumes dim", log_volumes.shape[0], H_eff),
    ("CGS total dim", cgs_dim, 40),
    ("mean_action dim", mean_action_dim, 30),
    ("CRM action+unc input", crm_action_input_dim, 70),
    ("VLM hidden (obs_features)", obs_features_dim, 960),
]

all_pass = True
for name, actual, expected in checks:
    status = "PASS" if actual == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  [{status}] {name}: {actual} (expected {expected})")

print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

print(f"\nTIMING:")
print(f"  Model load: {load_time:.1f}s")
print(f"  K=1 inference: {(t1-t0)*1000:.1f}ms")
print(f"  K={K} with cache reuse: {(t_end-t_start)*1000:.1f}ms ({(t_end-t_start)/K*1000:.1f}ms/sample)")
print(f"  K={K} without cache: {(t_no_cache_end-t_no_cache_start)*1000:.1f}ms ({(t_no_cache_end-t_no_cache_start)/K*1000:.1f}ms/sample)")
print(f"  Cache reuse speedup: {(t_no_cache_end-t_no_cache_start)/(t_end-t_start):.2f}x")
print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
