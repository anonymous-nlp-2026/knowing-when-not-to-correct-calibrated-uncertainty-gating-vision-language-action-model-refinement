"""
LIBERO evaluation script for ACR-VLA.

Rollout loop:
  1. Load LIBERO env + task suite
  2. Load SmolVLA (frozen) + CRM
  3. For each task, rollout N times:
     a. Reset env
     b. Each step: obs → SmolVLA K=10 samples → CGS on 6D pose → CRM
        → combine refined pose with gripper → execute 7D action
     c. Record success/failure
  4. Report per-task and overall success rate

Baselines:
  - "vla_only": SmolVLA mean of K=10, no CRM refinement
  - "crm_scalar": CRM with scalar uncertainty (single std, not per-DOF)
  - "crm_perdof": CRM with full per-DOF uncertainty (ours)
  - "acr_vla": alias for crm_perdof — the full ACR-VLA pipeline
  - "reconvla": ReconVLA baseline — scalar Euclidean argmin selection (plan_014)
  - "adaptive_k": Adaptive K — K=2 L2 probe, expand to K=10 + CRM obs_only if uncertain

Adaptive K mode (--mode adaptive_k):
  Each step: generate K=2 samples, compute pairwise L2 distance.
  If L2 > threshold (high uncertainty): generate 8 more (total K=10),
    select argmin(L2 to mean), apply CRM obs_only correction (epsilon=0.2).
  If L2 <= threshold (low uncertainty): use mean of K=2, skip CRM.

Usage:
  python evaluate.py --benchmark object --checkpoint /path/to/crm_best.pt --n_rollouts 20
  python evaluate.py --benchmark object --mode vla_only --n_rollouts 20
  python evaluate.py --benchmark spatial --mode reconvla --n_rollouts 20
  python evaluate.py --benchmark spatial --mode acr_vla --crm_checkpoint /path/to/crm_best.pt --n_rollouts 10
"""
import argparse
import functools
builtins_print = print
print = functools.partial(builtins_print, flush=True)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import json
import time
import signal
import traceback
from pathlib import Path

import torch
import numpy as np
from robosuite.utils.transform_utils import quat2axisangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgs import ConformilizedGaussianScoring
from crm import ConformalRefinementModule
from aci import AdaptiveConformalInference
from utils import set_seed, CORRECTION_DIM, FULL_ACTION_DIM, H_EFF, OBS_DIM
from reconvla_baseline import reconvla_select


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ACR-VLA on LIBERO")
    p.add_argument("--benchmark", type=str, default="object",
                   choices=["object", "spatial", "goal", "long", "90"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to CRM checkpoint (crm_best.pt)")
    p.add_argument("--crm_checkpoint", type=str, default=None,
                   help="Alias for --checkpoint (used with --mode acr_vla)")
    p.add_argument("--n_rollouts", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=300,
                   help="Max steps per rollout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--mode", type=str, default="crm_perdof",
                   choices=["vla_only", "crm_scalar", "crm_perdof", "acr_vla", "crm_obs_only", "reconvla", "random_selection", "adaptive_k"])
    p.add_argument("--baseline_mode", type=str, default="single",
                   choices=["single", "mean"],
                   help="vla_only baseline: single (K=1) or mean (K=10 avg)")
    p.add_argument("--smolvla_path", type=str, default="/root/autodl-tmp/models/smolvla_libero/")
    p.add_argument("--obs_dim", type=int, default=OBS_DIM)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--render", action="store_true")
    p.add_argument("--output_file", type=str, default=None,
                   help="Save results JSON to this path")
    p.add_argument('--max_correction_norm', type=float, default=None,
                   help='Override CRM max_correction_norm (default: use checkpoint value)')
    p.add_argument("--adaptive_k", action="store_true",
                   help="Enable adaptive K sampling (start with k_init, expand to k_full if needed)")
    p.add_argument("--adaptive_threshold", type=float, default=0.1,
                   help="L2 distance threshold for adaptive K branching")
    p.add_argument("--k_init", type=int, default=2,
                   help="Initial sample count for adaptive K")
    p.add_argument("--k_full", type=int, default=5,
                   help="Full sample count when adaptive K expands")
    return p.parse_args()


def get_robot_state(obs):
    """Extract 8D robot state: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)."""
    eef_pos = obs["robot0_eef_pos"]
    eef_axisangle = quat2axisangle(obs["robot0_eef_quat"])
    gripper_qpos = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float64)


def load_libero_env(benchmark_name):
    """Load LIBERO benchmark tasks and create environments."""
    from libero.libero import benchmark as bm
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "long": "libero_10",
        "90": "libero_90",
    }
    bm_name = benchmark_dict[benchmark_name]
    benchmark_instance = bm.get_benchmark(bm_name)()
    n_tasks = benchmark_instance.n_tasks

    task_envs = []
    for task_idx in range(n_tasks):
        task = benchmark_instance.get_task(task_idx)
        task_name = task.name
        task_description = task.language
        task_bddl = benchmark_instance.get_task_bddl_file_path(task_idx)

        env_args = {
            "bddl_file_name": task_bddl,
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "render_gpu_device_id": int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(42)

        task_envs.append({
            "task_name": task_name,
            "task_description": task_description,
            "env": env,
            "init_states": benchmark_instance.get_task_init_states(task_idx),
        })

    return task_envs


def select_action_vla_only(wrapper, obs, task_name, K, cgs):
    """Baseline: mean of K samples (full 7D), no CRM."""
    agentview = obs["agentview_image"]
    eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
    robot_state = get_robot_state(obs)

    obs_feat, samples = wrapper.get_obs_features_and_samples(
        agentview_rgb=agentview,
        task_name=task_name,
        robot_state=robot_state,
        eye_in_hand_rgb=eye_in_hand,
        K=K,
    )
    # samples: (1, K, chunk_size, FULL_ACTION_DIM)
    k_trunc = samples[0, :, :H_EFF, :]  # (K, H_eff, 7)
    mean_action = k_trunc.mean(dim=0)  # (H_eff, 7)
    return mean_action[0].cpu().numpy()


def _to_scalar_uf(uf):
    return torch.stack([
        uf[:, :30].norm(dim=1),
        uf[:, 30:35].mean(dim=1),
        uf[:, 35:40].mean(dim=1),
    ], dim=1)


def select_action_crm(wrapper, obs, task_name, K, cgs, crm, device, mode="crm_perdof"):
    """ACR-VLA: CGS on 6D pose → argmin base → CRM refinement → combine with mean gripper."""
    agentview = obs["agentview_image"]
    eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
    robot_state = get_robot_state(obs)

    obs_feat, samples = wrapper.get_obs_features_and_samples(
        agentview_rgb=agentview,
        task_name=task_name,
        robot_state=robot_state,
        eye_in_hand_rgb=eye_in_hand,
        K=K,
    )
    k_full = samples[0, :, :H_EFF, :]  # (K, H_eff, 7)
    k_pose = k_full[:, :, :CORRECTION_DIM]  # (K, H_eff, 6)

    uf, _mean = cgs.compute_features(k_pose)
    uf = uf.unsqueeze(0).to(device)
    if mode == "crm_scalar":
        uf = _to_scalar_uf(uf)

    # Argmin-selected base action instead of mean (avoids invalid inter-mode regions)
    k_pose_flat = k_pose.reshape(k_pose.shape[0], -1)  # (K, 30)
    mean_pose = k_pose_flat.mean(dim=0)  # (30,) reference for distance
    distances = ((k_pose_flat - mean_pose) ** 2).sum(dim=-1)  # (K,)
    best_idx = distances.argmin()
    base_action = k_pose_flat[best_idx].unsqueeze(0).to(device)  # (1, 30)

    obs_feat = obs_feat.to(device)

    uf_input = None if mode == "crm_obs_only" else uf
    with torch.no_grad():
        refined_pose = crm(base_action, uf_input, obs_feat)  # (1, 30)

    refined_pose = refined_pose[0].reshape(H_EFF, CORRECTION_DIM)
    mean_gripper = k_full[:, :, CORRECTION_DIM:].mean(dim=0)  # (H_eff, 1)
    refined_full = torch.cat([refined_pose, mean_gripper], dim=-1)  # (H_eff, 7)

    return refined_full[0].cpu().numpy()


def select_action_reconvla(wrapper, obs, task_name, K):
    """ReconVLA baseline: scalar Euclidean argmin on 6D pose, pass-through gripper."""
    agentview = obs["agentview_image"]
    eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
    robot_state = get_robot_state(obs)

    obs_feat, samples = wrapper.get_obs_features_and_samples(
        agentview_rgb=agentview,
        task_name=task_name,
        robot_state=robot_state,
        eye_in_hand_rgb=eye_in_hand,
        K=K,
    )
    k_full = samples[0, :, :H_EFF, :]  # (K, H_eff, 7)
    k_pose = k_full[:, :, :CORRECTION_DIM]  # (K, H_eff, 6)
    selected_pose = reconvla_select(k_pose)  # (H_eff, 6)
    mean_gripper = k_full[:, :, CORRECTION_DIM:].mean(dim=0)  # (H_eff, 1)
    selected_full = torch.cat([selected_pose, mean_gripper], dim=-1)
    return selected_full[0].cpu().numpy()


def select_action_random(wrapper, obs, task_name, K):
    """Random selection baseline: pick one of K candidates uniformly at random."""
    agentview = obs["agentview_image"]
    eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
    robot_state = get_robot_state(obs)

    obs_feat, samples = wrapper.get_obs_features_and_samples(
        agentview_rgb=agentview,
        task_name=task_name,
        robot_state=robot_state,
        eye_in_hand_rgb=eye_in_hand,
        K=K,
    )
    k_full = samples[0, :, :H_EFF, :]  # (K, H_eff, 7)
    idx = np.random.randint(0, k_full.shape[0])
    return k_full[idx, 0].cpu().numpy()



def select_action_crm_adaptive(wrapper, obs, task_name, cgs_map, crm, device, mode,
                                threshold, k_init, k_full, step_metadata):
    """ACR-VLA with adaptive K sampling."""
    agentview = obs["agentview_image"]
    eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
    robot_state = get_robot_state(obs)

    obs_feat, samples, meta = wrapper.get_obs_features_and_samples_adaptive(
        agentview_rgb=agentview,
        task_name=task_name,
        robot_state=robot_state,
        eye_in_hand_rgb=eye_in_hand,
        threshold=threshold,
        k_init=k_init,
        k_full=k_full,
    )
    step_metadata.append(meta)

    actual_k = meta['actual_k']
    k_actions = samples[0, :, :H_EFF, :]
    k_pose = k_actions[:, :, :CORRECTION_DIM]

    cgs = cgs_map[actual_k]
    uf, _mean = cgs.compute_features(k_pose)
    uf = uf.unsqueeze(0).to(device)
    if mode == "crm_scalar":
        uf = _to_scalar_uf(uf)

    k_pose_flat = k_pose.reshape(k_pose.shape[0], -1)
    mean_pose = k_pose_flat.mean(dim=0)
    distances = ((k_pose_flat - mean_pose) ** 2).sum(dim=-1)
    best_idx = distances.argmin()
    base_action = k_pose_flat[best_idx].unsqueeze(0).to(device)

    obs_feat = obs_feat.to(device)
    uf_input = None if mode == "crm_obs_only" else uf
    with torch.no_grad():
        refined_pose = crm(base_action, uf_input, obs_feat)

    refined_pose = refined_pose[0].reshape(H_EFF, CORRECTION_DIM)
    mean_gripper = k_actions[:, :, CORRECTION_DIM:].mean(dim=0)
    refined_full = torch.cat([refined_pose, mean_gripper], dim=-1)
    return refined_full[0].cpu().numpy()



def select_action_adaptive_k(wrapper, obs, task_name, crm, device,
                              threshold, step_metadata):
    """Adaptive K: K=2 L2 probe, conditional K=10 expansion + CRM obs_only.

    Fast path (L2 <= threshold): mean of K=2, no CRM.
    Full path (L2 > threshold): expand to K=10, argmin + CRM obs_only (epsilon=0.2).
    """
    agentview = obs["agentview_image"]
    eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
    robot_state = get_robot_state(obs)

    obs_feat, samples, meta = wrapper.get_obs_features_and_samples_adaptive(
        agentview_rgb=agentview,
        task_name=task_name,
        robot_state=robot_state,
        eye_in_hand_rgb=eye_in_hand,
        threshold=threshold,
        k_init=2,
        k_full=10,
    )
    step_metadata.append(meta)

    k_all = samples[0, :, :H_EFF, :]  # (actual_k, H_eff, 7)

    if meta["path"] == "fast":
        # L2 <= threshold: mean of K=2, skip CRM
        mean_action = k_all.mean(dim=0)  # (H_eff, 7)
        return mean_action[0].cpu().numpy()

    # L2 > threshold: K=10, argmin(L2 to mean) on 6D pose, then CRM obs_only
    k_pose = k_all[:, :, :CORRECTION_DIM]  # (10, H_eff, 6)
    k_pose_flat = k_pose.reshape(k_pose.shape[0], -1)  # (10, 30)
    mean_pose = k_pose_flat.mean(dim=0)
    distances = ((k_pose_flat - mean_pose) ** 2).sum(dim=-1)
    best_idx = distances.argmin()
    base_action = k_pose_flat[best_idx].unsqueeze(0).to(device)  # (1, 30)

    obs_feat = obs_feat.to(device)
    # CRM obs_only: uf=None
    with torch.no_grad():
        refined_pose = crm(base_action, None, obs_feat)

    refined_pose = refined_pose[0].reshape(H_EFF, CORRECTION_DIM)
    mean_gripper = k_all[:, :, CORRECTION_DIM:].mean(dim=0)
    refined_full = torch.cat([refined_pose, mean_gripper], dim=-1)
    return refined_full[0].cpu().numpy()

def rollout_single(env, init_state, action_fn, max_steps=300):
    """Run one rollout. action_fn returns 7D action directly."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    for step in range(max_steps):
        try:
            action_7d = action_fn(obs)
            obs, reward, done, info = env.step(action_7d)
        except Exception as e:
            print(f"[rollout] Exception at step {step}: {e}", flush=True)
            traceback.print_exc()
            return False, step

        if done or info.get("success", False):
            return True, step + 1

    return False, max_steps


def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --crm_checkpoint is an alias for --checkpoint
    if args.crm_checkpoint and not args.checkpoint:
        args.checkpoint = args.crm_checkpoint

    # Default checkpoint for adaptive_k mode
    if args.mode == "adaptive_k" and args.checkpoint is None:
        args.checkpoint = "/root/autodl-tmp/checkpoints/crm_v3_norm/obs_only/crm_best.pt"
        print(f"[eval] adaptive_k: using default CRM checkpoint {args.checkpoint}")

    print(f"[eval] Mode: {args.mode}, benchmark: {args.benchmark}")

    # Load SmolVLA
    from smolvla_wrapper import SmolVLAWrapper
    print(f"[eval] Loading SmolVLA from {args.smolvla_path}...")
    wrapper = SmolVLAWrapper(args.smolvla_path, device=str(device))

    # CGS — operates on 6D pose only (CORRECTION_DIM)
    cgs = ConformilizedGaussianScoring(K=args.K, horizon=H_EFF, action_dim=CORRECTION_DIM)

    all_adaptive_meta = []
    cgs_map = {}
    if getattr(args, "adaptive_k", False):
        cgs_map[args.k_init] = ConformilizedGaussianScoring(
            K=args.k_init, horizon=H_EFF, action_dim=CORRECTION_DIM)
        cgs_map[args.k_full] = ConformilizedGaussianScoring(
            K=args.k_full, horizon=H_EFF, action_dim=CORRECTION_DIM)
        print(f"[eval] Adaptive K: k_init={args.k_init}, k_full={args.k_full}, "
              f"threshold={args.adaptive_threshold}")

    # CRM (if needed)
    crm = None
    if args.mode in ("crm_scalar", "crm_perdof", "acr_vla", "crm_obs_only", "adaptive_k"):
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for CRM modes")

        if args.mode in ("crm_obs_only", "adaptive_k"):
            unc_dim = 0
        elif args.mode == "crm_scalar":
            unc_dim = 3
        else:  # crm_perdof or acr_vla
            unc_dim = CORRECTION_DIM * H_EFF + H_EFF + H_EFF  # 40

        crm = ConformalRefinementModule(
            action_dim=H_EFF * CORRECTION_DIM,
            uncertainty_dim=unc_dim,
            obs_dim=args.obs_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        crm.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                        weights_only=True))
        crm.eval()
        print(f"[eval] CRM loaded from {args.checkpoint} "
              f"({crm.count_parameters():,} params)")

    # Default max_correction_norm=0.2 for adaptive_k mode
    if args.mode == "adaptive_k" and args.max_correction_norm is None:
        args.max_correction_norm = 0.2

    # Override max_correction_norm if specified
    if args.max_correction_norm is not None and crm is not None:
        crm.max_correction_norm = args.max_correction_norm
        print(f"[eval] Override max_correction_norm = {args.max_correction_norm}")

    # ACI for online alpha tracking
    aci = AdaptiveConformalInference()

    # Load LIBERO envs
    print(f"[eval] Loading LIBERO {args.benchmark} environments...")
    task_envs = load_libero_env(args.benchmark)
    print(f"[eval] {len(task_envs)} tasks loaded.")

    # Rollout
    results = {}
    overall_successes = 0
    overall_total = 0

    for task_info in task_envs:
        task_name = task_info["task_name"]
        task_desc = task_info["task_description"]
        env = task_info["env"]
        init_states = task_info["init_states"]

        aci.reset()
        task_successes = 0

        for r in range(args.n_rollouts):
            init_state = init_states[r % len(init_states)]

            rollout_meta = []

            if args.mode == "adaptive_k":
                action_fn = lambda obs: select_action_adaptive_k(
                    wrapper, obs, task_desc, crm, device,
                    args.adaptive_threshold, rollout_meta
                )
            elif getattr(args, "adaptive_k", False) and args.mode not in ("vla_only", "reconvla", "random_selection"):
                action_fn = lambda obs: select_action_crm_adaptive(
                    wrapper, obs, task_desc, cgs_map, crm, device, args.mode,
                    args.adaptive_threshold, args.k_init, args.k_full, rollout_meta
                )
            elif args.mode == "vla_only":
                action_fn = lambda obs: select_action_vla_only(
                    wrapper, obs, task_desc, args.K, cgs
                )
            elif args.mode == "reconvla":
                action_fn = lambda obs: select_action_reconvla(
                    wrapper, obs, task_desc, args.K
                )
            elif args.mode == "random_selection":
                action_fn = lambda obs: select_action_random(
                    wrapper, obs, task_desc, args.K
                )
            else:
                action_fn = lambda obs: select_action_crm(
                    wrapper, obs, task_desc, args.K, cgs, crm, device, args.mode
                )

            success, steps = rollout_single(env, init_state, action_fn,
                                            max_steps=args.max_steps)
            if rollout_meta:
                all_adaptive_meta.extend(rollout_meta)
            task_successes += int(success)
            overall_successes += int(success)
            overall_total += 1

            if (r + 1) % 5 == 0:
                print(f"  {task_name}: rollout {r+1}/{args.n_rollouts}, "
                      f"success so far: {task_successes}/{r+1}")

        task_rate = task_successes / args.n_rollouts
        results[task_name] = {
            "success_rate": task_rate,
            "successes": task_successes,
            "total": args.n_rollouts,
        }
        print(f"[{task_name}] success rate: {task_rate:.1%} "
              f"({task_successes}/{args.n_rollouts})")

        env.close()

    overall_rate = overall_successes / max(overall_total, 1)

    print(f"\n{'='*60}")
    print(f"Overall success rate ({args.mode}): {overall_rate:.1%} "
          f"({overall_successes}/{overall_total})")
    print(f"{'='*60}")

    # Save results
    summary = {
        "mode": args.mode,
        "baseline_mode": getattr(args, "baseline_mode", "single"),
        "benchmark": args.benchmark,
        "n_rollouts": args.n_rollouts,
        "overall_success_rate": overall_rate,
        "per_task": results,
    }

    if all_adaptive_meta:
        fast_count = sum(1 for m in all_adaptive_meta if m["path"] == "fast")
        total_steps = len(all_adaptive_meta)
        per_step_k = [m["actual_k"] for m in all_adaptive_meta]

        if args.mode == "adaptive_k":
            summary["per_step_K"] = per_step_k
            summary["avg_K"] = float(np.mean(per_step_k))
            summary["adaptive_trigger_rate"] = sum(1 for k in per_step_k if k == 10) / max(total_steps, 1)
            summary["threshold"] = args.adaptive_threshold

        summary["adaptive_stats"] = {
            "fast_path_ratio": fast_count / max(total_steps, 1),
            "avg_actual_k": float(np.mean(per_step_k)),
            "mean_l2_distance": float(np.mean([m["l2_distance"] for m in all_adaptive_meta])),
            "total_steps": total_steps,
            "fast_steps": fast_count,
            "full_steps": total_steps - fast_count,
        }
        print(f"[eval] Adaptive stats: "
              f"fast_path={summary['adaptive_stats']['fast_path_ratio']:.1%}, "
              f"avg_k={summary['adaptive_stats']['avg_actual_k']:.1f}, "
              f"mean_l2={summary['adaptive_stats']['mean_l2_distance']:.4f}")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[eval] Results saved to {args.output_file}")

    return summary


def main():
    def _segfault_handler(signum, frame):
        print(f"[evaluate] FATAL: caught signal {signum} (segfault)", flush=True)
        sys.exit(139)

    signal.signal(signal.SIGSEGV, _segfault_handler)

    args = parse_args()
    if args.mode == "vla_only":
        if args.baseline_mode == "single":
            args.K = 1
        elif args.baseline_mode == "mean":
            args.K = 10
    if args.output_file is None:
        norm_tag = f"_norm{args.max_correction_norm}" if args.max_correction_norm is not None else ""
        if args.mode == "adaptive_k":
            adaptive_tag = f"_t{args.adaptive_threshold}"
        elif getattr(args, "adaptive_k", False):
            adaptive_tag = f"_adaptiveK{args.k_init}_{args.k_full}_t{args.adaptive_threshold}"
        else:
            adaptive_tag = ""
        args.output_file = f"logs/eval_{args.benchmark}_{args.mode}{norm_tag}{adaptive_tag}_seed{args.seed}.json"
    print(f"[evaluate] args: {vars(args)}", flush=True)
    evaluate(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
