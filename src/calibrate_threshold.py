"""
Calibrate adaptive K threshold by collecting pairwise L2 distances from K=10 samples.

Runs SmolVLA on N rollout steps from a LIBERO benchmark, generates K=10 action samples
per step, computes all pairwise L2 distances, and reports percentile statistics.

Usage:
  python calibrate_threshold.py --benchmark object --n_steps 200
  python calibrate_threshold.py --benchmark spatial --n_steps 500 --K 10
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
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_seed, FULL_ACTION_DIM, H_EFF


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate adaptive K threshold")
    p.add_argument("--benchmark", type=str, default="object",
                   choices=["object", "spatial", "goal", "long", "90"])
    p.add_argument("--n_steps", type=int, default=200,
                   help="Total steps to collect across rollouts")
    p.add_argument("--K", type=int, default=10,
                   help="Number of samples per step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smolvla_path", type=str,
                   default="./models/smolvla")
    p.add_argument("--max_steps_per_rollout", type=int, default=50,
                   help="Max steps per rollout before reset")
    p.add_argument("--output_file", type=str, default=None)
    return p.parse_args()


def compute_pairwise_l2(samples):
    """Compute all pairwise L2 distances between K samples.

    Args:
        samples: (K, chunk_size, action_dim)
    Returns:
        list of mean-L2 distances (one per pair)
    """
    K = samples.shape[0]
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = samples[i, :, :FULL_ACTION_DIM] - samples[j, :, :FULL_ACTION_DIM]
            per_step_l2 = diff.norm(dim=-1)
            mean_l2 = per_step_l2.mean().item()
            dists.append(mean_l2)
    return dists


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from smolvla_wrapper import SmolVLAWrapper
    print(f"[calibrate] Loading SmolVLA from {args.smolvla_path}...")
    wrapper = SmolVLAWrapper(args.smolvla_path, device=str(device))

    from evaluate import load_libero_env, get_robot_state
    print(f"[calibrate] Loading LIBERO {args.benchmark} environments...")
    task_envs = load_libero_env(args.benchmark)

    all_pairwise_dists = []
    steps_collected = 0

    for task_info in task_envs:
        if steps_collected >= args.n_steps:
            break

        task_desc = task_info["task_description"]
        env = task_info["env"]
        init_states = task_info["init_states"]

        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))

        for step in range(args.max_steps_per_rollout):
            if steps_collected >= args.n_steps:
                break

            agentview = obs["agentview_image"]
            eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
            robot_state = get_robot_state(obs)

            _, samples = wrapper.get_obs_features_and_samples(
                agentview_rgb=agentview,
                task_name=task_desc,
                robot_state=robot_state,
                eye_in_hand_rgb=eye_in_hand,
                K=args.K,
            )

            step_samples = samples[0]
            pairwise = compute_pairwise_l2(step_samples)
            all_pairwise_dists.extend(pairwise)
            steps_collected += 1

            mean_action = samples[0, :, :H_EFF, :FULL_ACTION_DIM].mean(dim=0)
            action_7d = mean_action[0].cpu().numpy()
            obs, _, done, info = env.step(action_7d)

            if done or info.get("success", False):
                env.reset()
                obs = env.set_init_state(init_states[step % len(init_states)])
                for _ in range(5):
                    obs, _, _, _ = env.step(np.zeros(7))

            if (steps_collected) % 50 == 0:
                print(f"  Collected {steps_collected}/{args.n_steps} steps")

        env.close()

    dists = np.array(all_pairwise_dists)

    percentiles = [10, 25, 50, 75, 90, 95]
    pct_values = {f"p{p}": float(np.percentile(dists, p)) for p in percentiles}

    print(f"\n{'='*60}")
    print(f"Pairwise L2 Distance Statistics (K={args.K}, {steps_collected} steps)")
    print(f"{'='*60}")
    print(f"  Mean:   {dists.mean():.4f}")
    print(f"  Std:    {dists.std():.4f}")
    print(f"  Min:    {dists.min():.4f}")
    print(f"  Max:    {dists.max():.4f}")
    for p in percentiles:
        print(f"  P{p:2d}:    {pct_values[f'p{p}']:.4f}")
    print(f"{'='*60}")
    print(f"\nRecommended thresholds:")
    print(f"  Conservative (more full paths): P25 = {pct_values['p25']:.4f}")
    print(f"  Balanced:                       P50 = {pct_values['p50']:.4f}")
    print(f"  Aggressive (more fast paths):   P75 = {pct_values['p75']:.4f}")

    result = {
        "benchmark": args.benchmark,
        "K": args.K,
        "n_steps": steps_collected,
        "n_pairs": len(dists),
        "mean": float(dists.mean()),
        "std": float(dists.std()),
        "min": float(dists.min()),
        "max": float(dists.max()),
        "percentiles": pct_values,
        "recommended": {
            "conservative": pct_values["p25"],
            "balanced": pct_values["p50"],
            "aggressive": pct_values["p75"],
        },
    }

    out_path = args.output_file or f"logs/calibrate_{args.benchmark}_K{args.K}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"\n[calibrate] Results saved to {out_path}")


if __name__ == "__main__":
    main()
