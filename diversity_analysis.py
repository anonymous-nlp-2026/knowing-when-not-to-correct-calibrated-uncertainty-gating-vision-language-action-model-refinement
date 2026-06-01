"""
Diversity analysis: compare K-sample diversity between SmolVLA (flow matching)
and OpenVLA-OFT (deterministic L1 + proprio noise).

Uses evaluate.py's load_libero_env for correct environment setup.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from evaluate import load_libero_env
from robosuite.utils.transform_utils import quat2axisangle


def run_diversity_analysis(wrapper, task_envs, K=2, n_rollouts=2, max_steps=50, seed=42):
    """Run episodes and collect K-sample L2 distances."""
    all_l2 = []
    
    for task_info in task_envs:
        task_name = task_info["task_name"]
        task_desc = task_info["task_description"]
        env = task_info["env"]
        init_states = task_info["init_states"]
        
        for ep in range(n_rollouts):
            rng = np.random.RandomState(seed + ep)
            init_idx = rng.randint(len(init_states))
            env.reset()
            obs = env.set_init_state(init_states[init_idx])
            wrapper.reset_action_queue()
            
            step_l2s = []
            for step in range(max_steps):
                agentview = obs["agentview_image"]
                robot_state = np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
                eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
                
                obs_feat, samples = wrapper.get_obs_features_and_samples(
                    agentview, task_desc, robot_state,
                    eye_in_hand_rgb=eye_in_hand, K=K
                )
                # samples: (1, K, chunk_size, action_dim)
                s = samples.squeeze(0)  # (K, chunk, action_dim)
                if isinstance(s, torch.Tensor):
                    s = s.detach().cpu().numpy()
                
                # Use first action of each chunk for L2 comparison
                if s.ndim == 3:
                    s_first = s[:, 0, :]  # (K, action_dim)
                else:
                    s_first = s  # (K, action_dim)
                
                # Pairwise L2 distances
                dists = []
                for i in range(K):
                    for j in range(i+1, K):
                        l2 = float(np.linalg.norm(s_first[i] - s_first[j]))
                        dists.append(l2)
                
                mean_l2 = np.mean(dists) if dists else 0.0
                step_l2s.append(mean_l2)
                all_l2.append(mean_l2)
                
                # Step env with first sample
                action = s_first[0][:7]  # 7D for LIBERO
                obs, reward, done, info = env.step(action)
                if done:
                    break
            
            print(f"  [{task_name[:50]}] ep{ep}: {len(step_l2s)} steps, "
                  f"mean_L2={np.mean(step_l2s):.6f}, max={np.max(step_l2s):.6f}",
                  flush=True)
    
    arr = np.array(all_l2)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "pct_below_1e4": float((arr < 1e-4).mean() * 100),
        "pct_below_1e3": float((arr < 1e-3).mean() * 100),
        "pct_below_1e2": float((arr < 1e-2).mean() * 100),
        "n_samples": len(all_l2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["smolvla", "openvla_oft"])
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--n_tasks", type=int, default=3)
    parser.add_argument("--n_rollouts", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    print(f"[diversity] Model={args.model}, K={args.K}, tasks={args.n_tasks}, "
          f"rollouts={args.n_rollouts}, steps={args.max_steps}", flush=True)
    
    if args.model == "openvla_oft":
        from openvla_oft_wrapper import OpenVLAOFTWrapper
        wrapper = OpenVLAOFTWrapper(device=str(device))
    else:
        from smolvla_wrapper import SmolVLAWrapper
        wrapper = SmolVLAWrapper(
            "./models/smolvla", device=str(device)
        )
    
    print("[diversity] Setting up LIBERO environments...", flush=True)
    all_task_envs = load_libero_env("object")
    task_envs = all_task_envs[:args.n_tasks]
    
    print(f"[diversity] Running {len(task_envs)} tasks x {args.n_rollouts} rollouts...", flush=True)
    t0 = time.time()
    results = run_diversity_analysis(
        wrapper, task_envs, K=args.K, n_rollouts=args.n_rollouts,
        max_steps=args.max_steps, seed=args.seed
    )
    elapsed = time.time() - t0
    
    results["model"] = args.model
    results["K"] = args.K
    results["elapsed_s"] = elapsed
    
    print(f"\n{'='*60}", flush=True)
    print(f"[RESULT] {args.model} K={args.K}", flush=True)
    print(f"  Mean L2:    {results['mean']:.6f}", flush=True)
    print(f"  Std L2:     {results['std']:.6f}", flush=True)
    print(f"  Median L2:  {results['median']:.6f}", flush=True)
    print(f"  Max L2:     {results['max']:.6f}", flush=True)
    print(f"  <1e-4:      {results['pct_below_1e4']:.1f}%", flush=True)
    print(f"  <1e-3:      {results['pct_below_1e3']:.1f}%", flush=True)
    print(f"  <1e-2:      {results['pct_below_1e2']:.1f}%", flush=True)
    print(f"  N samples:  {results['n_samples']}", flush=True)
    print(f"  Time:       {elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)
    
    out = args.output or f"logs/diversity_{args.model}_K{args.K}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[diversity] Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
