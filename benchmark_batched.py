"""Latency benchmark: sequential vs batched K-sample generation."""
import sys
sys.path.insert(0, "/root/acr-vla-conformal-refinement/src")

import time
import torch
import numpy as np
from smolvla_wrapper import SmolVLAWrapper
from utils import load_libero_demos

print("Loading model...")
model = SmolVLAWrapper()

print("Loading demo data...")
demos = load_libero_demos("object", max_demos_per_task=1)
demo = demos[0]

print("Encoding prefix...")
obs_feat, kv_cache, pad_masks = model.encode_prefix(
    *model.preprocess_obs(
        demo["agentview_rgb"][0],
        demo["task_name"],
        demo["ee_states"][0],
        demo["eye_in_hand_rgb"][0],
    )
)

for K in [5, 10, 20]:
    print(f"\n{'='*50}")
    print(f"K = {K}")
    print(f"{'='*50}")

    # Warmup
    for _ in range(5):
        model.generate_k_samples(kv_cache, pad_masks, K=K)
        model.generate_k_samples_batched(kv_cache, pad_masks, K=K)
    torch.cuda.synchronize()

    # Sequential benchmark
    times_seq = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate_k_samples(kv_cache, pad_masks, K=K)
        torch.cuda.synchronize()
        times_seq.append(time.perf_counter() - t0)

    # Batched benchmark
    times_batch = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate_k_samples_batched(kv_cache, pad_masks, K=K)
        torch.cuda.synchronize()
        times_batch.append(time.perf_counter() - t0)

    seq_mean = np.mean(times_seq) * 1000
    seq_std = np.std(times_seq) * 1000
    batch_mean = np.mean(times_batch) * 1000
    batch_std = np.std(times_batch) * 1000
    speedup = np.mean(times_seq) / np.mean(times_batch)

    print(f"Sequential: {seq_mean:.1f} +/- {seq_std:.1f} ms")
    print(f"Batched:    {batch_mean:.1f} +/- {batch_std:.1f} ms")
    print(f"Speedup:    {speedup:.2f}x")

print("\nDone.")
