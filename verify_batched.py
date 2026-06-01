"""Numerical verification: sequential vs batched K-sample generation."""
import sys
sys.path.insert(0, "/root/acr-vla-conformal-refinement/src")

import torch
from smolvla_wrapper import SmolVLAWrapper
from utils import load_libero_demos, FULL_ACTION_DIM

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

K = 5
B = pad_masks.shape[0]
chunk_size = model.config.chunk_size
max_act_dim = model.config.max_action_dim
num_steps = model.config.num_steps
dt = -1.0 / num_steps

print(f"B={B}, K={K}, chunk_size={chunk_size}, max_act_dim={max_act_dim}, num_steps={num_steps}")

# Generate shared noise (K tensors of shape (B, chunk_size, max_act_dim))
torch.manual_seed(42)
noise_list = [model.flow.sample_noise((B, chunk_size, max_act_dim), model.device) for _ in range(K)]

# --- Sequential ---
print("Running sequential...")
all_samples = []
for k in range(K):
    x_t = noise_list[k].clone()
    for step in range(num_steps):
        time_val = 1.0 + step * dt
        time_tensor = torch.full((B,), time_val, dtype=torch.float32, device=model.device)
        v_t = model.flow.denoise_step(
            x_t=x_t,
            prefix_pad_masks=pad_masks,
            past_key_values=kv_cache,
            timestep=time_tensor,
        )
        x_t = x_t + dt * v_t
    raw_action = x_t[:, :, :FULL_ACTION_DIM]
    all_samples.append(model._unnormalize_action(raw_action))
samples_seq = torch.stack(all_samples, dim=1)

# --- Batched ---
print("Running batched...")
# Rearrange noise: (K, B, C, D) -> (B, K, C, D) -> (B*K, C, D)
noise_batched = torch.stack(noise_list, dim=1).reshape(B * K, chunk_size, max_act_dim)
samples_batch = model.generate_k_samples_batched(kv_cache, pad_masks, K=K, noise=noise_batched)

# --- Compare ---
max_diff = (samples_seq - samples_batch).abs().max().item()
mean_diff = (samples_seq - samples_batch).abs().mean().item()
print(f"\nMax absolute diff:  {max_diff:.2e}")
print(f"Mean absolute diff: {mean_diff:.2e}")
print(f"Shape seq:    {samples_seq.shape}")
print(f"Shape batch:  {samples_batch.shape}")

# Check per-sample diffs
for k in range(K):
    diff_k = (samples_seq[:, k] - samples_batch[:, k]).abs().max().item()
    print(f"  Sample {k}: max diff = {diff_k:.2e}")

if torch.allclose(samples_seq, samples_batch, atol=1e-4):
    print("\n=== NUMERICAL VERIFICATION PASSED (atol=1e-4) ===")
elif torch.allclose(samples_seq, samples_batch, atol=1e-3):
    print("\n=== NUMERICAL VERIFICATION PASSED (atol=1e-3) ===")
else:
    print(f"\n=== VERIFICATION FAILED: max diff = {max_diff:.2e} ===")

print("\nDone.")
