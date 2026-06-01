import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

print('=== Loading Pi05Wrapper ===', flush=True)
from pi05_wrapper import Pi05Wrapper

w = Pi05Wrapper('./models/pi05', device='cuda')
print(f'Model loaded, dtype of first param: {next(w.policy.parameters()).dtype}', flush=True)
print(f'State mean shape: {w.state_mean.shape}, Action mean shape: {w.action_mean.shape}', flush=True)
print(f'State mean: {w.state_mean}', flush=True)
print(f'State std: {w.state_std}', flush=True)
print(f'Action mean: {w.action_mean}', flush=True)
print(f'Action std: {w.action_std}', flush=True)

# Create fake obs
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
eye_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
state = np.array([-0.05, 0.03, 0.76, 2.97, -0.22, -0.13, 0.027, -0.027], dtype=np.float64)

# Test state normalization
state_t = torch.tensor(state, dtype=torch.float32, device='cuda')
state_norm = w._normalize_state(state_t)
print(f'Normalized state: {state_norm.cpu().numpy()}', flush=True)

# Test tokenization
imgs, img_masks, lt, lm = w.preprocess_obs(img, 'put the moka pot on the stove', state, eye_img)
print(f'Num images: {len(imgs)}, shapes: {[i.shape for i in imgs]}', flush=True)
print(f'Image masks: {[m.item() for m in img_masks]}', flush=True)
print(f'Lang tokens shape: {lt.shape}', flush=True)
tokens_decoded = w.tokenizer.decode(lt[0][:30])
print(f'First 30 tokens decoded: {repr(tokens_decoded)}', flush=True)
# Also print raw token ids
print(f'First 30 token ids: {lt[0][:30].tolist()}', flush=True)

# Test inference
print('=== Running inference K=2 ===', flush=True)
obs_feat, samples = w.get_obs_features_and_samples(img, 'put the moka pot on the stove', state, eye_img, K=2)
print(f'Obs features shape: {obs_feat.shape}, range: [{obs_feat.min():.4f}, {obs_feat.max():.4f}]', flush=True)
print(f'Samples shape: {samples.shape}', flush=True)
print(f'Sample 0 first 3 steps:\n{samples[0, 0, :3, :].cpu().numpy()}', flush=True)
print(f'Sample 1 first 3 steps:\n{samples[0, 1, :3, :].cpu().numpy()}', flush=True)

# Check if actions are in reasonable range
act = samples[0, 0, 0, :].cpu().numpy()
print(f'\nFirst action: {act}', flush=True)
print(f'Action mean (stats): {w.action_mean[:7].cpu().numpy()}', flush=True)
print(f'Action std (stats): {w.action_std[:7].cpu().numpy()}', flush=True)

# Check if gripper is reasonable
print(f'Gripper values (sample 0): {samples[0, 0, :5, 6].cpu().numpy()}', flush=True)
print(f'Gripper values (sample 1): {samples[0, 1, :5, 6].cpu().numpy()}', flush=True)

# Also test with lerobot native API for comparison
print('\n=== Testing lerobot native API ===', flush=True)
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    
    # The model is already loaded via wrapper, reuse it
    policy = w.policy
    
    # Build batch in lerobot format
    batch = {}
    batch['observation.images.image'] = imgs[0]  # (1, C, H, W) from wrapper preprocess
    batch['observation.images.image2'] = imgs[1]  # (1, C, H, W)
    # Note: empty camera is handled by _preprocess_images
    batch[OBS_LANGUAGE_TOKENS] = lt
    batch[OBS_LANGUAGE_ATTENTION_MASK] = lm
    
    # Use predict_action_chunk
    actions_native = policy.predict_action_chunk(batch)
    print(f'Native API actions shape: {actions_native.shape}', flush=True)
    print(f'Native API first 3 steps:\n{actions_native[0, :3, :].cpu().numpy()}', flush=True)
    
    # Compare with wrapper output
    wrapper_act = samples[0, 0, :3, :].cpu().numpy()
    native_act = actions_native[0, :3, :].cpu().numpy()
    diff = np.abs(wrapper_act - native_act)
    print(f'\nDiff (wrapper - native) first 3 steps:\n{diff}', flush=True)
    print(f'Max diff: {diff.max():.6f}', flush=True)
except Exception as e:
    print(f'Native API test failed: {e}', flush=True)
    import traceback
    traceback.print_exc()

print('\n=== Done ===', flush=True)
