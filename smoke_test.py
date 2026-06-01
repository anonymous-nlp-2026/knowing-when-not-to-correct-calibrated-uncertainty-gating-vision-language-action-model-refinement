import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

print(f"[smoke] Starting smoke test...", flush=True)
t0 = time.time()

print(f"[smoke] Importing Pi05Wrapper...", flush=True)
from pi05_wrapper import Pi05Wrapper

print(f"[smoke] Creating wrapper (loading model)...", flush=True)
wrapper = Pi05Wrapper(
    pretrained_path='./models/pi05',
    device='cuda:0'
)
t1 = time.time()
print(f"[smoke] Model loaded in {t1-t0:.1f}s", flush=True)
print(f"[smoke] GPU memory: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB", flush=True)

# Create dummy obs
dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
dummy_state = np.random.randn(8).astype(np.float32)  # 8D: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
task_name = "pick up the red mug and place it on the plate"

print(f"[smoke] Testing get_obs_features_and_samples (K=2)...", flush=True)
t2 = time.time()
try:
    obs_feat, samples = wrapper.get_obs_features_and_samples(
        agentview_rgb=dummy_img,
        task_name=task_name,
        robot_state=dummy_state,
        eye_in_hand_rgb=dummy_img,
        K=2
    )
    t3 = time.time()
    print(f"[smoke] obs_features shape: {obs_feat.shape}", flush=True)
    print(f"[smoke] samples shape: {samples.shape}", flush=True)
    print(f"[smoke] obs_features range: [{obs_feat.min().item():.4f}, {obs_feat.max().item():.4f}]", flush=True)
    print(f"[smoke] samples range: [{samples.min().item():.4f}, {samples.max().item():.4f}]", flush=True)
    print(f"[smoke] Inference time: {t3-t2:.2f}s", flush=True)
    print(f"[smoke] GPU memory after inference: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB", flush=True)
    print(f"[smoke] Smoke test PASSED", flush=True)
except Exception as e:
    print(f"[smoke] Smoke test FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
