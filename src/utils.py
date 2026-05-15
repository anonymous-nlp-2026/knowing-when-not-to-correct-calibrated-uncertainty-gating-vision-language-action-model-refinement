"""
ACR-VLA utility functions.

- LIBERO dataset loading
- Synthetic augmentation (known noise injection for OOD simulation)
- Evaluation metrics
- Checkpoint management
"""
import torch
import numpy as np
from pathlib import Path
import h5py
import glob


CORRECTION_DIM = 6   # 6D pose — CRM corrects only pose, not gripper
FULL_ACTION_DIM = 7  # 6D pose + 1D gripper (LIBERO env action space)
H_EFF = 5            # effective horizon (truncated from chunk_size=50)
CHUNK_SIZE = 50      # SmolVLA native chunk_size
OBS_DIM = 960        # VLM hidden_size
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32

BENCHMARK_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
    "90": "libero_90",
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_libero_demos(benchmark_name: str,
                      data_dir: str = "/root/autodl-tmp/libero_data/",
                      max_demos_per_task: int = 50):
    """
    Load LIBERO demo trajectories from HDF5 files.

    Returns:
        list of dicts with keys:
            - "agentview_rgb": (T, 128, 128, 3) uint8
            - "eye_in_hand_rgb": (T, 128, 128, 3) uint8
            - "joint_states": (T, 7) float
            - "ee_states": (T, 8) float — eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
            - "actions": (T, 7) float (6D pose + 1D gripper)
            - "task_name": str
    """
    folder = BENCHMARK_MAP.get(benchmark_name, benchmark_name)
    data_path = Path(data_dir) / folder
    if not data_path.exists():
        raise FileNotFoundError(f"LIBERO data not found at {data_path}")

    hdf5_files = sorted(data_path.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {data_path}")

    trajectories = []
    for hdf5_file in hdf5_files:
        task_name = hdf5_file.stem.replace("_demo", "").replace("_", " ")
        with h5py.File(hdf5_file, "r") as f:
            demo_keys = sorted(
                [k for k in f["data"].keys() if k.startswith("demo")],
                key=lambda x: int(x.split("_")[1])
            )
            for dk in demo_keys[:max_demos_per_task]:
                demo = f[f"data/{dk}"]
                traj = {
                    "agentview_rgb": np.array(demo["obs/agentview_rgb"]),
                    "eye_in_hand_rgb": np.array(demo["obs/eye_in_hand_rgb"]),
                    "joint_states": np.array(demo["obs/joint_states"]),
                    # BugFix C1: include gripper_states to match eval's 8D state
                    "ee_states": np.concatenate([
                        np.array(demo["obs/ee_states"]),      # (T, 6) eef_pos + axisangle
                        np.array(demo["obs/gripper_states"]),  # (T, 2) gripper_qpos
                    ], axis=-1),  # (T, 8)
                    "actions": np.array(demo["actions"]),
                    "task_name": task_name,
                }
                trajectories.append(traj)

    return trajectories


def build_training_samples(trajectories: list, h_eff: int = H_EFF,
                           action_dim: int = CORRECTION_DIM):
    """
    Convert trajectories into (timestep, image, state, expert_action_chunk) tuples.

    expert_action is sliced to action_dim (default CORRECTION_DIM=6, pose only).

    Returns:
        list of dicts with keys:
            - "agentview_rgb": (128, 128, 3) uint8
            - "eye_in_hand_rgb": (128, 128, 3) uint8
            - "ee_states": (8,) float — eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
            - "expert_action": (h_eff, action_dim) float
            - "task_name": str
    """
    samples = []
    for traj in trajectories:
        T = traj["actions"].shape[0]
        for t in range(T - h_eff + 1):
            samples.append({
                "agentview_rgb": traj["agentview_rgb"][t],
                "eye_in_hand_rgb": traj["eye_in_hand_rgb"][t],
                "ee_states": traj["ee_states"][t],
                "expert_action": traj["actions"][t:t + h_eff, :action_dim].astype(np.float32),
                "task_name": traj["task_name"],
            })
    return samples


def synthetic_augmentation(expert_action: torch.Tensor,
                           noise_scale: torch.Tensor,
                           perturbation: torch.Tensor | None = None) -> torch.Tensor:
    """Add known noise to expert actions to simulate OOD conditions."""
    if expert_action.dim() == 2:
        noise = torch.randn_like(expert_action) * noise_scale.repeat(H_EFF).unsqueeze(0)
    else:
        noise = torch.randn_like(expert_action) * noise_scale.unsqueeze(0).unsqueeze(0)
    noisy = expert_action + noise
    if perturbation is not None:
        noisy = noisy + perturbation
    return noisy


class CheckpointManager:
    def __init__(self, output_dir: str, max_keep: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, metrics: dict):
        path = self.output_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        self._cleanup()
        return path

    def load_best(self, model: torch.nn.Module, metric_key: str = "val_loss",
                  lower_is_better: bool = True):
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints in {self.output_dir}")

        best_path = None
        best_metric = float("inf") if lower_is_better else float("-inf")
        for cp_path in checkpoints:
            cp = torch.load(cp_path, map_location="cpu", weights_only=False)
            metric = cp["metrics"].get(metric_key, float("inf"))
            if (lower_is_better and metric < best_metric) or \
               (not lower_is_better and metric > best_metric):
                best_metric = metric
                best_path = cp_path

        cp = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(cp["model_state_dict"])
        return cp

    def _cleanup(self):
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pt"))
        while len(checkpoints) > self.max_keep:
            checkpoints[0].unlink()
            checkpoints.pop(0)
