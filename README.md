# ACR-VLA: Adaptive Conformal Refinement for Frozen Flow-Matching VLAs

Adaptive test-time refinement framework that learns *when not to correct* frozen Vision-Language-Action models, using calibrated uncertainty gating to avoid harmful corrections on confident steps.

## Environment Setup

```bash
# Python 3.10+ required
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 datasets accelerate einops
pip install robosuite mujoco

# Install LIBERO benchmark
pip install libero

# Install project dependencies
pip install -r requirements.txt  # if exists, otherwise install above packages
```

## Code Structure

```
src/
├── crm.py                    # Conformal Refinement Module (CRM) — 3-layer MLP correction network
├── gated_crm.py              # Gated CRM with uncertainty-based selective correction
├── train_crm.py              # CRM training pipeline (multi-suite, observation-only features)
├── evaluate.py               # Main evaluation script — supports all modes (vla_only, crm_obs_only, adaptive_conformal_crm, etc.)
├── calibrate_threshold.py    # Conformal threshold (τ) calibration from calibration data
├── calibrate_crm_signal.py   # CRM signal analysis and calibration
├── smolvla_wrapper.py        # SmolVLA (0.4B) VLA wrapper for LIBERO evaluation
├── pi05_wrapper.py           # π₀.₅ (3.6B) VLA wrapper for LIBERO evaluation
├── openvla_oft_wrapper.py    # OpenVLA-OFT (7B, deterministic) wrapper — applicability boundary test
├── reconvla_baseline.py      # ReConVLA baseline implementation for comparison
├── utils.py                  # Shared utilities (action processing, logging, metrics)
├── aci.py                    # Adaptive Conformal Inference module
├── cgs.py                    # Conformal Guided Search
└── lerobot/                  # Patched LeRobot library for SmolVLA inference
```

## Reproducing Main Results (Table 1)

### Train CRM
```bash
python src/train_crm.py \
    --benchmark spatial object goal \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 32 \
    --output_dir checkpoints/crm_smolvla
```

### Evaluate: VLA-only baseline (K=10 mean)
```bash
python src/evaluate.py \
    --model smolvla \
    --mode vla_only \
    --baseline_mode mean \
    --K 10 \
    --benchmark spatial \
    --seed 42 \
    --n_rollouts 20
```

### Evaluate: CRM always-on (K=10)
```bash
python src/evaluate.py \
    --model smolvla \
    --mode crm_obs_only \
    --K 10 \
    --max_correction_norm 0.05 \
    --benchmark spatial \
    --seed 42 \
    --n_rollouts 20 \
    --checkpoint checkpoints/crm_smolvla/crm_best.pt
```

### Evaluate: Adaptive conformal gating
```bash
python src/evaluate.py \
    --model smolvla \
    --mode adaptive_conformal_crm \
    --K 10 \
    --max_correction_norm 0.05 \
    --conformal_threshold 0.074855 \
    --benchmark spatial \
    --seed 42 \
    --n_rollouts 20 \
    --checkpoint checkpoints/crm_smolvla/crm_best.pt
```

### Available benchmarks
`spatial`, `object`, `goal`, `long` (4 LIBERO task suites, 10 tasks each)

### Available models
- `smolvla` — SmolVLA 0.4B (flow-matching, stochastic)
- `pi05` — π₀.₅ 3.6B (flow-matching, stochastic)
- `openvla_oft` — OpenVLA-OFT 7B (deterministic, for applicability boundary analysis)

## Hardware Requirements

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) — single GPU sufficient
- **CRM training**: ~6 GPU-hours (3 suites, 50 epochs)
- **Evaluation**: ~2.5 GPU-hours per suite (20 rollouts × 10 tasks)

## License

This code is provided for review purposes.
