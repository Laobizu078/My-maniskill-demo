# ManiSkill Pick-and-Lift Demo

## Abstract
This repository provides a compact ManiSkill3 pick-and-lift pipeline with three stages: data collection, behavior cloning training, and evaluation with optional video export.  
The current setup uses scripted expert trajectories (align, descend, close gripper, lift) to generate high-quality offline data, then trains a continuous-control policy for grasp-and-lift behavior on `PickCube-v1`.

## Reproducibility Guide

### 1) Requirements
- Python environment with: `mani_skill`, `gymnasium`, `torch`, `numpy`, `imageio`.
- A working conda env (recommended name: `maniskill`).
- Optional but recommended: NVIDIA GPU + driver for faster training and video rendering.

### 2) Configure Experiment
- Edit [demo_config.py] for all default settings:
  - version tag
  - paths
  - collect/train/eval hyperparameters
- Current default version is `v3`.

### 3) Run End-to-End
Use this from the project root:
```bash
cd ~/maniskill-demo
/home/bizu/miniconda3/bin/conda run --no-capture-output -n maniskill python collect.py
/home/bizu/miniconda3/bin/conda run --no-capture-output -n maniskill python train.py
/home/bizu/miniconda3/bin/conda run --no-capture-output -n maniskill python eval.py
```

### 4) Expected Outputs
- Dataset: `data/dataset_v3.npz`
- Model: `model/bc_policy_v3.pt`
- Video: `videos/demo_v3.mp4` (when rendering is available)

## File Overview
- `collect.py`: collect offline demonstrations in ManiSkill3.
- `train.py`: train a behavior cloning policy from collected data.
- `eval.py`: evaluate policy and optionally export rollout videos.
- `demo_config.py`: centralized experiment configuration (with comments).

## Troubleshooting
- `ModuleNotFoundError` for `gymnasium`/`mani_skill`:
  - Run scripts inside the correct conda env.
- `render init failed` or no video generated:
  - Your runtime may not have a usable render device.
  - Set `render_mode="none"` and `render_backend="none"` in config for headless eval.
- `torch.cuda.is_available() == False`:
  - GPU is not visible in the current runtime/session even if host machine has GPU.
