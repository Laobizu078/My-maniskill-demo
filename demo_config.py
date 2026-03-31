"""
Centralized experiment configuration.

Usage:
1. Update values in CONFIG.
2. Run scripts directly:
   - python collect.py
   - python train.py
   - python eval.py
3. Override any field temporarily via CLI args when needed.
"""

from __future__ import annotations


CONFIG = {
    # Shared experiment version. collect/train/eval all use this by default.
    "version": "v3",
    # Environment and control settings.
    "env": {
        "env_id": "PickCube-v1",
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pos",
        "reward_mode": "dense",
    },
    # Output path templates. {version} is resolved automatically.
    "paths": {
        "dataset": "data/dataset_{version}.npz",
        "checkpoint": "model/bc_policy_{version}.pt",
        "video": "videos/demo_{version}.mp4",
    },
    # Default collection settings (collect.py).
    "collect": {
        "episodes": 500,
        "max_steps": 140,
        "seed": 0,
        "policy": "scripted",  # random | planner | scripted
        "planner_samples": 32,
        "render_mode": "none",  # none | rgb_array
        "render_backend": "none",  # none | cpu | gpu | sapien_cpu | sapien_cuda
    },
    # Default training settings (train.py).
    "train": {
        "epochs": 120,
        "batch_size": 4096,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "val_ratio": 0.1,
        "hidden_dims": "512,512,256",
        "seed": 0,
        "device": "auto",  # auto | cpu | cuda
        "amp": True,
    },
    # Default evaluation settings (eval.py).
    "eval": {
        "episodes": 12,
        "max_steps": 160,
        "seed": 0,
        "record_episodes": -1,  # -1 means record all evaluation episodes
        "fps": 20,
        "render_mode": "rgb_array",  # none | rgb_array
        "render_backend": "gpu",  # none | cpu | gpu | sapien_cpu | sapien_cuda
        "device": "auto",  # auto | cpu | cuda
        "action_scale": 1.0,
        "action_smoothing": 0.0,
    },
}


def default_path(kind: str, version: str) -> str:
    """Build a default output path from the version tag."""
    return CONFIG["paths"][kind].format(version=version)
