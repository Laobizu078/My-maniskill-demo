import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from demo_config import CONFIG, default_path

TRAIN_CFG = CONFIG["train"]


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        dims = [obs_dim, *hidden_dims, act_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC policy for ManiSkill3 PickCube.")
    parser.add_argument("--version", type=str, default=CONFIG["version"], help="Version tag like v1/v2.")
    parser.add_argument("--data", type=str, default="", help="Training dataset path.")
    parser.add_argument("--save", type=str, default="", help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CFG["lr"])
    parser.add_argument("--weight-decay", type=float, default=TRAIN_CFG["weight_decay"])
    parser.add_argument("--val-ratio", type=float, default=TRAIN_CFG["val_ratio"])
    parser.add_argument("--hidden-dims", type=str, default=TRAIN_CFG["hidden_dims"])
    parser.add_argument("--seed", type=int, default=TRAIN_CFG["seed"])
    parser.add_argument("--device", type=str, default=TRAIN_CFG["device"], choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(amp=bool(TRAIN_CFG["amp"]))
    return parser.parse_args()


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"[train] device={device}, amp={use_amp}")

    if args.data.strip() == "":
        args.data = default_path("dataset", args.version)
    if args.save.strip() == "":
        args.save = default_path("checkpoint", args.version)

    data_path = str(Path(args.data).expanduser())
    save_path = str(Path(args.save).expanduser())
    ensure_parent_dir(save_path)

    d = np.load(data_path, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    actions = d["actions"].astype(np.float32)

    if obs.ndim != 2:
        obs = obs.reshape(len(obs), -1)
    if actions.ndim != 2:
        actions = actions.reshape(len(actions), -1)
    if len(obs) != len(actions):
        raise ValueError(f"obs/action size mismatch: {len(obs)} vs {len(actions)}")

    n = len(obs)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_size = max(1, int(n * args.val_ratio))
    train_idx = idx[val_size:]
    val_idx = idx[:val_size]

    train_obs_np = obs[train_idx]
    train_actions_np = actions[train_idx]
    val_obs_np = obs[val_idx]
    val_actions_np = actions[val_idx]

    obs_mean = train_obs_np.mean(axis=0)
    obs_std = train_obs_np.std(axis=0)
    obs_std = np.clip(obs_std, 1e-6, None)

    # Normalize observations to stabilize BC training across runs.
    train_obs = torch.from_numpy((train_obs_np - obs_mean) / obs_std)
    train_actions = torch.from_numpy(train_actions_np)
    val_obs = torch.from_numpy((val_obs_np - obs_mean) / obs_std)
    val_actions = torch.from_numpy(val_actions_np)

    train_loader = DataLoader(
        TensorDataset(train_obs, train_actions),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_obs, val_actions),
        batch_size=args.batch_size,
        shuffle=False,
    )

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
    model = BCPolicy(obs_dim=obs.shape[1], act_dim=actions.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)
            optimizer.zero_grad()
            # Use GradScaler so the same loop works for fp32 and mixed precision.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(x)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss += float(loss.item()) * len(x)
        val_loss /= len(val_idx)

        if val_loss < best_val:
            # Keep the best checkpoint on validation loss.
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(
                f"[train] epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Training failed: no model checkpoint was produced.")

    checkpoint = {
        "model_state_dict": best_state,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": int(obs.shape[1]),
        "act_dim": int(actions.shape[1]),
        "hidden_dims": hidden_dims,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "best_val_loss": float(best_val),
    }
    torch.save(checkpoint, save_path)
    print(f"[train] saved best policy to {save_path} (best_val_loss={best_val:.6f})")


if __name__ == "__main__":
    main()
