import argparse
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401  # needed for env registration
import numpy as np
import torch
from demo_config import CONFIG, default_path

EVAL_CFG = CONFIG["eval"]
ENV_CFG = CONFIG["env"]


class BCPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        dims = [obs_dim, *hidden_dims, act_dim]
        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC policy on ManiSkill3 PickCube.")
    parser.add_argument("--env-id", type=str, default=ENV_CFG["env_id"])
    parser.add_argument("--version", type=str, default=CONFIG["version"], help="Version tag like v1/v2.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint input path.")
    parser.add_argument("--episodes", type=int, default=EVAL_CFG["episodes"])
    parser.add_argument("--max-steps", type=int, default=EVAL_CFG["max_steps"])
    parser.add_argument("--seed", type=int, default=EVAL_CFG["seed"])
    parser.add_argument("--video", type=str, default="", help="Video output path.")
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=EVAL_CFG["record_episodes"],
        help="-1 means record all evaluation episodes into one video.",
    )
    parser.add_argument("--fps", type=int, default=EVAL_CFG["fps"])
    parser.add_argument(
        "--render-mode",
        type=str,
        default=EVAL_CFG["render_mode"],
        choices=["none", "rgb_array"],
        help="If rgb_array fails on headless machines, script will auto fallback to none.",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        default=EVAL_CFG["render_backend"],
        choices=["none", "cpu", "gpu", "sapien_cpu", "sapien_cuda"],
        help="Use gpu/cpu only when rendering is supported on the machine.",
    )
    parser.add_argument("--device", type=str, default=EVAL_CFG["device"], choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--action-scale",
        type=float,
        default=EVAL_CFG["action_scale"],
        help="Scale policy output to make motion less aggressive.",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=EVAL_CFG["action_smoothing"],
        help="EMA factor in [0, 1). Higher means smoother/slower action change.",
    )
    return parser.parse_args()


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        keys = sorted(obs.keys())
        return np.concatenate([flatten_obs(obs[k]) for k in keys], axis=0)
    if isinstance(obs, (list, tuple)):
        return np.concatenate([flatten_obs(x) for x in obs], axis=0)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def parse_success(info: dict[str, Any]) -> bool:
    if "success" not in info:
        return False
    val = info["success"]
    if isinstance(val, (list, tuple)):
        return bool(np.any(np.asarray(val)))
    if hasattr(val, "item"):
        try:
            return bool(val.item())
        except Exception:
            pass
    return bool(val)


def parse_flag(info: dict[str, Any], key: str) -> bool:
    if key not in info:
        return False
    val = info[key]
    if isinstance(val, (list, tuple)):
        return bool(np.any(np.asarray(val)))
    if hasattr(val, "item"):
        try:
            return bool(val.item())
        except Exception:
            pass
    return bool(val)


def to_frame(render_out: Any) -> np.ndarray | None:
    # Normalize different render outputs (torch/np/list/dict) into one RGB frame.
    if render_out is None:
        return None
    if isinstance(render_out, torch.Tensor):
        t = render_out.detach()
        if t.ndim == 4 and t.shape[0] >= 1:
            t = t[0]
        if t.is_cuda:
            t = t.cpu()
        arr = t.numpy()
        return arr
    if isinstance(render_out, np.ndarray):
        return render_out
    if isinstance(render_out, (list, tuple)) and len(render_out) > 0:
        if isinstance(render_out[0], np.ndarray):
            return render_out[0]
    if isinstance(render_out, dict):
        for k in ("rgb", "rgb_array", "image"):
            if k in render_out and isinstance(render_out[k], np.ndarray):
                return render_out[k]
    return None


def main() -> None:
    args = parse_args()
    if args.ckpt.strip() == "":
        args.ckpt = default_path("checkpoint", args.version)
    if args.video.strip() == "":
        args.video = default_path("video", args.version)
    ckpt_path = str(Path(args.ckpt).expanduser())
    video_path = str(Path(args.video).expanduser())
    ensure_parent_dir(video_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[eval] device={device}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = BCPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)

    render_mode = None if args.render_mode == "none" else args.render_mode
    try:
        env = gym.make(
            args.env_id,
            obs_mode=ENV_CFG["obs_mode"],
            control_mode=ENV_CFG["control_mode"],
            render_mode=render_mode,
            render_backend=args.render_backend,
        )
    except RuntimeError as e:
        if "Failed to find a supported physical device" in str(e):
            # Gracefully fallback for headless machines without a render device.
            print("[eval] render init failed, fallback to render_mode=none (no video).")
            env = gym.make(
                args.env_id,
                obs_mode=ENV_CFG["obs_mode"],
                control_mode=ENV_CFG["control_mode"],
                render_mode=None,
                render_backend="none",
            )
            args.record_episodes = 0
        else:
            raise

    if args.record_episodes < 0:
        args.record_episodes = args.episodes
    if args.record_episodes > 0 and (args.render_backend == "none" or args.render_mode == "none"):
        print("[eval] rendering disabled, disable video recording.")
        args.record_episodes = 0

    all_returns: list[float] = []
    all_success: list[float] = []
    all_grasp: list[float] = []
    all_lift: list[float] = []
    frames: list[np.ndarray] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r = 0.0
        success_flag = False
        grasp_flag = False
        lift_flag = False
        start_cube_z = float(env.unwrapped.cube.pose.p[0, 2].item())
        max_cube_z = start_cube_z
        prev_action = None

        for _ in range(args.max_steps):
            obs_vec = flatten_obs(obs)
            obs_in = (obs_vec - obs_mean) / np.clip(obs_std, 1e-6, None)
            x = torch.from_numpy(obs_in).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(x).squeeze(0).detach().cpu().numpy()
            action = action * float(args.action_scale)
            if prev_action is not None:
                alpha = float(np.clip(args.action_smoothing, 0.0, 0.999))
                action = alpha * prev_action + (1.0 - alpha) * action
            prev_action = action.copy()

            if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            success_flag = success_flag or parse_success(info)
            # Track grasp/lift explicitly; task success can include additional criteria.
            grasp_flag = grasp_flag or parse_flag(info, "is_grasped")
            cube_z = float(env.unwrapped.cube.pose.p[0, 2].item())
            max_cube_z = max(max_cube_z, cube_z)
            if max_cube_z - start_cube_z > 0.08:
                lift_flag = True

            if ep < args.record_episodes:
                frame = to_frame(env.render())
                if frame is not None:
                    frames.append(frame)

            if terminated or truncated:
                break

        all_returns.append(total_r)
        all_success.append(float(success_flag))
        all_grasp.append(float(grasp_flag))
        all_lift.append(float(lift_flag))
        print(
            f"[eval] episode {ep + 1:3d}/{args.episodes} | "
            f"return={total_r:8.3f} | success={int(success_flag)} | "
            f"grasp={int(grasp_flag)} | lift={int(lift_flag)}"
        )

    if len(frames) > 0:
        imageio.mimsave(video_path, frames, fps=args.fps)
        print(f"[eval] saved video to {video_path}")

    print(
        f"[eval] avg_return={np.mean(all_returns):.3f}, "
        f"success_rate={np.mean(all_success):.3f}, "
        f"grasp_rate={np.mean(all_grasp):.3f}, "
        f"lift_rate={np.mean(all_lift):.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
