import argparse
import copy
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import mani_skill.envs  # noqa: F401  # needed for env registration
import numpy as np
import torch
from demo_config import CONFIG, default_path

COLLECT_CFG = CONFIG["collect"]
ENV_CFG = CONFIG["env"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ManiSkill3 PickCube data.")
    parser.add_argument("--env-id", type=str, default=ENV_CFG["env_id"])
    parser.add_argument("--episodes", type=int, default=COLLECT_CFG["episodes"])
    parser.add_argument("--max-steps", type=int, default=COLLECT_CFG["max_steps"])
    parser.add_argument("--seed", type=int, default=COLLECT_CFG["seed"])
    parser.add_argument("--version", type=str, default=CONFIG["version"], help="Version tag like v1/v2.")
    parser.add_argument("--out", type=str, default="", help="Output dataset path.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default=COLLECT_CFG["render_mode"],
        choices=["none", "rgb_array"],
        help="Use none for headless data collection.",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        default=COLLECT_CFG["render_backend"],
        choices=["none", "cpu", "gpu", "sapien_cpu", "sapien_cuda"],
        help="Set to none to disable render system creation.",
    )
    parser.add_argument(
        "--planner-samples",
        type=int,
        default=COLLECT_CFG["planner_samples"],
        help="One-step random shooting samples. Set 0 to disable.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=COLLECT_CFG["policy"],
        choices=["random", "planner", "scripted"],
        help="Data collection policy.",
    )
    return parser.parse_args()


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def flatten_obs(obs: Any) -> np.ndarray:
    # Convert nested observation structures into a single flat state vector.
    if isinstance(obs, dict):
        keys = sorted(obs.keys())
        return np.concatenate([flatten_obs(obs[k]) for k in keys], axis=0)
    if isinstance(obs, (list, tuple)):
        return np.concatenate([flatten_obs(x) for x in obs], axis=0)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def flatten_action(action: Any) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


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


def find_state_api(env: gym.Env) -> tuple[Callable[[], Any] | None, Callable[[Any], None] | None]:
    # Support multiple ManiSkill versions with different state API names.
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    candidates = [
        ("get_state_dict", "set_state_dict"),
        ("get_state", "set_state"),
        ("get_sim_state", "set_sim_state"),
    ]
    for get_name, set_name in candidates:
        getter = getattr(base_env, get_name, None)
        setter = getattr(base_env, set_name, None)
        if callable(getter) and callable(setter):
            return getter, setter
    return None, None


def choose_action(
    env: gym.Env,
    get_state: Callable[[], Any] | None,
    set_state: Callable[[Any], None] | None,
    planner_samples: int,
) -> np.ndarray:
    # One-step random shooting: sample actions, score immediate reward, and pick the best.
    if planner_samples <= 0 or get_state is None or set_state is None:
        return flatten_action(env.action_space.sample())

    state = get_state()
    best_reward = -np.inf
    best_action = None

    try:
        for _ in range(planner_samples):
            a = flatten_action(env.action_space.sample())
            _, reward, _, _, info = env.step(a)
            score = float(reward) + (10.0 if parse_success(info) else 0.0)
            if score > best_reward:
                best_reward = score
                best_action = a.copy()
            set_state(copy.deepcopy(state))
    except Exception:
        return flatten_action(env.action_space.sample())

    if best_action is None:
        return flatten_action(env.action_space.sample())
    return best_action


class ScriptedPickPolicy:
    """Simple finite-state policy: hover -> descend -> close -> lift."""

    def __init__(self) -> None:
        self.phase = 0
        self.close_count = 0

    def reset(self) -> None:
        self.phase = 0
        self.close_count = 0

    @staticmethod
    def _vec3(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        return arr.reshape(-1)[:3]

    def act(self, env: gym.Env) -> np.ndarray:
        # Read key poses from the simulator and execute a finite-state pick sequence.
        base = env.unwrapped
        tcp = self._vec3(base.agent.tcp.pose.p)
        cube = self._vec3(base.cube.pose.p)
        goal = self._vec3(base.goal_site.pose.p)

        hover_target = cube + np.array([0.0, 0.0, 0.10], dtype=np.float32)
        grasp_target = cube + np.array([0.0, 0.0, 0.015], dtype=np.float32)
        lift_target = np.array([cube[0], cube[1], max(cube[2] + 0.18, goal[2])], dtype=np.float32)

        if self.phase == 0:
            target = hover_target
            grip = 1.0
            if np.linalg.norm(tcp - hover_target) < 0.025:
                self.phase = 1
        elif self.phase == 1:
            target = grasp_target
            grip = 1.0
            dist_xy = np.linalg.norm((tcp - grasp_target)[:2])
            if dist_xy < 0.015 and abs(tcp[2] - grasp_target[2]) < 0.015:
                self.phase = 2
        elif self.phase == 2:
            target = grasp_target
            grip = -1.0
            self.close_count += 1
            if self.close_count >= 15:
                self.phase = 3
        else:
            target = lift_target
            grip = -1.0

        delta = target - tcp
        xyz = np.clip(delta * 8.0, -1.0, 1.0)
        action = np.concatenate([xyz, np.array([grip], dtype=np.float32)], axis=0).astype(np.float32)
        return action


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    if args.out.strip() == "":
        args.out = default_path("dataset", args.version)
    ensure_parent_dir(args.out)

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = gym.make(
        args.env_id,
        obs_mode=ENV_CFG["obs_mode"],
        control_mode=ENV_CFG["control_mode"],
        reward_mode=ENV_CFG["reward_mode"],
        render_mode=render_mode,
        render_backend=args.render_backend,
    )

    get_state, set_state = find_state_api(env)
    use_planner = (
        args.policy == "planner"
        and get_state is not None
        and set_state is not None
        and args.planner_samples > 0
    )
    scripted = ScriptedPickPolicy()
    print(f"[collect] policy={args.policy}, planner enabled: {use_planner}")

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    ep_returns: list[float] = []
    ep_success: list[float] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        scripted.reset()
        total_r = 0.0
        success_flag = False

        for _ in range(args.max_steps):
            obs_vec = flatten_obs(obs)
            # Switch between scripted, planner-based, and random collection policies.
            if args.policy == "scripted":
                act = scripted.act(env)
            elif args.policy == "planner" and use_planner:
                act = choose_action(env, get_state, set_state, args.planner_samples)
            else:
                act = flatten_action(env.action_space.sample())

            nxt, reward, terminated, truncated, info = env.step(act)
            obs_buf.append(obs_vec)
            act_buf.append(act)
            total_r += float(reward)
            success_flag = success_flag or parse_success(info)
            obs = nxt

            if terminated or truncated:
                break

        ep_returns.append(total_r)
        ep_success.append(float(success_flag))
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(
                f"[collect] episode {ep + 1:4d}/{args.episodes} | "
                f"return={total_r:8.3f} | success={int(success_flag)}"
            )

    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    act_arr = np.asarray(act_buf, dtype=np.float32)
    ep_ret_arr = np.asarray(ep_returns, dtype=np.float32)
    ep_succ_arr = np.asarray(ep_success, dtype=np.float32)

    out_path = str(Path(args.out).expanduser())
    np.savez_compressed(
        out_path,
        obs=obs_arr,
        actions=act_arr,
        episode_returns=ep_ret_arr,
        episode_success=ep_succ_arr,
        env_id=np.array([args.env_id]),
        control_mode=np.array([ENV_CFG["control_mode"]]),
    )

    print(f"[collect] saved to {out_path}")
    print(
        f"[collect] transitions={len(obs_arr)}, avg_return={ep_ret_arr.mean():.3f}, "
        f"success_rate={ep_succ_arr.mean():.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
