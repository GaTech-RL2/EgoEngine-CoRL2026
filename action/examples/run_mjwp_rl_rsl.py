from __future__ import annotations

import io
import json
import math
import os
import statistics
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path

import loguru
import mujoco
import numpy as np
import torch
import tyro
import warp as wp

from spider.config import Config, process_config
from spider.io import load_data
from spider.rl.mjwp_rsl_env import MJWPRLEnvCfg, MJWPRslResidualEnv
from spider.simulators.mjwp import (
    get_qvel,
    get_qpos,
    get_reward,
    get_terminal_reward,
    get_terminate,
    load_state,
    reset_inactive_side_to_ref,
    save_state,
    step_env,
)
from spider.viewers import render_image


class RLTrainingStop(Exception):
    """Raised when eval-driven PPO stop criteria are reached."""

    def __init__(self, message: str, *, failed: bool = False):
        super().__init__(message)
        self.failed = bool(failed)


@dataclass
class RLTrainArgs:
    # Dataset / task
    dataset_dir: str = "example_datasets"
    dataset_name: str = "gigahand"
    robot_type: str = "xhand"
    embodiment_type: str = "right"
    active_hand: str = "bimanual"  # right/left/bimanual (used when scene is bimanual)
    task: str = "mustard_901_1"
    data_id: int = 0
    data_path: str = ""
    residual_schedule_start_timestep: int = 0
    residual_schedule_total_timesteps: int = 0

    # Timing / planning (align with MPC)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.4
    horizon: float = 1.6
    knot_dt: float = 0.4

    # Reward scales (follow run_aria.sh)
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0
    contact_rew_scale: float = 0.0

    # Termination thresholds (MPC logic)
    object_pos_threshold: float = 0.05
    object_rot_threshold: float = 1.0
    object_mass_scale: float = 1.0

    # Domain randomization (match mjwp run_mjwp.py logic / parameterization)
    num_dr: int = 1
    pair_margin_range_min: float = -0.005
    pair_margin_range_max: float = 0.005
    xy_offset_range_min: float = -0.005
    xy_offset_range_max: float = 0.005
    object_mass_scale_range_min: float = 1.0
    object_mass_scale_range_max: float = 1.0

    # Residual action scale base (use MPC noise scales)
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    joint_noise_scale: float = 0.15
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03

    # RL
    seed: int = 0
    device: str = "cuda:0"
    num_envs: int = 8912
    eval_num_envs: int = 256
    future_ctrl_steps: int = 0  # kept for compatibility; minimal obs ignores future steps
    max_episode_ctrl_steps: int = 64
    eval_max_ctrl_steps: int = 0
    success_eval_ctrl_steps: int = 0
    export_execute_ctrl_steps: int = 0
    num_steps_per_env: int = 24
    learning_iterations: int = 1000
    early_stop_on_success: int = 1
    success_extra_iterations: int = 0
    early_stop_on_full_eval_success: int = 0
    train_success_trigger_eval: int = 0
    train_success_trigger_count: int = 2
    train_success_trigger_ctrl_steps: int = 0
    train_success_trigger_backoff_max: int = 8
    fail_no_progress_iterations: int = 0
    fail_no_progress_eval_count: int = 0
    plateau_after_success_eval_count: int = 0
    plateau_after_success_min_delta: float = 0.0
    train_plateau_after_success_window: int = 0
    train_plateau_after_success_min_delta: float = 0.0
    ckpt_interval: int = 50
    viz_interval: int = 10
    # <0 disables video; 0 renders until termination/reference end; >0 limits control steps.
    rollout_video_ctrl_steps: int = 32
    lifting_rew_scale: float = 0.0
    action_smoothness_rew_scale: float = 0.0
    # Optional lightweight W&B media upload (curve + rollout videos only)
    wandb_media_only: bool = False
    wandb_project: str = "spider-rl"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    log_dir: str = "outputs/rl_rsl"
    experiment_name: str = "aria_residual_single_demo"
    # Optional RSL checkpoint to initialize actor/critic/optimizer for a fresh run.
    load_ckpt: str = ""
    # Optional RSL checkpoint to continue training with the saved iteration counter.
    resume_ckpt: str = ""


def _infer_wandb_run_id(log_dir: Path) -> str:
    wandb_dir = log_dir / "wandb"
    candidates = []
    latest = wandb_dir / "latest-run"
    if latest.exists():
        try:
            candidates.append(latest.resolve())
        except Exception:
            candidates.append(latest)
    if wandb_dir.exists():
        candidates.extend(sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        name = path.name
        if name.startswith("run-"):
            parts = name.split("-")
            if parts:
                return parts[-1]
    return ""


def _make_spider_config(args: RLTrainArgs) -> Config:
    cfg = Config(
        simulator="mjwp",
        dataset_dir=os.path.abspath(args.dataset_dir),
        dataset_name=args.dataset_name,
        robot_type=args.robot_type,
        embodiment_type=args.embodiment_type,
        active_hand=args.active_hand,
        task=args.task,
        data_id=args.data_id,
        seed=args.seed,
        device=args.device,
        sim_dt=args.sim_dt,
        ctrl_dt=args.ctrl_dt,
        horizon=args.horizon,
        knot_dt=args.knot_dt,
        num_samples=args.num_envs,  # reuse MJWP vector worlds as RL env count
        # reward
        pos_rew_scale=args.pos_rew_scale,
        rot_rew_scale=args.rot_rew_scale,
        joint_rew_scale=args.joint_rew_scale,
        base_pos_rew_scale=args.base_pos_rew_scale,
        base_rot_rew_scale=args.base_rot_rew_scale,
        vel_rew_scale=args.vel_rew_scale,
        contact_rew_scale=args.contact_rew_scale,
        # terminate thresholds
        object_pos_threshold=args.object_pos_threshold,
        object_rot_threshold=args.object_rot_threshold,
        object_mass_scale=args.object_mass_scale,
        # domain randomization (same parameterization as mjwp)
        num_dr=args.num_dr,
        pair_margin_range=(args.pair_margin_range_min, args.pair_margin_range_max),
        xy_offset_range=(args.xy_offset_range_min, args.xy_offset_range_max),
        object_mass_scale_range=(
            args.object_mass_scale_range_min,
            args.object_mass_scale_range_max,
        ),
        # noise scales for residual action scaling
        first_ctrl_noise_scale=args.first_ctrl_noise_scale,
        last_ctrl_noise_scale=args.last_ctrl_noise_scale,
        final_noise_scale=args.final_noise_scale,
        joint_noise_scale=args.joint_noise_scale,
        pos_noise_scale=args.pos_noise_scale,
        rot_noise_scale=args.rot_noise_scale,
        # disable mpc-only outputs / viewer
        show_viewer=False,
        save_video=False,
        save_info=False,
        save_metrics=False,
        save_rerun=False,
        use_torch_compile=False,
    )
    cfg = process_config(cfg)
    if args.data_path:
        cfg.data_path = os.path.abspath(args.data_path)
    cfg.residual_schedule_start_timestep = int(args.residual_schedule_start_timestep)
    cfg.residual_schedule_total_timesteps = int(args.residual_schedule_total_timesteps)
    return cfg


def _make_rsl_train_cfg(args: RLTrainArgs) -> dict:
    # rsl_rl (newer API) expects top-level actor/critic/algorithm/obs_groups.
    return {
        "seed": int(args.seed),
        "num_steps_per_env": int(args.num_steps_per_env),
        "save_interval": int(args.ckpt_interval),
        "multi_gpu": None,
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo:PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "rsl_rl.models.mlp_model:MLPModel",
            "hidden_dims": [256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "stochastic": True,
            "init_noise_std": 1.0,
        },
        "critic": {
            "class_name": "rsl_rl.models.mlp_model:MLPModel",
            "hidden_dims": [256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "stochastic": False,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
    }


def _max_step_reward(cfg: Config) -> float:
    pos_term = float(cfg.pos_rew_scale) * float(cfg.object_pos_threshold)
    rot_term = float(cfg.rot_rew_scale) * float(cfg.object_rot_threshold)
    if cfg.embodiment_type == "bimanual":
        if cfg.active_hand in ["right", "left"]:
            return float((pos_term**2 + rot_term**2) ** 0.5)
        return float((2.0 * (pos_term**2 + rot_term**2)) ** 0.5)
    if cfg.embodiment_type in ["right", "left", "humanoid_object"]:
        return float((pos_term**2 + rot_term**2) ** 0.5)
    if cfg.embodiment_type in ["humanoid"]:
        base_pos_term = float(cfg.base_pos_rew_scale) * float(cfg.base_pos_threshold)
        base_rot_term = float(cfg.base_rot_rew_scale) * float(cfg.base_rot_threshold)
        return float((base_pos_term**2 + base_rot_term**2) ** 0.5)
    return 0.0


def _quat_wxyz_to_axis_angle_np(quat: np.ndarray) -> list[float]:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    q = q / norm
    if q[0] < 0.0:
        q = -q
    xyz = q[1:4]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(xyz_norm, float(q[0]))
    return ((xyz / xyz_norm) * angle).astype(float).tolist()


def _object_qpos_to_pose6(obj_qpos: np.ndarray) -> list[float]:
    obj = np.asarray(obj_qpos, dtype=np.float64).reshape(-1)
    if obj.size < 7:
        raise ValueError(f"Expected object qpos with at least 7 values, got {obj.size}")
    return obj[:3].astype(float).tolist() + _quat_wxyz_to_axis_angle_np(obj[3:7])


def _axis_angle_to_rotmat_np(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / angle
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rotmat_to_axis_angle_np(rotmat: np.ndarray) -> list[float]:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rotmat, dtype=np.float64).reshape(9))
    return _quat_wxyz_to_axis_angle_np(quat)


def _pose6_to_mat_np(pose6: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose6, dtype=np.float64).reshape(6)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _axis_angle_to_rotmat_np(pose[3:6])
    mat[:3, 3] = pose[:3]
    return mat


def _mat_to_pose6_np(mat: np.ndarray) -> list[float]:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return mat[:3, 3].astype(float).tolist() + _rotmat_to_axis_angle_np(mat[:3, :3])


_ARM6_TO_EEF_POS = np.array([0.0, 0.0, -0.102], dtype=np.float64)
_ARM6_TO_EEF_QUAT_WXYZ = np.array([0.0, 0.70710678118, 0.70710678118, 0.0], dtype=np.float64)
_HAND_LINK_TO_REPLAY_ARM6_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_RIGHT_XHAND_JOINTS = [
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
]


def _quat_wxyz_to_rotmat_np(quat: np.ndarray) -> np.ndarray:
    rot = np.zeros(9, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    mujoco.mju_quat2Mat(rot, q)
    return rot.reshape(3, 3)


def _rpy_to_xhand_base_rotmat_np(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    return (
        _axis_angle_to_rotmat_np(np.array([0.0, 0.0, roll], dtype=np.float64))
        @ _axis_angle_to_rotmat_np(np.array([pitch, 0.0, 0.0], dtype=np.float64))
        @ _axis_angle_to_rotmat_np(np.array([0.0, -yaw, 0.0], dtype=np.float64))
    )


def _arm6_to_eef_mat_np() -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = _ARM6_TO_EEF_POS
    mat[:3, :3] = _quat_wxyz_to_rotmat_np(_ARM6_TO_EEF_QUAT_WXYZ)
    return mat


def _xhand_eef_qpos6_to_right_arm6_pose6(
    eef_qpos6: np.ndarray,
    q_right_link_palm: np.ndarray,
) -> np.ndarray:
    eef_qpos6 = np.asarray(eef_qpos6, dtype=np.float64).reshape(6)
    eef_mat = np.eye(4, dtype=np.float64)
    eef_mat[:3, 3] = eef_qpos6[:3]
    eef_mat[:3, :3] = (
        _rpy_to_xhand_base_rotmat_np(eef_qpos6[3:6])
        @ _quat_wxyz_to_rotmat_np(q_right_link_palm)
    )
    arm6_mat = eef_mat @ np.linalg.inv(_arm6_to_eef_mat_np())
    arm6_mat[:3, :3] = arm6_mat[:3, :3] @ _HAND_LINK_TO_REPLAY_ARM6_ROT
    return np.asarray(_mat_to_pose6_np(arm6_mat), dtype=np.float64)


def _replace_eef_with_right_arm6(
    compact: np.ndarray,
    q_right_link_palm: np.ndarray,
) -> np.ndarray:
    compact = np.asarray(compact, dtype=np.float64).reshape(-1).copy()
    compact[:6] = _xhand_eef_qpos6_to_right_arm6_pose6(compact[:6], q_right_link_palm)
    return compact


def _evaluate_policy_summary(
    runner,
    env: MJWPRslResidualEnv,
    args: RLTrainArgs,
    cfg: Config,
    *,
    print_line: bool = True,
) -> dict[str, float]:
    policy = runner.alg.get_policy()
    policy_was_training = bool(getattr(policy, "training", False))
    policy.eval()

    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = env.ref_data
    real_steps = max(1, int(cfg.ref_steps))
    total_steps = int(qpos_ref.shape[0] - cfg.horizon_steps - cfg.ctrl_steps)
    total_steps = max(total_steps, 0)
    if int(args.eval_max_ctrl_steps) > 0:
        total_steps = min(total_steps, int(args.eval_max_ctrl_steps) * real_steps)
    if total_steps == 0:
        raise RuntimeError(f"No valid RL eval steps for task={args.task}")
    ctrl_steps = max(1, int(cfg.ctrl_steps))
    success_steps = total_steps
    if int(args.success_eval_ctrl_steps) > 0:
        success_steps = min(total_steps, int(args.success_eval_ctrl_steps) * real_steps)

    reward_sum = 0.0
    reward_sum_nonterminal = 0.0
    execute_reward_sum_nonterminal = 0.0
    terminal_bonus = 0.0
    reward_last = 0.0
    steps_executed = 0
    execute_steps = 0
    terminated = False
    termination_step = -1
    execute_step_limit = total_steps
    if int(args.export_execute_ctrl_steps) > 0:
        execute_step_limit = min(
            total_steps,
            int(args.export_execute_ctrl_steps) * real_steps,
        )

    env._set_state_from_ref(0)
    env.obs_buf = env._get_obs()

    t0 = time.perf_counter()
    with torch.no_grad():
        while steps_executed < total_steps and not terminated:
            obs_td = env.get_observations()
            actions = policy(obs_td)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            actions = torch.clamp(
                actions.to(env.device),
                -env.env_cfg.action_clip,
                env.env_cfg.action_clip,
            )

            base_ctrl_idx = env._clamp_ref_idx(env._sim_step)
            ctrl, _, _ = env.residual_actions_to_ctrl(actions, base_ctrl_idx)

            sim_step_base = int(env._sim_step)
            for sub in range(env.cfg.ctrl_steps):
                if steps_executed >= total_steps:
                    break

                step_env(env.cfg, env.env, ctrl)
                ref_idx = min(sim_step_base + sub, qpos_ref.shape[0] - 1)
                reset_inactive_side_to_ref(
                    env.cfg,
                    env.env,
                    qpos_ref[ref_idx],
                    qvel_ref[ref_idx],
                )
                env._sim_step += 1
                env._sim_step_buf += 1

                ref_step = (
                    qpos_ref[ref_idx],
                    qvel_ref[ref_idx],
                    ctrl_ref[ref_idx],
                    contact_ref[ref_idx],
                    contact_pos_ref[ref_idx],
                )
                rew, _ = get_reward(env.cfg, env.env, ref_step)
                reward_last = float(rew[0].item())
                reward_sum += reward_last
                reward_sum_nonterminal += reward_last
                steps_executed += 1
                if steps_executed <= execute_step_limit:
                    execute_reward_sum_nonterminal += reward_last
                    execute_steps += 1

                term = get_terminate(env.cfg, env.env, ref_step)
                if bool(term[0].item()):
                    term_rew, _ = get_terminal_reward(env.cfg, env.env, ref_step)
                    terminal_bonus = float(term_rew[0].item())
                    reward_sum += terminal_bonus
                    terminated = True
                    termination_step = steps_executed - 1
                    break

            env.obs_buf = env._get_obs()

    elapsed = max(time.perf_counter() - t0, 1e-8)
    steps_actual_real = float(steps_executed) / float(real_steps)
    steps_max_real = float(total_steps) / float(real_steps)
    termination_step_real = (
        float(termination_step) / float(real_steps) if termination_step >= 0 else -1.0
    )
    fps = steps_actual_real / elapsed
    avg_reward = reward_sum / max(steps_executed, 1)
    status = "early_terminated" if terminated else "finished"
    max_step_reward = _max_step_reward(cfg)
    max_total_reward = max(total_steps, 1) * max_step_reward
    execute_reward_max = max(execute_step_limit, 1) * max_step_reward
    execute_steps_actual_real = float(execute_steps) / float(real_steps)
    execute_steps_max_real = float(execute_step_limit) / float(real_steps)
    success = 1 if steps_executed >= success_steps else 0

    summary = {
        "status": status,
        "fps": float(fps),
        "reward_actual": float(reward_sum_nonterminal),
        "reward_max": float(max_total_reward),
        "steps_actual": float(steps_actual_real),
        "steps_max": float(steps_max_real),
        "execute_reward_actual": float(execute_reward_sum_nonterminal),
        "execute_reward_max": float(execute_reward_max),
        "execute_steps_actual": float(execute_steps_actual_real),
        "execute_steps_max": float(execute_steps_max_real),
        "success_steps": float(success_steps) / float(real_steps),
        "success": float(success),
        "reward_total": float(reward_sum),
        "reward_avg": float(avg_reward),
        "reward_last": float(reward_last),
        "terminal_bonus": float(terminal_bonus),
        "termination_step": float(termination_step_real),
    }
    if print_line:
        print(
            "[RL] "
            f"task={args.task} "
            f"hand={args.active_hand} "
            f"status={summary['status']} "
            f"fps={summary['fps']:.2f} "
            f"reward_actual={summary['reward_actual']:.6f} "
                f"reward_max={summary['reward_max']:.6f} "
                f"steps_actual={summary['steps_actual']:.2f} "
                f"steps_max={summary['steps_max']:.2f} "
                f"execute_reward_actual={summary['execute_reward_actual']:.6f} "
                f"execute_reward_max={summary['execute_reward_max']:.6f} "
                f"execute_steps_actual={summary['execute_steps_actual']:.2f} "
                f"execute_steps_max={summary['execute_steps_max']:.2f} "
                f"success_steps={summary['success_steps']:.2f} "
                f"success={int(summary['success'])} "
                f"reward_total={summary['reward_total']:.6f} "
            f"reward_avg={summary['reward_avg']:.6f} "
            f"steps={summary['steps_actual']:.2f}/{summary['steps_max']:.2f} "
            f"termination_step={summary['termination_step']:.2f} "
            f"reward_last={summary['reward_last']:.6f} "
            f"terminal_bonus={summary['terminal_bonus']:.6f}"
        )

    try:
        policy.train(policy_was_training)
    except Exception:
        pass
    return summary


def _install_training_viz_hook(
    runner,
    env: MJWPRslResidualEnv,
    args: RLTrainArgs,
    cfg: Config,
    log_dir: Path,
    viz_interval: int,
    rollout_video_ctrl_steps: int,
    wandb_run=None,
    checkpoint_save_fn=None,
    resume_start_iter: int = 0,
):
    """Hook rsl_rl logger.log to persist reward snapshots and optional plots."""
    if viz_interval <= 0:
        return [], lambda: (None, -1), lambda: 0.0

    viz_dir = log_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    best_eval_summary: dict[str, float] | None = None
    best_eval_iter: int = -1
    first_eval_summary: dict[str, float] | None = None
    first_success_iter: int | None = None
    eval_count = 0
    plateau_best_reward = float("-inf")
    plateau_evals_since_improve = 0
    train_rew_print_ema = float("nan")
    train_extra_rew_print_ema = float("nan")
    orig_log = runner.logger.log
    warned_video_backend = {"done": False}
    latest_rollout_video_path: Path | None = None
    hook_excluded_time = 0.0
    last_hook_end_time: float | None = None
    train_elapsed_at_hook = 0.0
    farthest_steps_seen = float("-inf")
    first_farthest_elapsed = 0.0
    train_trigger_enabled = int(args.train_success_trigger_eval) != 0
    train_trigger_count = max(1, int(args.train_success_trigger_count))
    train_trigger_ctrl_steps = int(args.train_success_trigger_ctrl_steps)
    train_trigger_backoff_max = max(1, int(args.train_success_trigger_backoff_max))
    train_trigger_fail_streak = 0
    train_trigger_hits_since_eval = 0
    if train_trigger_ctrl_steps <= 0:
        train_trigger_ctrl_steps = int(args.export_execute_ctrl_steps)
    if train_trigger_ctrl_steps <= 0:
        train_trigger_ctrl_steps = int(args.success_eval_ctrl_steps)
    train_max_start = max(
        0,
        int(getattr(env, "max_ref_idx", int(env.qpos_ref.shape[0]) - 1))
        - int(cfg.horizon_steps)
        - int(cfg.ctrl_steps)
        - int(args.max_episode_ctrl_steps) * int(cfg.ctrl_steps)
        - 2,
    )
    train_observable_end = min(
        max(1, int(getattr(env, "max_ref_idx", int(env.qpos_ref.shape[0]) - 1)) - 2),
        int(train_max_start) + int(args.max_episode_ctrl_steps) * int(cfg.ctrl_steps),
    )
    train_eval_total_steps = max(
        0,
        int(env.qpos_ref.shape[0]) - int(cfg.horizon_steps) - int(cfg.ctrl_steps),
    )
    train_trigger_target_sim_step = 0
    if train_trigger_enabled and train_trigger_ctrl_steps > 0:
        requested_target = train_trigger_ctrl_steps * max(1, int(cfg.ref_steps))
        train_trigger_target_sim_step = min(
            int(requested_target),
            max(1, int(train_eval_total_steps) - 1),
            max(1, int(train_observable_end)),
        )
    if train_trigger_target_sim_step > 0:
        setattr(env, "train_success_target_sim_step", int(train_trigger_target_sim_step))

    history_path = viz_dir / "reward_history.json"

    def _is_better_eval(candidate: dict[str, float], current_best: dict[str, float] | None) -> bool:
        if current_best is None:
            return True
        candidate_key = (
            float(candidate.get("success", 0.0)) > 0.0,
            float(candidate.get("steps_actual", 0.0)),
            float(candidate.get("reward_actual", float("-inf"))),
        )
        best_key = (
            float(current_best.get("success", 0.0)) > 0.0,
            float(current_best.get("steps_actual", 0.0)),
            float(current_best.get("reward_actual", float("-inf"))),
        )
        return candidate_key > best_key

    if resume_start_iter > 0 and history_path.exists():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            history = [
                dict(item)
                for item in payload.get("history", [])
                if float(item.get("iter", -1)) < float(resume_start_iter)
            ]
            eval_history = [
                dict(item)
                for item in payload.get("eval_history", [])
                if float(item.get("iter", -1)) < float(resume_start_iter)
            ]
            for item in eval_history:
                item_iter = int(float(item.get("iter", -1)))
                candidate = {
                    "status": "resumed_best",
                    "fps": 0.0,
                    "reward_actual": float(item.get("reward_actual", float("-inf"))),
                    "reward_max": float(item.get("reward_max", 0.0)),
                    "steps_actual": float(item.get("steps_actual", 0.0)),
                    "steps_max": float(item.get("steps_max", 0.0)),
                    "success": float(item.get("success", 0.0)),
                    "reward_total": float(item.get("reward_actual", 0.0)),
                    "reward_avg": float(item.get("reward_actual", 0.0)),
                    "reward_last": 0.0,
                    "terminal_bonus": 0.0,
                    "termination_step": float(item.get("steps_actual", 0.0)),
                    "time_to_farthest": float(item.get("time_to_farthest", 0.0)),
                }
                is_best = _is_better_eval(candidate, best_eval_summary)
                if is_best:
                    best_eval_summary = candidate
                    best_eval_iter = item_iter
                if (
                    first_success_iter is None
                    and (
                        float(item.get("success", 0.0)) > 0.0
                        or (
                            float(item.get("steps_actual", 0.0)) >= float(item.get("steps_max", 0.0))
                            and float(item.get("steps_max", 0.0)) > 0.0
                        )
                    )
                ):
                    first_success_iter = item_iter
            loguru.logger.info(
                "[RL][resume] restored viz history: train={} eval={} best_iter={}",
                len(history),
                len(eval_history),
                best_eval_iter,
            )
        except Exception as e:
            loguru.logger.warning("[RL][resume] failed to restore viz history: {}", e)

    def _finish_hook_timing(hook_t0: float) -> None:
        nonlocal hook_excluded_time, last_hook_end_time
        hook_end = time.perf_counter()
        hook_excluded_time += hook_end - hook_t0
        last_hook_end_time = hook_end

    def _save_eval_checkpoint(it: int, is_best: bool):
        if checkpoint_save_fn is None:
            return
        try:
            ckpt_dir = log_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_save_fn(ckpt_dir / "eval_latest.pt")
            checkpoint_save_fn(ckpt_dir / f"eval_iter_{it}.pt")
            if is_best:
                checkpoint_save_fn(ckpt_dir / "eval_best.pt")
            loguru.logger.info(
                "[RL][ckpt] saved eval checkpoint iter={} best={} dir={}",
                it,
                int(is_best),
                ckpt_dir,
            )
        except Exception as e:
            loguru.logger.warning("eval checkpoint save failed at iter {}: {}", it, e)

    def _log_wandb_scalars(it: int, metrics: dict[str, float], prefix: str | None = None):
        if wandb_run is None:
            return
        payload = {}
        for key, value in metrics.items():
            try:
                value_f = float(value)
            except Exception:
                continue
            if not math.isfinite(value_f):
                continue
            payload[f"{prefix}/{key}" if prefix else key] = value_f
        if payload:
            wandb_run.log(payload, step=it)

    def _wandb_upload(it: int, upload_video: bool):
        if wandb_run is None:
            return
        media = {}
        reward_curve_png = viz_dir / "reward_curve_latest.png"
        if reward_curve_png.exists():
            try:
                import wandb  # type: ignore

                media["reward_curve"] = wandb.Image(str(reward_curve_png))
            except Exception as e:
                loguru.logger.warning("wandb curve upload failed at iter {}: {}", it, e)
        steps_curve_png = viz_dir / "steps_curve_latest.png"
        if upload_video and steps_curve_png.exists():
            try:
                import wandb  # type: ignore

                media["steps_curve"] = wandb.Image(str(steps_curve_png))
            except Exception as e:
                loguru.logger.warning("wandb steps curve upload failed at iter {}: {}", it, e)
        if upload_video and latest_rollout_video_path is not None and latest_rollout_video_path.exists():
            try:
                import wandb  # type: ignore

                media["rollout_video"] = wandb.Video(str(latest_rollout_video_path), fps=max(1, int(round(1.0 / env.cfg.sim_dt))), format="mp4")
            except Exception as e:
                loguru.logger.warning("wandb video upload failed at iter {}: {}", it, e)
        if media:
            wandb_run.log(media, step=it)

    def _save_best_trajectory_payload(payload: dict) -> None:
        path = log_dir / "best_trajectory.json"
        json_payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
        path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        loguru.logger.info(
            "[RL][best_trajectory] saved frames={} path={}",
            int(payload.get("num_frames", 0)),
            path,
        )

    def _save_best_rollout_npz(rollout_payload: dict | None) -> None:
        if not rollout_payload:
            return
        full_traj = rollout_payload.get("_full_traj")
        if not full_traj:
            return
        if int(args.export_execute_ctrl_steps) > 0:
            max_sim_step = int(args.export_execute_ctrl_steps) * max(1, int(env.cfg.ref_steps))
            full_traj = [
                item for item in full_traj if int(item.get("sim_step", 0)) <= max_sim_step
            ]
            if not full_traj:
                return
        try:
            path = log_dir / "best_rollout.npz"
            np.savez(
                path,
                sim_step=np.asarray([item["sim_step"] for item in full_traj], dtype=np.int64),
                qpos=np.asarray([item["qpos"] for item in full_traj], dtype=np.float64),
                qvel=np.asarray([item["qvel"] for item in full_traj], dtype=np.float64),
                ctrl=np.asarray([item["ctrl"] for item in full_traj], dtype=np.float64),
            )
            loguru.logger.info("[RL][best_rollout] saved frames={} path={}", len(full_traj), path)
        except Exception as e:
            loguru.logger.warning("best rollout npz save failed: {}", e)

    def _save_rollout_video(it: int):
        nonlocal latest_rollout_video_path
        if int(rollout_video_ctrl_steps) < 0:
            latest_rollout_video_path = None
            return None

        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            if not warned_video_backend["done"]:
                loguru.logger.warning(
                    "rollout video skipped: `imageio` not installed in current Python env. "
                    "Install with `pip install imageio`."
                )
                warned_video_backend["done"] = True
            imageio = None

        # Snapshot training env state and RL bookkeeping, then restore after rendering rollout.
        saved_state = save_state(env.env)
        saved_rl = {
            "_sim_step": int(env._sim_step),
            "_start_sim_step": int(env._start_sim_step),
            "_episode_ctrl_steps": int(env._episode_ctrl_steps),
            "_sim_step_buf": env._sim_step_buf.clone(),
            "_start_sim_step_buf": env._start_sim_step_buf.clone(),
            "_episode_ctrl_steps_buf": env._episode_ctrl_steps_buf.clone(),
            "rew_buf": env.rew_buf.clone(),
            "reset_buf": env.reset_buf.clone(),
            "episode_length_buf": env.episode_length_buf.clone(),
            "episode_return_buf": env.episode_return_buf.clone(),
            "episode_base_return_buf": env.episode_base_return_buf.clone(),
            "episode_extra_return_buf": env.episode_extra_return_buf.clone(),
            "obs_buf": None if env.obs_buf is None else env.obs_buf.clone(),
            "extras": dict(env.extras),
        }

        frames = []
        renderer = None
        best_trajectory_payload = None
        try:
            if imageio is not None:
                # Ensure offscreen framebuffer is large enough for requested render size.
                env.mj_model.vis.global_.offwidth = max(int(env.mj_model.vis.global_.offwidth), 720)
                env.mj_model.vis.global_.offheight = max(int(env.mj_model.vis.global_.offheight), 480)
                try:
                    renderer = mujoco.Renderer(env.mj_model, height=480, width=720)
                except Exception:
                    # Fallback to a conservative size if backend/model limits still reject 720x480.
                    renderer = mujoco.Renderer(env.mj_model, height=360, width=640)
            mj_data_ref = mujoco.MjData(env.mj_model)
            policy = runner.alg.get_policy()
            policy_was_training = bool(getattr(policy, "training", False))
            policy.eval()
            raw_data = np.load(cfg.data_path)
            raw_frame_count = int(raw_data["qpos"].shape[0])
            if "ctrl" in raw_data:
                raw_ctrl_np = np.asarray(raw_data["ctrl"], dtype=np.float64)
            else:
                raw_ctrl_np = np.asarray(raw_data["qpos"][:, : -env.nq_obj], dtype=np.float64)
            export_right_arm6 = (
                str(getattr(env.cfg, "robot_type", "")) == "xhand"
                and str(getattr(env.cfg, "embodiment_type", "")) in {"right", "bimanual"}
                and str(getattr(env.cfg, "active_hand", "right")) != "left"
            )
            q_right_link_palm = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if export_right_arm6:
                sid_right_palm = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
                if sid_right_palm < 0:
                    raise ValueError("right_palm site not found; cannot export right_arm6 trajectory")
                q_right_link_palm = np.asarray(env.mj_model.site_quat[sid_right_palm], dtype=np.float64)

            def _compact_traj_frame(
                frame_index: int,
                sim_step: int,
                ctrl: torch.Tensor,
                render_env_idx: int,
                action_override: np.ndarray | None = None,
            ):
                qpos_all = get_qpos(env.cfg, env.env)
                qpos = qpos_all[render_env_idx].detach().cpu().numpy()
                ctrl_np = ctrl[render_env_idx].detach().cpu().numpy()

                robot_qpos = qpos[: -env.nq_obj]
                obj_qpos = qpos[-env.nq_obj :]
                action = ctrl_np
                if getattr(env, "_single_hand_mode", False):
                    robot_half_qpos = robot_qpos.shape[0] // 2
                    if env.cfg.active_hand == "right":
                        robot_qpos = robot_qpos[:robot_half_qpos]
                    else:
                        robot_qpos = robot_qpos[robot_half_qpos:]
                    obj_qpos = obj_qpos[env._obj_qpos_slice]
                    action = ctrl_np[env._ctrl_slice]

                if action_override is not None:
                    action = np.asarray(action_override, dtype=np.float64).reshape(-1)
                action = np.asarray(action, dtype=np.float64).reshape(-1)
                proprio = np.asarray(robot_qpos, dtype=np.float64).reshape(-1)
                if action.size != 18 or proprio.size != 18:
                    raise ValueError(
                        "Expected compact action/proprio dims of 18, "
                        f"got action={action.size} proprio={proprio.size}"
                    )
                if export_right_arm6:
                    action = _replace_eef_with_right_arm6(action, q_right_link_palm)
                    proprio = _replace_eef_with_right_arm6(proprio, q_right_link_palm)

                return (
                    int(sim_step),
                    {
                        "frame_index": int(frame_index),
                        "right_arm6": action[:6].astype(float).tolist(),
                        "right_arm6_proprio": proprio[:6].astype(float).tolist(),
                        "xhand": action[6:18].astype(float).tolist(),
                        "xhand_proprio": proprio[6:18].astype(float).tolist(),
                        "object": _object_qpos_to_pose6(obj_qpos),
                    },
                )

            def _compact_ctrl_np(ctrl_np: np.ndarray) -> np.ndarray:
                ctrl_np = np.asarray(ctrl_np, dtype=np.float64).reshape(-1)
                if getattr(env, "_single_hand_mode", False):
                    ctrl_np = ctrl_np[env._ctrl_slice]
                if ctrl_np.size != 18:
                    raise ValueError(f"Expected compact ctrl dim 18, got {ctrl_np.size}")
                return ctrl_np

            def _raw_ctrl_frame(raw_frame_index: int) -> np.ndarray:
                raw_action = np.asarray(raw_ctrl_np[int(raw_frame_index)], dtype=np.float64).reshape(-1)
                if getattr(env, "_single_hand_mode", False) and raw_action.size != 18:
                    raw_action = raw_action[env._ctrl_slice]
                if raw_action.size != 18:
                    raise ValueError(
                        "Expected raw ctrl dim 18 for trajectory export, "
                        f"got {raw_action.size}"
                    )
                return raw_action

            def _sim_traj_sample(
                frame_index: int,
                sim_step: int,
                ctrl: torch.Tensor,
                base_ctrl_idx: int,
                render_env_idx: int,
            ) -> dict:
                _step, frame = _compact_traj_frame(
                    frame_index,
                    sim_step,
                    ctrl,
                    render_env_idx,
                )
                ctrl_np = ctrl[render_env_idx].detach().cpu().numpy()
                base_ctrl = env.ctrl_ref[env._clamp_ref_idx(int(base_ctrl_idx))]
                base_ctrl_np = base_ctrl.detach().cpu().numpy()
                residual = _compact_ctrl_np(ctrl_np) - _compact_ctrl_np(base_ctrl_np)
                qpos_all = get_qpos(env.cfg, env.env)
                qvel_all = get_qvel(env.cfg, env.env)
                return {
                    "sim_step": int(sim_step),
                    "frame": frame,
                    "residual": residual,
                    "qpos": qpos_all[render_env_idx].detach().cpu().numpy().astype(np.float64),
                    "qvel": qvel_all[render_env_idx].detach().cpu().numpy().astype(np.float64),
                    "ctrl": np.asarray(ctrl_np, dtype=np.float64).reshape(-1),
                }

            def _resample_to_input_timestep(sim_traj: list[dict]) -> list[dict]:
                if raw_frame_count <= 0:
                    return []
                if not sim_traj:
                    return []
                sim_traj_by_step = {int(item["sim_step"]): item for item in sim_traj}
                available_steps = sorted(sim_traj_by_step)
                if not available_steps:
                    return []
                last_available = int(available_steps[-1])
                rollout_ref_steps = max(
                    1,
                    int(env.qpos_ref.shape[0]) - int(env.cfg.horizon_steps) - int(env.cfg.ctrl_steps),
                )
                input_ref_stride = max(1, int(round(float(rollout_ref_steps) / float(raw_frame_count))))
                cursor = 0
                out = []
                for raw_frame_index in range(raw_frame_count):
                    target_step = raw_frame_index * input_ref_stride
                    if target_step > last_available:
                        break
                    while (
                        cursor + 1 < len(available_steps)
                        and abs(available_steps[cursor + 1] - target_step)
                        <= abs(available_steps[cursor] - target_step)
                    ):
                        cursor += 1
                    sample = sim_traj_by_step[available_steps[cursor]]
                    frame = dict(sample["frame"])
                    action = _raw_ctrl_frame(raw_frame_index) + np.asarray(sample["residual"])
                    if export_right_arm6:
                        action_export = _replace_eef_with_right_arm6(action, q_right_link_palm)
                    else:
                        action_export = action
                    frame["right_arm6"] = action_export[:6].astype(float).tolist()
                    frame["xhand"] = action_export[6:18].astype(float).tolist()
                    item = frame
                    item["frame_index"] = int(raw_frame_index)
                    out.append(item)
                return out

            def _run_rollout(score_only: bool, render_env_idx: int = 0, collect_trajectory: bool = False):
                rollout_rewards = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
                rollout_alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
                rollout_term_sim_step = torch.full(
                    (env.num_envs,),
                    fill_value=-1,
                    device=env.device,
                    dtype=torch.long,
                )
                env._set_state_from_ref(0)
                env._sim_step = 0
                env._start_sim_step = 0
                env._episode_ctrl_steps = 0
                env._sim_step_buf.zero_()
                env._start_sim_step_buf.zero_()
                env._episode_ctrl_steps_buf.zero_()
                env.obs_buf = env._get_obs()
                frozen_qpos = get_qpos(env.cfg, env.env).detach().clone()
                frozen_qvel = get_qvel(env.cfg, env.env).detach().clone()
                local_frames: list[np.ndarray] = []
                local_sim_traj: list[dict] = []
                max_ctrl_iters = (
                    None
                    if int(rollout_video_ctrl_steps) <= 0
                    else max(1, int(rollout_video_ctrl_steps))
                )
                ctrl_iter = 0

                with torch.no_grad():
                    while True:
                        if max_ctrl_iters is not None and ctrl_iter >= max_ctrl_iters:
                            break
                        obs_td = env.get_observations()
                        actions = policy(obs_td)
                        if actions.ndim == 1:
                            actions = actions.unsqueeze(0)
                        actions = torch.clamp(
                            actions.to(env.device),
                            -env.env_cfg.action_clip,
                            env.env_cfg.action_clip,
                        )
                        actions = torch.where(
                            rollout_alive.unsqueeze(1),
                            actions,
                            torch.zeros_like(actions),
                        )

                        base_ctrl_idx = env._clamp_ref_idx(env._sim_step)
                        ctrl, _, _ = env.residual_actions_to_ctrl(
                            actions,
                            base_ctrl_idx,
                        )

                        if collect_trajectory and not local_sim_traj:
                            local_sim_traj.append(
                                _sim_traj_sample(
                                    len(local_sim_traj),
                                    int(env._sim_step),
                                    ctrl,
                                    int(base_ctrl_idx),
                                    render_env_idx,
                                )
                            )

                        for sub in range(env.cfg.ctrl_steps):
                            step_env(env.cfg, env.env, ctrl)
                            env._sim_step += 1
                            env._sim_step_buf += 1
                            ref_idx = max(0, min(env._sim_step, env.max_ref_idx))
                            reset_inactive_side_to_ref(
                                env.cfg,
                                env.env,
                                env.qpos_ref[ref_idx],
                                env.qvel_ref[ref_idx],
                            )
                            ref_step, _ = env._ref_at(env._sim_step)
                            rew, _ = get_reward(env.cfg, env.env, ref_step)
                            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
                            rollout_rewards += (rew * rollout_alive.to(rew.dtype)) / max(1, env.cfg.ctrl_steps)

                            term = get_terminate(env.cfg, env.env, ref_step)
                            if term.any():
                                term_rew, _ = get_terminal_reward(env.cfg, env.env, ref_step)
                                term_rew = torch.nan_to_num(term_rew, nan=0.0, posinf=0.0, neginf=0.0)
                                newly_term = term & rollout_alive
                                rollout_rewards += (term_rew * newly_term.to(term_rew.dtype)) / max(1, env.cfg.ctrl_steps)
                                if newly_term.any():
                                    qpos_now = get_qpos(env.cfg, env.env).detach().clone()
                                    qvel_now = get_qvel(env.cfg, env.env).detach().clone()
                                    frozen_qpos[newly_term] = qpos_now[newly_term]
                                    frozen_qvel[newly_term] = qvel_now[newly_term]
                                rollout_term_sim_step[newly_term] = env._sim_step
                                rollout_alive[newly_term] = False

                            dead = ~rollout_alive
                            if dead.any():
                                qpos_now = get_qpos(env.cfg, env.env).detach().clone()
                                qvel_now = get_qvel(env.cfg, env.env).detach().clone()
                                qpos_now[dead] = frozen_qpos[dead]
                                qvel_now[dead] = frozen_qvel[dead]
                                wp.copy(env.env.data_wp.qpos, wp.from_torch(qpos_now))
                                wp.copy(env.env.data_wp.qvel, wp.from_torch(qvel_now))

                            if not score_only:
                                qpos_all = get_qpos(env.cfg, env.env)
                                qpos_render = qpos_all[render_env_idx].detach().cpu().numpy()
                                env.mj_data.qpos[:] = qpos_render
                                env.mj_data.time = float(env._sim_step * env.cfg.sim_dt)
                                mujoco.mj_forward(env.mj_model, env.mj_data)

                                if renderer is not None:
                                    mj_data_ref.qpos[:] = env.qpos_ref[ref_idx].detach().cpu().numpy()
                                    mujoco.mj_forward(env.mj_model, mj_data_ref)
                                    local_frames.append(
                                        render_image(env.cfg, renderer, env.mj_model, env.mj_data, mj_data_ref)
                                    )

                            if collect_trajectory:
                                local_sim_traj.append(
                                    _sim_traj_sample(
                                        len(local_sim_traj),
                                        int(env._sim_step),
                                        ctrl,
                                        int(base_ctrl_idx),
                                        render_env_idx,
                                    )
                                )

                            if env._sim_step >= env.max_ref_idx - 1:
                                break
                            if (not score_only) and (not bool(rollout_alive[render_env_idx].item())):
                                # Stop rendering once the selected env terminates.
                                break

                        env._episode_ctrl_steps += 1
                        env._episode_ctrl_steps_buf += 1
                        env.obs_buf = env._get_obs()
                        if env._sim_step >= env.max_ref_idx - 1:
                            break
                        if not rollout_alive.any():
                            break
                        if (not score_only) and (not bool(rollout_alive[render_env_idx].item())):
                            break
                        ctrl_iter += 1

                # If never terminated, treat as surviving until the current rollout end.
                rollout_len_sim = rollout_term_sim_step.clone()
                alive_end = rollout_len_sim < 0
                rollout_len_sim[alive_end] = int(env._sim_step)
                return (
                    rollout_rewards,
                    local_frames,
                    rollout_len_sim,
                    local_sim_traj,
                )

            # Pass 1: score all parallel envs, choose the highest-score rollout.
            score_rewards, _, score_lengths, _ = _run_rollout(score_only=True)
            best_env_idx = int(torch.argmax(score_rewards).item())
            best_len = int(score_lengths[best_env_idx].item())
            best_reward = float(score_rewards[best_env_idx].item())

            # Pass 2: rerun and render only the best env (much cheaper than rendering all envs).
            _, frames, _, compact_traj = _run_rollout(
                score_only=False,
                render_env_idx=best_env_idx,
                collect_trajectory=True,
            )
            input_timestep_traj = _resample_to_input_timestep(compact_traj)
            if int(args.export_execute_ctrl_steps) > 0:
                input_timestep_traj = input_timestep_traj[: int(args.export_execute_ctrl_steps)]
            best_trajectory_payload = {
                "num_frames": len(input_timestep_traj),
                "fields": {
                    "right_arm6": "command pose for link_right_arm_6, [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]",
                    "right_arm6_proprio": "replayed MuJoCo body pose for link_right_arm_6, [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]",
                    "xhand": f"right XHand command qpos, joints={_RIGHT_XHAND_JOINTS}",
                    "xhand_proprio": f"replayed right XHand qpos, joints={_RIGHT_XHAND_JOINTS}",
                    "object": "right object pose6 [x,y,z,axis_angle_x,axis_angle_y,axis_angle_z]",
                },
                "right_xhand_joints": list(_RIGHT_XHAND_JOINTS),
                "traj": input_timestep_traj,
                "_full_traj": [
                    {
                        "sim_step": int(item["sim_step"]),
                        "qpos": np.asarray(item["qpos"], dtype=np.float64).tolist(),
                        "qvel": np.asarray(item["qvel"], dtype=np.float64).tolist(),
                        "ctrl": np.asarray(item["ctrl"], dtype=np.float64).tolist(),
                    }
                    for item in compact_traj
                ],
            }

            if imageio is not None and frames:
                video_path = viz_dir / f"rollout_iter_{it}.mp4"
                imageio.mimsave(str(video_path), frames, fps=max(1, int(round(1.0 / env.cfg.sim_dt))))
                latest_rollout_video_path = video_path
                if it == 0:
                    imageio.mimsave(str(viz_dir / "rollout_iter_0.mp4"), frames, fps=max(1, int(round(1.0 / env.cfg.sim_dt))))
        except Exception as e:
            loguru.logger.warning("rollout video save failed at iter {}: {}", it, e)
        finally:
            try:
                if "policy" in locals():
                    policy.train(policy_was_training)
            except Exception:
                pass
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass
                renderer = None
            load_state(env.env, saved_state)
            env._sim_step = saved_rl["_sim_step"]
            env._start_sim_step = saved_rl["_start_sim_step"]
            env._episode_ctrl_steps = saved_rl["_episode_ctrl_steps"]
            env._sim_step_buf = saved_rl["_sim_step_buf"]
            env._start_sim_step_buf = saved_rl["_start_sim_step_buf"]
            env._episode_ctrl_steps_buf = saved_rl["_episode_ctrl_steps_buf"]
            env.rew_buf = saved_rl["rew_buf"]
            env.reset_buf = saved_rl["reset_buf"]
            env.episode_length_buf = saved_rl["episode_length_buf"]
            env.episode_return_buf = saved_rl["episode_return_buf"]
            env.episode_base_return_buf = saved_rl["episode_base_return_buf"]
            env.episode_extra_return_buf = saved_rl["episode_extra_return_buf"]
            env.obs_buf = saved_rl["obs_buf"]
            env.extras = saved_rl["extras"]
            qpos_restore = get_qpos(env.cfg, env.env)[0].detach().cpu().numpy()
            env.mj_data.qpos[:] = qpos_restore
            env.mj_data.time = float(env._sim_step * env.cfg.sim_dt)
            mujoco.mj_forward(env.mj_model, env.mj_data)
        return best_trajectory_payload

    def _save_viz_snapshot(it: int):
        payload = {"history": history, "eval_history": eval_history}
        (viz_dir / "reward_history.json").write_text(json.dumps(payload, indent=2))
        try:
            import matplotlib.pyplot as plt  # type: ignore

            xs = np.asarray([h.get("env_steps", h["iter"]) for h in history], dtype=np.float64)
            ys = np.asarray([h["mean_reward"] for h in history], dtype=np.float64)
            y_std = np.asarray([h.get("std_reward", 0.0) for h in history], dtype=np.float64)
            valid = np.isfinite(xs) & np.isfinite(ys)
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            if valid.any():
                x_plot = xs[valid]
                y_plot = ys[valid]
                std_plot = np.nan_to_num(y_std[valid], nan=0.0, posinf=0.0, neginf=0.0)
                ax.plot(x_plot, y_plot, linewidth=2.0, label="mean reward")
                ax.fill_between(
                    x_plot,
                    y_plot - std_plot,
                    y_plot + std_plot,
                    alpha=0.22,
                    linewidth=0.0,
                    label="std",
                )
            ax.set_ylabel("Mean Reward")
            ax.set_xlabel("Number of Env Steps")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(viz_dir / "reward_curve_latest.png", dpi=140)
            plt.close(fig)

            if eval_history:
                xs_eval = [h.get("env_steps", h["iter"]) for h in eval_history]
                ys_steps = [h["steps_actual"] for h in eval_history]
                ys_best = []
                running_best = float("-inf")
                for value in ys_steps:
                    running_best = max(running_best, value)
                    ys_best.append(running_best)

                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                ax.plot(xs_eval, ys_steps, linewidth=2.0, marker="o", label="eval steps")
                ax.plot(xs_eval, ys_best, linewidth=1.8, linestyle="--", marker="s", label="best so far")
                ax.set_ylabel("Steps Reached")
                ax.set_xlabel("Number of Env Steps")
                ax.grid(True, alpha=0.3)
                ax.legend()
                if len(xs_eval) == 1:
                    x0 = xs_eval[0]
                    y0 = ys_steps[0]
                    ax.set_xlim(x0 - 1.0, x0 + 1.0)
                    ax.set_ylim(y0 - 1.0, y0 + 1.0)
                fig.tight_layout()
                fig.savefig(viz_dir / "steps_curve_latest.png", dpi=140)
                plt.close(fig)
        except Exception:
            # Matplotlib is optional; JSON snapshots are always written.
            pass

    def wrapped_log(*log_args, **kwargs):
        nonlocal best_eval_summary, best_eval_iter, first_eval_summary, first_success_iter, eval_count
        nonlocal plateau_best_reward, plateau_evals_since_improve
        nonlocal train_rew_print_ema, train_extra_rew_print_ema, hook_excluded_time
        nonlocal last_hook_end_time, train_elapsed_at_hook
        nonlocal farthest_steps_seen, first_farthest_elapsed
        nonlocal train_trigger_fail_streak, train_trigger_hits_since_eval
        ep_extras_before = list(runner.logger.ep_extras)
        with redirect_stdout(io.StringIO()):
            orig_log(*log_args, **kwargs)
        hook_t0 = time.perf_counter()
        if last_hook_end_time is None:
            train_elapsed_at_hook = 0.0
        else:
            train_elapsed_at_hook += max(0.0, hook_t0 - last_hook_end_time)
        # rsl_rl passes `it` as kwarg; keep a fallback for positional usage.
        it = int(kwargs.get("it", log_args[0] if log_args else runner.current_learning_iteration))
        # rsl_rl's reward buffer is the actual PPO reward, including any extra terms.
        train_total_rew_mean = (
            float(statistics.mean(runner.logger.rewbuffer))
            if len(runner.logger.rewbuffer) > 0
            else float("nan")
        )
        train_total_rew_std = (
            float(statistics.pstdev(runner.logger.rewbuffer))
            if len(runner.logger.rewbuffer) > 1
            else 0.0
        )
        train_total_rew_min = (
            float(min(runner.logger.rewbuffer))
            if len(runner.logger.rewbuffer) > 0
            else float("nan")
        )
        train_total_rew_max = (
            float(max(runner.logger.rewbuffer))
            if len(runner.logger.rewbuffer) > 0
            else float("nan")
        )
        train_base_rew_mean = float("nan")
        train_base_rew_std = float("nan")
        train_base_rew_min = float("nan")
        train_base_rew_max = float("nan")
        train_extra_rew_mean = float("nan")
        train_extra_rew_std = float("nan")
        train_extra_rew_min = float("nan")
        train_extra_rew_max = float("nan")
        extras_r_mean = float("nan")
        extras_r_std = float("nan")
        extras_r_min = float("nan")
        extras_r_max = float("nan")
        train_sim_step_max = float("nan")
        train_success_count = 0
        if ep_extras_before:
            rs = []
            r_stds = []
            r_mins = []
            r_maxs = []
            base_rs = []
            base_stds = []
            base_mins = []
            base_maxs = []
            extra_rs = []
            extra_stds = []
            extra_mins = []
            extra_maxs = []
            sim_step_maxs = []
            sim_step_success_counts = []
            for ep in ep_extras_before:
                if "r" in ep:
                    rs.append(float(ep["r"]))
                elif "/episode_reward" in ep:
                    rs.append(float(ep["/episode_reward"]))
                if "r_std" in ep:
                    r_stds.append(float(ep["r_std"]))
                elif "/episode_reward_std" in ep:
                    r_stds.append(float(ep["/episode_reward_std"]))
                if "r_min" in ep:
                    r_mins.append(float(ep["r_min"]))
                elif "/episode_reward_min" in ep:
                    r_mins.append(float(ep["/episode_reward_min"]))
                if "r_max" in ep:
                    r_maxs.append(float(ep["r_max"]))
                elif "/episode_reward_max" in ep:
                    r_maxs.append(float(ep["/episode_reward_max"]))
                if "base_r" in ep:
                    base_rs.append(float(ep["base_r"]))
                elif "/episode_reward_base" in ep:
                    base_rs.append(float(ep["/episode_reward_base"]))
                if "base_r_std" in ep:
                    base_stds.append(float(ep["base_r_std"]))
                elif "/episode_reward_base_std" in ep:
                    base_stds.append(float(ep["/episode_reward_base_std"]))
                if "base_r_min" in ep:
                    base_mins.append(float(ep["base_r_min"]))
                elif "/episode_reward_base_min" in ep:
                    base_mins.append(float(ep["/episode_reward_base_min"]))
                if "base_r_max" in ep:
                    base_maxs.append(float(ep["base_r_max"]))
                elif "/episode_reward_base_max" in ep:
                    base_maxs.append(float(ep["/episode_reward_base_max"]))
                if "extra_r" in ep:
                    extra_rs.append(float(ep["extra_r"]))
                elif "/episode_reward_extra" in ep:
                    extra_rs.append(float(ep["/episode_reward_extra"]))
                if "extra_r_std" in ep:
                    extra_stds.append(float(ep["extra_r_std"]))
                elif "/episode_reward_extra_std" in ep:
                    extra_stds.append(float(ep["/episode_reward_extra_std"]))
                if "extra_r_min" in ep:
                    extra_mins.append(float(ep["extra_r_min"]))
                elif "/episode_reward_extra_min" in ep:
                    extra_mins.append(float(ep["/episode_reward_extra_min"]))
                if "extra_r_max" in ep:
                    extra_maxs.append(float(ep["extra_r_max"]))
                elif "/episode_reward_extra_max" in ep:
                    extra_maxs.append(float(ep["/episode_reward_extra_max"]))
                if "sim_step_max" in ep:
                    sim_step_maxs.append(float(ep["sim_step_max"]))
                elif "/sim_step_max" in ep:
                    sim_step_maxs.append(float(ep["/sim_step_max"]))
                if "sim_step_success_count" in ep:
                    sim_step_success_counts.append(float(ep["sim_step_success_count"]))
                elif "/sim_step_success_count" in ep:
                    sim_step_success_counts.append(float(ep["/sim_step_success_count"]))
            if rs:
                extras_r_mean = float(sum(rs) / len(rs))
                extras_r_std = float(sum(r_stds) / len(r_stds)) if r_stds else (
                    float(statistics.pstdev(rs)) if len(rs) > 1 else 0.0
                )
                extras_r_min = float(min(r_mins)) if r_mins else float(min(rs))
                extras_r_max = float(max(r_maxs)) if r_maxs else float(max(rs))
            if base_rs:
                train_base_rew_mean = float(sum(base_rs) / len(base_rs))
                train_base_rew_std = float(sum(base_stds) / len(base_stds)) if base_stds else (
                    float(statistics.pstdev(base_rs)) if len(base_rs) > 1 else 0.0
                )
                train_base_rew_min = float(min(base_mins)) if base_mins else float(min(base_rs))
                train_base_rew_max = float(max(base_maxs)) if base_maxs else float(max(base_rs))
            if extra_rs:
                train_extra_rew_mean = float(sum(extra_rs) / len(extra_rs))
                train_extra_rew_std = float(sum(extra_stds) / len(extra_stds)) if extra_stds else (
                    float(statistics.pstdev(extra_rs)) if len(extra_rs) > 1 else 0.0
                )
                train_extra_rew_min = float(min(extra_mins)) if extra_mins else float(min(extra_rs))
                train_extra_rew_max = float(max(extra_maxs)) if extra_maxs else float(max(extra_rs))
            if sim_step_maxs:
                train_sim_step_max = float(max(sim_step_maxs))
            if sim_step_success_counts:
                train_success_count = int(sum(sim_step_success_counts))
        if train_base_rew_mean != train_base_rew_mean:
            if train_extra_rew_mean == train_extra_rew_mean and train_total_rew_mean == train_total_rew_mean:
                train_base_rew_mean = train_total_rew_mean - train_extra_rew_mean
                train_base_rew_std = train_total_rew_std
                train_base_rew_min = train_total_rew_min
                train_base_rew_max = train_total_rew_max

        history.append(
            {
                "iter": float(it),
                "env_steps": float((it + 1) * int(args.num_envs) * int(args.num_steps_per_env)),
                "mean_reward": train_base_rew_mean,
                "std_reward": train_base_rew_std,
                "min_reward": train_base_rew_min,
                "max_reward": train_base_rew_max,
                "mean_extra_reward": train_extra_rew_mean,
                "std_extra_reward": train_extra_rew_std,
                "min_extra_reward": train_extra_rew_min,
                "max_extra_reward": train_extra_rew_max,
                "mean_total_reward": train_total_rew_mean,
                "std_total_reward": train_total_rew_std,
                "min_total_reward": train_total_rew_min,
                "max_total_reward": train_total_rew_max,
                "extras_r_mean": extras_r_mean,
                "extras_r_std": extras_r_std,
                "extras_r_min": extras_r_min,
                "extras_r_max": extras_r_max,
                "train_sim_step_max": train_sim_step_max,
                "train_success_count": float(train_success_count),
            }
        )
        _log_wandb_scalars(
            it,
            {
                "mean_reward": train_base_rew_mean,
                "std_reward": train_base_rew_std,
                "min_reward": train_base_rew_min,
                "max_reward": train_base_rew_max,
                "extra_reward": train_extra_rew_mean,
                "extra_reward_std": train_extra_rew_std,
                "extra_reward_min": train_extra_rew_min,
                "extra_reward_max": train_extra_rew_max,
                "total_reward": train_total_rew_mean,
                "total_reward_std": train_total_rew_std,
                "total_reward_min": train_total_rew_min,
                "total_reward_max": train_total_rew_max,
                "extras_reward": extras_r_mean,
                "extras_reward_std": extras_r_std,
                "extras_reward_min": extras_r_min,
                "extras_reward_max": extras_r_max,
                "sim_step_max": train_sim_step_max,
                "success_count": float(train_success_count),
            },
            prefix="train",
        )
        _save_viz_snapshot(it)
        reached_success_before_eval = (
            best_eval_summary is not None
            and float(best_eval_summary["success"]) > 0.0
        )
        train_plateau_trigger = False
        train_plateau_improvement = float("nan")
        train_plateau_window = max(0, int(args.train_plateau_after_success_window))
        if reached_success_before_eval and train_plateau_window > 0 and len(history) > train_plateau_window:
            cur_train_reward = float(history[-1].get("mean_reward", float("nan")))
            prev_train_reward = float(history[-1 - train_plateau_window].get("mean_reward", float("nan")))
            train_plateau_improvement = cur_train_reward - prev_train_reward
            train_plateau_trigger = (
                math.isfinite(train_plateau_improvement)
                and train_plateau_improvement < max(0.0, float(args.train_plateau_after_success_min_delta))
            )
        train_success_eval_candidate = (
            train_trigger_enabled
            and train_trigger_target_sim_step > 0
            and train_sim_step_max == train_sim_step_max
            and train_sim_step_max >= float(train_trigger_target_sim_step)
        )
        train_success_eval_trigger = False
        if train_success_eval_candidate:
            train_trigger_hits_since_eval += 1
            train_trigger_required_hits = min(
                int(train_trigger_backoff_max),
                2 ** max(0, int(train_trigger_fail_streak) - 1),
            )
            train_success_eval_trigger = train_trigger_hits_since_eval >= train_trigger_required_hits
        else:
            train_trigger_required_hits = 1
        if train_success_eval_trigger:
            train_trigger_hits_since_eval = 0
            loguru.logger.info(
                "[RL][train-trigger] iter={} train_sim_step_max={} target_sim_step={} "
                "fail_streak={} backoff_hits={}/{} -> eval",
                it,
                float(train_sim_step_max),
                int(train_trigger_target_sim_step),
                int(train_trigger_fail_streak),
                int(train_trigger_required_hits),
                int(train_trigger_required_hits),
            )
        elif train_success_eval_candidate:
            loguru.logger.info(
                "[RL][train-trigger] iter={} train_sim_step_max={} target_sim_step={} "
                "fail_streak={} backoff_hits={}/{} -> skip eval",
                it,
                float(train_sim_step_max),
                int(train_trigger_target_sim_step),
                int(train_trigger_fail_streak),
                int(train_trigger_hits_since_eval),
                int(train_trigger_required_hits),
            )
        is_viz_iter = it == 0 or (it + 1) % int(viz_interval) == 0
        if train_trigger_enabled:
            is_viz_iter = it == 0 or train_success_eval_trigger
        if train_plateau_trigger:
            is_viz_iter = True
        if not is_viz_iter:
            _wandb_upload(it, upload_video=False)
        ran_eval = False
        if is_viz_iter:
            ran_eval = True
            saved_state = save_state(env.env)
            saved_rl = {
                "_sim_step": int(env._sim_step),
                "_start_sim_step": int(env._start_sim_step),
                "_episode_ctrl_steps": int(env._episode_ctrl_steps),
                "_sim_step_buf": env._sim_step_buf.clone(),
                "_start_sim_step_buf": env._start_sim_step_buf.clone(),
                "_episode_ctrl_steps_buf": env._episode_ctrl_steps_buf.clone(),
                "rew_buf": env.rew_buf.clone(),
                "reset_buf": env.reset_buf.clone(),
                "episode_length_buf": env.episode_length_buf.clone(),
                "episode_return_buf": env.episode_return_buf.clone(),
                "episode_base_return_buf": env.episode_base_return_buf.clone(),
                "episode_extra_return_buf": env.episode_extra_return_buf.clone(),
                "obs_buf": None if env.obs_buf is None else env.obs_buf.clone(),
                "extras": dict(env.extras),
            }
            try:
                current_eval = _evaluate_policy_summary(
                    runner, env, args, cfg, print_line=False
                )
            finally:
                load_state(env.env, saved_state)
                env._sim_step = saved_rl["_sim_step"]
                env._start_sim_step = saved_rl["_start_sim_step"]
                env._episode_ctrl_steps = saved_rl["_episode_ctrl_steps"]
                env._sim_step_buf = saved_rl["_sim_step_buf"]
                env._start_sim_step_buf = saved_rl["_start_sim_step_buf"]
                env._episode_ctrl_steps_buf = saved_rl["_episode_ctrl_steps_buf"]
                env.rew_buf = saved_rl["rew_buf"]
                env.reset_buf = saved_rl["reset_buf"]
                env.episode_length_buf = saved_rl["episode_length_buf"]
                env.episode_return_buf = saved_rl["episode_return_buf"]
                env.episode_base_return_buf = saved_rl["episode_base_return_buf"]
                env.episode_extra_return_buf = saved_rl["episode_extra_return_buf"]
                env.obs_buf = saved_rl["obs_buf"]
                env.extras = saved_rl["extras"]
                qpos_restore = get_qpos(env.cfg, env.env)[0].detach().cpu().numpy()
                env.mj_data.qpos[:] = qpos_restore
                env.mj_data.time = float(env._sim_step * env.cfg.sim_dt)
                mujoco.mj_forward(env.mj_model, env.mj_data)
            steps_ratio = (
                float(current_eval["steps_actual"]) / float(current_eval["steps_max"])
                if float(current_eval["steps_max"]) > 0
                else float("nan")
            )
            reward_ratio = (
                float(current_eval["reward_actual"]) / float(current_eval["reward_max"])
                if float(current_eval["reward_max"]) > 0
                else float("nan")
            )
            current_eval = dict(current_eval)
            current_steps = float(current_eval["steps_actual"])
            if current_steps > farthest_steps_seen + 1e-9:
                farthest_steps_seen = current_steps
                first_farthest_elapsed = float(train_elapsed_at_hook)
            if abs(current_steps - farthest_steps_seen) <= 1e-9:
                current_eval["time_to_farthest"] = float(first_farthest_elapsed)
            else:
                current_eval["time_to_farthest"] = float(train_elapsed_at_hook)
            eval_history.append(
                {
                    "iter": float(it),
                    "env_steps": float((it + 1) * int(args.num_envs) * int(args.num_steps_per_env)),
                    "train_elapsed": float(train_elapsed_at_hook),
                    "time_to_farthest": float(current_eval["time_to_farthest"]),
                    "steps_actual": float(current_eval["steps_actual"]),
                    "steps_max": float(current_eval["steps_max"]),
                    "steps_ratio": float(steps_ratio),
                    "reward_actual": float(current_eval["reward_actual"]),
                    "reward_max": float(current_eval["reward_max"]),
                    "reward_ratio": float(reward_ratio),
                    "success": float(current_eval["success"]),
                }
            )
            eval_count += 1
            if first_eval_summary is None:
                first_eval_summary = dict(current_eval)
            is_best_eval = _is_better_eval(current_eval, best_eval_summary)
            if is_best_eval:
                best_eval_summary = dict(current_eval)
                best_eval_iter = int(it)
            if (
                first_success_iter is None
                and best_eval_summary is not None
                and float(best_eval_summary["success"]) > 0.0
            ):
                first_success_iter = int(best_eval_iter)
            _save_eval_checkpoint(it, is_best_eval)
            current_eval_metrics = dict(current_eval)
            current_eval_metrics["steps_ratio"] = float(steps_ratio)
            current_eval_metrics["reward_ratio"] = float(reward_ratio)
            _log_wandb_scalars(it, current_eval_metrics, prefix="eval")
            if best_eval_summary is not None:
                best_metrics = dict(best_eval_summary)
                best_metrics["iter"] = float(best_eval_iter)
                best_metrics["steps_ratio"] = (
                    float(best_eval_summary["steps_actual"]) / float(best_eval_summary["steps_max"])
                    if float(best_eval_summary["steps_max"]) > 0
                    else float("nan")
                )
                best_metrics["reward_ratio"] = (
                    float(best_eval_summary["reward_actual"]) / float(best_eval_summary["reward_max"])
                    if float(best_eval_summary["reward_max"]) > 0
                    else float("nan")
                )
                _log_wandb_scalars(it, best_metrics, prefix="eval_best")
            _save_viz_snapshot(it)
            best_trajectory_payload = _save_rollout_video(it)
            if is_best_eval and best_trajectory_payload is not None:
                _save_best_trajectory_payload(best_trajectory_payload)
                _save_best_rollout_npz(best_trajectory_payload)
            _wandb_upload(it, upload_video=True)
            if train_success_eval_trigger:
                if float(current_eval["success"]) > 0.0:
                    train_trigger_fail_streak = 0
                else:
                    train_trigger_fail_streak += 1
            reached_success = (
                best_eval_summary is not None
                and float(best_eval_summary["success"]) > 0.0
            )
            current_reached_full_eval = (
                int(args.early_stop_on_full_eval_success) != 0
                and float(current_eval["steps_max"]) > 0.0
                and float(current_eval["steps_actual"]) >= float(current_eval["steps_max"])
            )
            if current_reached_full_eval:
                loguru.logger.info(
                    "[RL][stop] iter={} reason=current rollout reached full eval horizon ({}/{})",
                    it,
                    int(current_eval["steps_actual"]),
                    int(current_eval["steps_max"]),
                )
                _finish_hook_timing(hook_t0)
                raise RLTrainingStop("current rollout reached full eval horizon")
            if train_plateau_trigger:
                min_delta = max(0.0, float(args.train_plateau_after_success_min_delta))
                loguru.logger.info(
                    "[RL][stop] iter={} reason=train reward plateau after success "
                    "window={} improvement={} min_delta={} eval_success={} eval_reward={}",
                    it,
                    int(train_plateau_window),
                    float(train_plateau_improvement),
                    float(min_delta),
                    int(current_eval["success"]),
                    float(current_eval["reward_actual"]),
                )
                _finish_hook_timing(hook_t0)
                raise RLTrainingStop("train reward plateau after success")
            plateau_eval_count = int(args.plateau_after_success_eval_count)
            if plateau_eval_count > 0 and reached_success:
                current_reward = float(current_eval["reward_actual"])
                min_delta = max(0.0, float(args.plateau_after_success_min_delta))
                if float(current_eval["success"]) > 0.0:
                    if (
                        not math.isfinite(plateau_best_reward)
                        or current_reward > plateau_best_reward + min_delta
                    ):
                        plateau_best_reward = current_reward
                        plateau_evals_since_improve = 0
                    else:
                        plateau_evals_since_improve += 1
                if plateau_evals_since_improve >= plateau_eval_count:
                    loguru.logger.info(
                        "[RL][stop] iter={} reason=reward plateau after success "
                        "evals_since_improve={} min_delta={} best_reward={} current_reward={}",
                        it,
                        int(plateau_evals_since_improve),
                        float(min_delta),
                        float(plateau_best_reward),
                        float(current_reward),
                    )
                    _finish_hook_timing(hook_t0)
                    raise RLTrainingStop("reward plateau after success")
            if (
                int(args.fail_no_progress_iterations) > 0
                and int(args.fail_no_progress_eval_count) > 0
                and first_eval_summary is not None
                and best_eval_summary is not None
                and int(eval_count) >= int(args.fail_no_progress_eval_count)
                and (it + 1) >= int(args.fail_no_progress_iterations)
                and not reached_success
                and float(best_eval_summary["steps_actual"]) <= float(first_eval_summary["steps_actual"])
                and float(best_eval_summary["reward_actual"]) <= float(first_eval_summary["reward_actual"])
            ):
                loguru.logger.info(
                    "[RL][stop] iter={} reason=no eval progress after {} iterations and {} evals "
                    "best_steps={}/{} first_steps={}/{}",
                    it,
                    int(args.fail_no_progress_iterations),
                    int(eval_count),
                    int(best_eval_summary["steps_actual"]),
                    int(best_eval_summary["steps_max"]),
                    int(first_eval_summary["steps_actual"]),
                    int(first_eval_summary["steps_max"]),
                )
                _finish_hook_timing(hook_t0)
                raise RLTrainingStop("no eval progress and rollout still failing", failed=True)
            if (
                int(args.early_stop_on_success) != 0
                and reached_success
                and first_success_iter is not None
            ):
                success_extra_iterations = max(0, int(args.success_extra_iterations))
                success_iterations_elapsed = int(it) - int(first_success_iter) + 1
                should_stop_after_success = (
                    success_extra_iterations == 0
                    or success_iterations_elapsed >= success_extra_iterations
                )
            else:
                success_extra_iterations = max(0, int(args.success_extra_iterations))
                success_iterations_elapsed = 0
                should_stop_after_success = False
            if should_stop_after_success:
                loguru.logger.info(
                    "[RL][stop] iter={} reason=best rollout reached final timestep ({}/{}) "
                    "first_success_iter={} success_iters_elapsed={} extra_iters={}",
                    best_eval_iter,
                    int(best_eval_summary["steps_actual"]),
                    int(best_eval_summary["steps_max"]),
                    int(first_success_iter),
                    int(success_iterations_elapsed),
                    int(success_extra_iterations),
                )
                _finish_hook_timing(hook_t0)
                raise RLTrainingStop("best rollout reached final timestep")
        if train_base_rew_mean == train_base_rew_mean:
            if train_rew_print_ema != train_rew_print_ema:
                train_rew_print_ema = train_base_rew_mean
            else:
                train_rew_print_ema = 0.9 * train_rew_print_ema + 0.1 * train_base_rew_mean
        if train_extra_rew_mean == train_extra_rew_mean:
            if train_extra_rew_print_ema != train_extra_rew_print_ema:
                train_extra_rew_print_ema = train_extra_rew_mean
            else:
                train_extra_rew_print_ema = 0.9 * train_extra_rew_print_ema + 0.1 * train_extra_rew_mean
        reward_text = (
            f"{train_rew_print_ema:.4f}"
            if train_rew_print_ema == train_rew_print_ema
            else "nan"
        )
        extra_reward_text = (
            f"{train_extra_rew_print_ema:.4f}"
            if train_extra_rew_print_ema == train_extra_rew_print_ema
            else "nan"
        )
        if ran_eval:
            sys.__stderr__.write(
                "[ITER] "
                f"{it} "
                f"train_reward={reward_text} "
                f"train_extra_reward={extra_reward_text} "
                f"eval_reward_actual={float(current_eval['reward_actual']):.4f} "
                f"eval_steps={int(current_eval['steps_actual'])}/{int(current_eval['steps_max'])} "
                f"eval_success_steps={int(current_eval['success_steps'])} "
                f"eval_success={int(current_eval['success'])}\n"
            )
        else:
            sys.__stderr__.write(
                f"[ITER] {it} train_reward={reward_text} train_extra_reward={extra_reward_text} "
                f"train_sim_step_max={train_sim_step_max:.0f}\n"
            )
        sys.__stderr__.flush()
        _finish_hook_timing(hook_t0)

    runner.logger.log = wrapped_log
    return history, lambda: (best_eval_summary, best_eval_iter), lambda: hook_excluded_time


def main(args: RLTrainArgs):
    cfg = _make_spider_config(args)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(cfg, cfg.data_path)
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)

    env = MJWPRslResidualEnv(
        cfg,
        ref_data,
        MJWPRLEnvCfg(
            future_ctrl_steps=args.future_ctrl_steps,
            max_episode_ctrl_steps=args.max_episode_ctrl_steps,
            lifting_rew_scale=args.lifting_rew_scale,
            contact_rew_scale=args.contact_rew_scale,
            action_smoothness_rew_scale=args.action_smoothness_rew_scale,
        ),
    )
    eval_args = replace(args, num_envs=max(1, int(args.eval_num_envs)))
    eval_cfg = _make_spider_config(eval_args)
    eval_ref_data = load_data(eval_cfg, eval_cfg.data_path)
    eval_env = MJWPRslResidualEnv(
        eval_cfg,
        eval_ref_data,
        MJWPRLEnvCfg(
            future_ctrl_steps=args.future_ctrl_steps,
            max_episode_ctrl_steps=args.max_episode_ctrl_steps,
            lifting_rew_scale=args.lifting_rew_scale,
            contact_rew_scale=args.contact_rew_scale,
            action_smoothness_rew_scale=args.action_smoothness_rew_scale,
        ),
    )
    loguru.logger.info("[RL][eval] using eval_num_envs={}", eval_env.num_envs)

    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rsl_rl.runners import OnPolicyRunner  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "rsl_rl is not installed. Install it in your environment, then rerun.\n"
            "Example (inside repo venv): `pip install rsl-rl-lib` or your preferred rsl_rl package source."
        ) from e
    resume_start_iter = 0
    resume_wandb = bool(args.resume_ckpt)
    wandb_run = None
    if args.wandb_media_only:
        try:
            os.environ["WANDB_CONSOLE"] = "off"
            os.environ["WANDB_SILENT"] = "true"
            import wandb  # type: ignore

            wandb_id_path = log_dir / "wandb_run_id.txt"
            wandb_kwargs = {
                "project": args.wandb_project,
                "name": args.wandb_run_name or args.experiment_name,
                "save_code": False,
                "config": {},
                "dir": str(log_dir),
                "settings": wandb.Settings(
                    console="off",
                    silent=True,
                    show_info=False,
                    show_warnings=False,
                ),
            }
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if resume_wandb and wandb_id_path.exists():
                run_id = wandb_id_path.read_text(encoding="utf-8").strip()
                if run_id:
                    wandb_kwargs["id"] = run_id
                    wandb_kwargs["resume"] = "allow"
            elif resume_wandb:
                run_id = _infer_wandb_run_id(log_dir)
                if run_id:
                    wandb_kwargs["id"] = run_id
                    wandb_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**wandb_kwargs)
            if getattr(wandb_run, "id", None):
                wandb_id_path.write_text(str(wandb_run.id), encoding="utf-8")
        except Exception as e:
            loguru.logger.warning("Failed to initialize wandb media-only uploader: {}", e)

    from spider.rl.rsl_compat import adapt_mlp_config
    train_cfg = adapt_mlp_config(_make_rsl_train_cfg(args))

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=cfg.device)
    if args.load_ckpt and args.resume_ckpt:
        raise ValueError("Use only one of --load-ckpt or --resume-ckpt.")
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume-ckpt not found: {ckpt_path}")
        runner.load(str(ckpt_path), map_location=cfg.device)
        resume_start_iter = int(runner.current_learning_iteration) + 1
        runner.current_learning_iteration = resume_start_iter
        loguru.logger.info(
            "[RL][ckpt] resumed checkpoint: {} start_iter={}",
            ckpt_path,
            resume_start_iter,
        )
    elif args.load_ckpt:
        ckpt_path = Path(args.load_ckpt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--load-ckpt not found: {ckpt_path}")
        runner.load(
            str(ckpt_path),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": False,
                "rnd": True,
            },
            map_location=cfg.device,
        )
        runner.current_learning_iteration = 0
        loguru.logger.info("[RL][ckpt] loaded checkpoint for fresh run: {}", ckpt_path)
    original_runner_save = runner.save

    def _save_checkpoint(path: Path):
        original_runner_save(str(path))

    # Disable automatic interval checkpoint writes; save explicitly at eval steps.
    runner.save = lambda *args, **kwargs: None  # type: ignore[method-assign]
    reward_history, get_best_eval, get_excluded_hook_time = _install_training_viz_hook(
        runner,
        eval_env,
        args,
        eval_cfg,
        log_dir,
        args.viz_interval,
        args.rollout_video_ctrl_steps,
        wandb_run=wandb_run,
        checkpoint_save_fn=_save_checkpoint,
        resume_start_iter=resume_start_iter,
    )
    # rsl_rl runner API commonly supports `learn(num_learning_iterations=...)`.
    stopped_early = False
    stopped_failed = False
    learn_t0 = time.perf_counter()
    try:
        runner.learn(num_learning_iterations=args.learning_iterations, init_at_random_ep_len=False)
    except RLTrainingStop as e:
        stopped_early = True
        stopped_failed = bool(e.failed)
        loguru.logger.info("[RL][stop] training ended early: {}", e)
        if runner.logger.writer is not None:
            try:
                runner.logger.stop_logging_writer()
            except Exception as stop_exc:
                loguru.logger.warning("[RL][stop] failed to stop logging writer cleanly: {}", stop_exc)
    learn_elapsed = max(time.perf_counter() - learn_t0 - float(get_excluded_hook_time()), 1e-8)
    best_eval_summary, best_eval_iter = get_best_eval()
    final_eval_summary = _evaluate_policy_summary(runner, eval_env, args, eval_cfg, print_line=False)
    report_eval_summary = best_eval_summary if best_eval_summary is not None else final_eval_summary
    report_eval_summary = dict(report_eval_summary)
    time_to_farthest = float(report_eval_summary.get("time_to_farthest", learn_elapsed))
    if time_to_farthest > 0.0:
        report_eval_summary["fps"] = float(report_eval_summary["steps_actual"]) / time_to_farthest
    elif float(report_eval_summary["steps_actual"]) > 0.0:
        report_eval_summary["fps"] = float("inf")
    else:
        report_eval_summary["fps"] = 0.0
    report_eval_summary["training_elapsed"] = float(time_to_farthest)
    print(
        "[RL] "
        f"task={args.task} "
        f"hand={args.active_hand} "
        f"status={report_eval_summary['status']} "
        f"fps={report_eval_summary['fps']:.2f} "
        f"reward_actual={report_eval_summary['reward_actual']:.6f} "
        f"reward_max={report_eval_summary['reward_max']:.6f} "
        f"steps_actual={report_eval_summary['steps_actual']:.2f} "
        f"steps_max={report_eval_summary['steps_max']:.2f} "
        f"execute_reward_actual={report_eval_summary.get('execute_reward_actual', report_eval_summary['reward_actual']):.6f} "
        f"execute_reward_max={report_eval_summary.get('execute_reward_max', report_eval_summary['reward_max']):.6f} "
        f"execute_steps_actual={report_eval_summary.get('execute_steps_actual', report_eval_summary['steps_actual']):.2f} "
        f"execute_steps_max={report_eval_summary.get('execute_steps_max', report_eval_summary['steps_max']):.2f} "
        f"success_steps={report_eval_summary['success_steps']:.2f} "
        f"success={int(report_eval_summary['success'])} "
        f"training_elapsed={float(report_eval_summary['training_elapsed']):.2f} "
        f"time_to_farthest={float(report_eval_summary['training_elapsed']):.2f} "
        f"reward_total={report_eval_summary['reward_total']:.6f} "
        f"reward_avg={report_eval_summary['reward_avg']:.6f} "
        f"steps={report_eval_summary['steps_actual']:.2f}/{report_eval_summary['steps_max']:.2f} "
        f"termination_step={report_eval_summary['termination_step']:.2f} "
        f"reward_last={report_eval_summary['reward_last']:.6f} "
        f"terminal_bonus={report_eval_summary['terminal_bonus']:.6f}"
    )
    eval_env.close()
    env.close()
    if stopped_failed:
        raise SystemExit(1)
if __name__ == "__main__":
    main(tyro.cli(RLTrainArgs))
