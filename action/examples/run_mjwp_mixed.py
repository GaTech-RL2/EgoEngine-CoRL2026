from __future__ import annotations

import io
import json
import math
import os
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
import warp as wp
from termcolor import cprint

from spider.config import Config, process_config
from spider.io import load_data
from spider.mode_switching import (
    FarthestProgress,
    LegacyTrainingTime,
    OptimizationClock,
    WorkLedger,
    planning_windows,
    select_mode,
)
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.rl.rsl_compat import adapt_mlp_config
from spider.simulators.mjwp import (
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    reset_inactive_side_to_ref,
    save_env_params,
    save_state,
    setup_env,
    step_env,
    sync_env_mujoco,
)

METHOD_TO_ID = {"replay": 0, "mpc": 1, "rl": 2}


@dataclass
class MixedArgs:
    dataset_dir: str = "example_datasets"
    dataset_name: str = "gigahand"
    robot_type: str = "xhand"
    embodiment_type: str = "bimanual"
    active_hand: str = "right"
    task: str = "brush_brush_bowl_20231005_186"
    data_id: int = 0
    seed: int = 0
    device: str = "cuda:0"

    sim_dt: float = 0.01
    ctrl_dt: float = 0.05
    horizon: float = 1.6
    knot_dt: float = 0.4

    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.0
    base_pos_rew_scale: float = 0.0
    base_rot_rew_scale: float = 0.0
    vel_rew_scale: float = 0.0
    contact_rew_scale: float = 0.0
    object_pos_threshold: float = 0.05
    object_rot_threshold: float = 1.0
    object_mass_scale: float = 0.1

    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    joint_noise_scale: float = 0.3
    pos_noise_scale: float = 0.0325
    rot_noise_scale: float = 0.05
    lifting_rew_scale: float = 5.0

    window_size: int = 20  # Control steps per committed chunk, not physics steps.
    lookahead_chunks: int = 2
    method_seed: int = 0  # Retained CLI compatibility; greedy selection is deterministic.
    methods: str = "replay,mpc,rl"

    mpc_num_samples: int = 8
    mpc_max_iters: int = 2
    mpc_temperature: float = 0.1
    mpc_improve_thresh: float = 0.005

    rl_total_learning_iterations: int = 1500
    rl_window_learning_iterations: int = 300
    rl_continue_from_previous: bool = True
    rl_num_envs: int = 1024
    rl_num_steps_per_env: int = 16
    rl_max_episode_ctrl_steps: int = 64
    rl_root_position_scale: float = 0.10
    rl_root_rotation_scale: float = 0.30
    rl_finger_scale: float = 0.30

    save_npz: bool = True
    output_name: str = "trajectory_mjwp_mixed.npz"
    debug_output_dir: str = ""


@dataclass
class EnvSnapshot:
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    time: float
    state_fields: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class WindowSpec:
    window_id: int
    start_step: int
    end_step: int
    method_name: str
    rl_iters_allocated: int = 0
    execute_end_step: int | None = None


@dataclass
class WindowResult:
    ctrl: np.ndarray
    reward: np.ndarray
    error: np.ndarray
    method_id: np.ndarray
    steps_executed: int
    terminated: bool
    termination_step: int
    summary: dict[str, Any] = field(default_factory=dict)
    commit_state: EnvSnapshot | None = None
    qpos_states: list[torch.Tensor] = field(default_factory=list)
    qvel_states: list[torch.Tensor] = field(default_factory=list)


@dataclass
class MixedResult:
    ctrl: np.ndarray
    reward: np.ndarray
    error: np.ndarray
    method_id: np.ndarray
    window_summaries: list[dict[str, Any]]
    success: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    initial_qpos: np.ndarray | None = None
    initial_qvel: np.ndarray | None = None


def _synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _execute_end(spec):
    return spec.end_step if spec.execute_end_step is None else spec.execute_end_step


def _parse_methods(methods: str) -> list[str]:
    items = [m.strip().lower() for m in methods.split(",") if m.strip()]
    if not items:
        raise ValueError("methods must contain at least one method")
    invalid = [m for m in items if m not in METHOD_TO_ID]
    if invalid:
        raise ValueError(f"invalid methods: {invalid}")
    return items


def _ordered_methods(enabled_methods: list[str]) -> list[str]:
    return [m for m in ("replay", "mpc", "rl") if m in enabled_methods]


def _build_config(
    args: MixedArgs,
    *,
    num_samples: int,
    max_num_iterations: int | None = None,
    temperature: float | None = None,
    improvement_threshold: float | None = None,
) -> Config:
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
        num_samples=int(num_samples),
        max_num_iterations=(
            int(max_num_iterations) if max_num_iterations is not None else Config.max_num_iterations
        ),
        temperature=float(temperature) if temperature is not None else Config.temperature,
        improvement_threshold=(
            float(improvement_threshold)
            if improvement_threshold is not None
            else Config.improvement_threshold
        ),
        pos_rew_scale=args.pos_rew_scale,
        rot_rew_scale=args.rot_rew_scale,
        joint_rew_scale=args.joint_rew_scale,
        base_pos_rew_scale=args.base_pos_rew_scale,
        base_rot_rew_scale=args.base_rot_rew_scale,
        vel_rew_scale=args.vel_rew_scale,
        contact_rew_scale=args.contact_rew_scale,
        object_pos_threshold=args.object_pos_threshold,
        object_rot_threshold=args.object_rot_threshold,
        object_mass_scale=args.object_mass_scale,
        first_ctrl_noise_scale=args.first_ctrl_noise_scale,
        last_ctrl_noise_scale=args.last_ctrl_noise_scale,
        final_noise_scale=args.final_noise_scale,
        joint_noise_scale=args.joint_noise_scale,
        pos_noise_scale=args.pos_noise_scale,
        rot_noise_scale=args.rot_noise_scale,
        num_dr=0,
        show_viewer=False,
        save_video=False,
        save_info=False,
        save_metrics=False,
        save_rerun=False,
        save_config=False,
        use_torch_compile=False,
    )
    cfg = process_config(cfg)
    env_params_list = []
    if int(cfg.num_dr) == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            cfg.xy_offset_range[0], cfg.xy_offset_range[1], int(cfg.num_dr)
        )
        pair_margin_list = np.linspace(
            cfg.pair_margin_range[0], cfg.pair_margin_range[1], int(cfg.num_dr)
        )
    env_group_size = max(int(cfg.num_dr), 1)
    for _ in range(int(cfg.max_num_iterations)):
        env_params = []
        for j in range(env_group_size):
            src_idx = min(j, len(xy_offset_list) - 1)
            env_params.append(
                {
                    "xy_offset": float(xy_offset_list[src_idx]),
                    "pair_margin": float(pair_margin_list[src_idx]),
                }
            )
        env_params_list.append(env_params)
    cfg.env_params_list = env_params_list
    if args.debug_output_dir:
        debug_output_dir = Path(args.debug_output_dir).expanduser()
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir = str(debug_output_dir)
    return cfg


def _make_rsl_train_cfg(num_steps_per_env: int, save_interval: int, seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "num_steps_per_env": int(num_steps_per_env),
        "save_interval": int(save_interval),
        "multi_gpu": None,
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo:PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.001,
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
            "distribution_cfg": {
                "class_name": "spider.rl.residual_distribution:ResidualGaussian",
                "init_std": math.exp(-2.0),
                "std_type": "log",
                "std_range": [math.exp(-5.0), math.exp(-1.0)],
                "learn_std": True,
            },
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


def _slice_ref(
    ref_data: tuple[torch.Tensor, ...], start: int, length: int
) -> tuple[torch.Tensor, ...]:
    end = start + length
    return tuple(r[start:end] for r in ref_data)


def _make_ref_step(ref_data: tuple[torch.Tensor, ...], ref_idx: int) -> tuple[torch.Tensor, ...]:
    return tuple(r[ref_idx] for r in ref_data)


def _snapshot_env_state(env) -> EnvSnapshot:
    qpos = get_qpos(None, env)[0].detach().clone()
    qvel = get_qvel(None, env)[0].detach().clone()
    ctrl = wp.to_torch(env.data_wp.ctrl)[0].detach().clone()
    time_tensor = wp.to_torch(env.data_wp.time)
    time_value = float(time_tensor.reshape(-1)[0].item())
    fields = {}
    for name in (
        "act",
        "qacc_warmstart",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "qfrc_applied",
        "xfrc_applied",
    ):
        value = getattr(env.data_wp, name, None)
        if value is not None:
            fields[name] = wp.to_torch(value)[0].detach().clone()
    return EnvSnapshot(qpos=qpos, qvel=qvel, ctrl=ctrl, time=time_value, state_fields=fields)


def _restore_env_state(
    cfg: Config, env, snapshot: EnvSnapshot, mj_data: mujoco.MjData | None = None
) -> mujoco.MjData:
    if mj_data is None:
        mj_data = mujoco.MjData(env.model_cpu)
    mj_data.qpos[:] = snapshot.qpos.detach().cpu().numpy()
    mj_data.qvel[:] = snapshot.qvel.detach().cpu().numpy()
    if env.model_cpu.nu > 0:
        mj_data.ctrl[:] = snapshot.ctrl.detach().cpu().numpy()
    mj_data.time = float(snapshot.time)
    for name, value in snapshot.state_fields.items():
        if name != "qacc_warmstart":
            getattr(mj_data, name)[:] = value.detach().cpu().numpy()
    mujoco.mj_forward(env.model_cpu, mj_data)
    if "qacc_warmstart" in snapshot.state_fields:
        mj_data.qacc_warmstart[:] = snapshot.state_fields["qacc_warmstart"].detach().cpu().numpy()
    mj_data.time = float(snapshot.time)
    sync_env_mujoco(cfg, env, mj_data)
    # The generic CPU sync does not copy equality activation on every version.
    # Copy saved runtime fields explicitly, preserving their native Warp dtypes.
    for name, value in snapshot.state_fields.items():
        target = wp.to_torch(getattr(env.data_wp, name))
        target.copy_(value.expand_as(target))
    return mj_data


def _error_from_reward_info(reward_value: float, info: dict[str, torch.Tensor]) -> float:
    if "total_dist" in info:
        return float(info["total_dist"][0].item())
    if "reward_offset_c" in info:
        return float(info["reward_offset_c"][0].item() - reward_value)
    return float("nan")


def _stack_or_empty(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _array_or_empty(values: list[float], *, dtype=np.float32) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _record_step(
    ctrl_rows: list[np.ndarray],
    reward_rows: list[float],
    error_rows: list[float],
    method_rows: list[int],
    *,
    ctrl: torch.Tensor,
    reward_value: float,
    error_value: float,
    method_name: str,
) -> None:
    ctrl_rows.append(ctrl.detach().cpu().numpy().astype(np.float32))
    reward_rows.append(float(reward_value))
    error_rows.append(float(error_value))
    method_rows.append(int(METHOD_TO_ID[method_name]))


def _finalize_window_result(
    cfg: Config,
    spec: WindowSpec,
    ctrl_rows: list[np.ndarray],
    reward_rows: list[float],
    error_rows: list[float],
    method_rows: list[int],
    *,
    terminated: bool,
    termination_step: int,
    commit_state: EnvSnapshot | None = None,
    qpos_states: list[torch.Tensor] | None = None,
    qvel_states: list[torch.Tensor] | None = None,
) -> WindowResult:
    ctrl = _stack_or_empty(ctrl_rows, cfg.nu)
    reward = _array_or_empty(reward_rows)
    error = _array_or_empty(error_rows)
    method_id = _array_or_empty(method_rows, dtype=np.int64)
    steps_executed = int(ctrl.shape[0])
    mean_error = float(np.mean(error)) if steps_executed > 0 else float("nan")
    final_error = float(error[-1]) if steps_executed > 0 else float("nan")
    total_reward = float(np.sum(reward)) if steps_executed > 0 else 0.0
    summary = {
        "window_id": int(spec.window_id),
        "start_step": int(spec.start_step),
        "end_step": int(spec.end_step),
        "method_name": spec.method_name,
        "method_id": int(METHOD_TO_ID[spec.method_name]),
        "steps_executed": steps_executed,
        "terminated": bool(terminated),
        "termination_step": int(termination_step),
        "mean_error": mean_error,
        "final_error": final_error,
        "total_reward": total_reward,
        "rl_iters_allocated": int(spec.rl_iters_allocated),
    }
    return WindowResult(
        ctrl=ctrl,
        reward=reward,
        error=error,
        method_id=method_id,
        steps_executed=steps_executed,
        terminated=bool(terminated),
        termination_step=int(termination_step),
        summary=summary,
        commit_state=commit_state,
        qpos_states=qpos_states or [],
        qvel_states=qvel_states or [],
    )


def _run_replay_window(
    cfg: Config,
    env,
    ref_data: tuple[torch.Tensor, ...],
    spec: WindowSpec,
) -> WindowResult:
    qpos_ref, qvel_ref, ctrl_ref, _, _ = ref_data
    ctrl_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    error_rows: list[float] = []
    method_rows: list[int] = []
    terminated = False
    termination_step = -1
    commit_state = None
    qpos_states, qvel_states = [], []
    for ref_idx in range(spec.start_step, spec.end_step):
        ctrl = ctrl_ref[ref_idx]
        step_env(cfg, env, ctrl.unsqueeze(0))
        reset_inactive_side_to_ref(cfg, env, qpos_ref[ref_idx], qvel_ref[ref_idx])
        ref_step = _make_ref_step(ref_data, ref_idx)
        rew, info = get_reward(cfg, env, ref_step)
        reward_value = float(rew[0].item())
        error_value = _error_from_reward_info(reward_value, info)
        term = bool(get_terminate(cfg, env, ref_step)[0].item())
        if term:
            term_rew, _ = get_terminal_reward(cfg, env, ref_step)
            reward_value += float(term_rew[0].item())
            terminated = True
            termination_step = ref_idx
        _record_step(
            ctrl_rows,
            reward_rows,
            error_rows,
            method_rows,
            ctrl=ctrl,
            reward_value=reward_value,
            error_value=error_value,
            method_name="replay",
        )
        qpos_states.append(get_qpos(cfg, env)[0].detach().clone())
        qvel_states.append(get_qvel(cfg, env)[0].detach().clone())
        if ref_idx + 1 == _execute_end(spec):
            commit_state = _snapshot_env_state(env)
        if term:
            break
    return _finalize_window_result(
        cfg,
        spec,
        ctrl_rows,
        reward_rows,
        error_rows,
        method_rows,
        terminated=terminated,
        termination_step=termination_step,
        commit_state=commit_state,
        qpos_states=qpos_states,
        qvel_states=qvel_states,
    )


def _run_mpc_window(
    cfg: Config,
    env,
    optimize,
    ref_data: tuple[torch.Tensor, ...],
    spec: WindowSpec,
) -> WindowResult:
    qpos_ref, qvel_ref, ctrl_ref, _, _ = ref_data
    ctrl_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    error_rows: list[float] = []
    method_rows: list[int] = []
    ctrls = ctrl_ref[spec.start_step : spec.start_step + cfg.horizon_steps].clone()
    current_step = int(spec.start_step)
    terminated = False
    termination_step = -1
    commit_state = None
    qpos_states, qvel_states = [], []
    while current_step < spec.end_step and not terminated:
        ref_slice = _slice_ref(ref_data, current_step, cfg.horizon_steps)
        ctrls, _infos = optimize(cfg, env, ctrls, ref_slice)
        max_exec = min(cfg.ctrl_steps, spec.end_step - current_step)
        steps_this_tick = 0
        for i in range(max_exec):
            ctrl_exec = ctrls[i].unsqueeze(0).repeat(env.num_worlds, 1)
            step_env(cfg, env, ctrl_exec)
            ref_idx = current_step + i
            reset_inactive_side_to_ref(cfg, env, qpos_ref[ref_idx], qvel_ref[ref_idx])
            ref_step = _make_ref_step(ref_data, ref_idx)
            rew, info = get_reward(cfg, env, ref_step)
            reward_value = float(rew[0].item())
            error_value = _error_from_reward_info(reward_value, info)
            term = bool(get_terminate(cfg, env, ref_step)[0].item())
            if term:
                term_rew, _ = get_terminal_reward(cfg, env, ref_step)
                reward_value += float(term_rew[0].item())
                terminated = True
                termination_step = ref_idx
            _record_step(
                ctrl_rows,
                reward_rows,
                error_rows,
                method_rows,
                ctrl=ctrls[i],
                reward_value=reward_value,
                error_value=error_value,
                method_name="mpc",
            )
            qpos_states.append(get_qpos(cfg, env)[0].detach().clone())
            qvel_states.append(get_qvel(cfg, env)[0].detach().clone())
            if ref_idx + 1 == _execute_end(spec):
                commit_state = _snapshot_env_state(env)
            steps_this_tick += 1
            if term:
                break
        current_step += steps_this_tick
        if terminated or current_step >= spec.end_step:
            break
        prev_ctrl = ctrls[steps_this_tick:]
        new_ctrl = ctrl_ref[
            current_step + prev_ctrl.shape[0] : current_step + prev_ctrl.shape[0] + steps_this_tick
        ]
        ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)
    return _finalize_window_result(
        cfg,
        spec,
        ctrl_rows,
        reward_rows,
        error_rows,
        method_rows,
        terminated=terminated,
        termination_step=termination_step,
        commit_state=commit_state,
        qpos_states=qpos_states,
        qvel_states=qvel_states,
    )


def _build_eval_obs(cfg, env, train_env, ref_idx):
    from tensordict import TensorDict

    obs = train_env.observation_for(cfg, env, ref_idx)
    return TensorDict({"policy": obs}, batch_size=[1], device=cfg.device)


def _evaluate_rl_window(main_cfg, main_env, ref_data, spec, train_env, policy, snapshot):
    _restore_env_state(main_cfg, main_env, snapshot)
    qpos_ref, qvel_ref, ctrl_ref, _, _ = ref_data
    ctrl_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    error_rows: list[float] = []
    method_rows: list[int] = []
    current_step = int(spec.start_step)
    terminated = False
    termination_step = -1
    commit_state = None
    qpos_states, qvel_states = [], []
    with torch.no_grad():
        while current_step < spec.end_step and not terminated:
            obs_td = _build_eval_obs(main_cfg, main_env, train_env, current_step)
            actions = policy(obs_td)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            actions = torch.clamp(
                actions.to(main_cfg.device),
                -train_env.env_cfg.action_clip,
                train_env.env_cfg.action_clip,
            )
            ctrl, _, _ = train_env.residual_actions_to_ctrl(actions, current_step)
            max_exec = min(main_cfg.ctrl_steps, spec.end_step - current_step)
            steps_this_tick = 0
            for sub in range(max_exec):
                step_env(main_cfg, main_env, ctrl)
                ref_idx = current_step + sub + 1
                reset_inactive_side_to_ref(main_cfg, main_env, qpos_ref[ref_idx], qvel_ref[ref_idx])
                ref_step = _make_ref_step(ref_data, ref_idx)
                rew, info = get_reward(main_cfg, main_env, ref_step)
                reward_value = float(rew[0].item())
                error_value = _error_from_reward_info(reward_value, info)
                term = bool(get_terminate(main_cfg, main_env, ref_step)[0].item())
                if term:
                    term_rew, _ = get_terminal_reward(main_cfg, main_env, ref_step)
                    reward_value += float(term_rew[0].item())
                    terminated = True
                    termination_step = ref_idx - 1
                _record_step(
                    ctrl_rows,
                    reward_rows,
                    error_rows,
                    method_rows,
                    ctrl=ctrl[0],
                    reward_value=reward_value,
                    error_value=error_value,
                    method_name="rl",
                )
                qpos_states.append(get_qpos(main_cfg, main_env)[0].detach().clone())
                qvel_states.append(get_qvel(main_cfg, main_env)[0].detach().clone())
                if ref_idx == _execute_end(spec):
                    commit_state = _snapshot_env_state(main_env)
                steps_this_tick += 1
                if term:
                    break
            current_step += steps_this_tick
    return _finalize_window_result(
        main_cfg,
        spec,
        ctrl_rows,
        reward_rows,
        error_rows,
        method_rows,
        terminated=terminated,
        termination_step=termination_step,
        commit_state=commit_state,
        qpos_states=qpos_states,
        qvel_states=qvel_states,
    )


def _run_rl_window(
    args: MixedArgs,
    main_cfg: Config,
    main_env,
    rl_cfg: Config,
    ref_data: tuple[torch.Tensor, ...],
    spec: WindowSpec,
    previous_checkpoint: str | None = None,
) -> WindowResult:
    try:
        from rsl_rl.runners import OnPolicyRunner  # type: ignore

        from spider.rl.goal_observation import restore_masked_state
        from spider.rl.mjwp_rsl_env import MJWPRLEnvCfg, MJWPRslResidualEnv
    except Exception as exc:
        raise RuntimeError(
            "rsl_rl is required for the RL mode. Install it in the target environment."
        ) from exc

    class WindowResidualEnv(MJWPRslResidualEnv):
        def __init__(
            self,
            config: Config,
            ref_data_local: tuple[torch.Tensor, ...],
            env_cfg: MJWPRLEnvCfg,
            *,
            window_snapshot: EnvSnapshot,
            window_start_step: int,
            window_end_step: int,
        ) -> None:
            self._window_snapshot = window_snapshot
            self._window_start_step = int(window_start_step)
            self._window_end_step = int(window_end_step)
            super().__init__(config, ref_data_local, env_cfg)

        def _reset_all(self) -> torch.Tensor:
            _restore_env_state(self.cfg, self.env, self._window_snapshot, self.mj_data)
            self._sim_step = int(self._window_start_step)
            self._start_sim_step = int(self._window_start_step)
            self._episode_ctrl_steps = 0
            self._sim_step_buf.fill_(int(self._window_start_step))
            self._start_sim_step_buf.fill_(int(self._window_start_step))
            self._episode_ctrl_steps_buf.zero_()
            self.rew_buf.zero_()
            self.reset_buf.zero_()
            self.episode_length_buf.zero_()
            self.episode_return_buf.zero_()
            self.episode_base_return_buf.zero_()
            self.episode_extra_return_buf.zero_()
            self.extras = {}
            return self._get_obs()

        def _reset_done(self, done_mask):
            restore_masked_state(self.env.data_wp, self._window_snapshot, done_mask, wp.to_torch)
            self._sim_step_buf[done_mask] = self._window_start_step
            self._start_sim_step_buf[done_mask] = self._window_start_step
            for buffer in (self._episode_ctrl_steps_buf, self.episode_length_buf,
                           self.episode_return_buf, self.episode_base_return_buf,
                           self.episode_extra_return_buf, self._prev_actions,
                           self._has_prev_action):
                buffer[done_mask] = 0
            self._sync_scalar_clocks_from_env0()

        def step(self, actions: torch.Tensor):
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
            actions = actions.to(self.device, dtype=torch.float32)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            actions = torch.clamp(actions, -self.env_cfg.action_clip, self.env_cfg.action_clip)
            if actions.shape[1] != self.num_actions:
                raise ValueError(f"Expected action dim {self.num_actions}, got {actions.shape[1]}")

            total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            base_total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            extra_total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            term_reason_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

            base_ctrl_idx = torch.clamp(self._sim_step_buf, 0, self.max_ref_idx)
            ctrl, _, _ = self.residual_actions_to_ctrl(actions, base_ctrl_idx)

            for _sub in range(self.cfg.ctrl_steps):
                step_env(self.cfg, self.env, ctrl)
                alive = ~done
                self._sim_step_buf += alive.to(torch.long)
                clamped_idx = torch.clamp(self._sim_step_buf, 0, self.max_ref_idx)
                reset_inactive_side_to_ref(
                    self.cfg,
                    self.env,
                    self.qpos_ref.index_select(0, clamped_idx),
                    self.qvel_ref.index_select(0, clamped_idx),
                )
                ref_step, _ = self._ref_at_indices(self._sim_step_buf)
                base_rew, _ = get_reward(self.cfg, self.env, ref_step)
                extra_rew = self._active_object_z_tracking_reward(ref_step[0])
                rew = base_rew + extra_rew
                if not torch.isfinite(rew).all():
                    bad = ~torch.isfinite(rew)
                    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
                    base_rew = torch.nan_to_num(base_rew, nan=0.0, posinf=0.0, neginf=0.0)
                    extra_rew = torch.nan_to_num(extra_rew, nan=0.0, posinf=0.0, neginf=0.0)
                    done |= bad
                    term_reason_id_buf[bad] = 1.0
                total_rew += torch.where(alive, rew, 0.0)
                base_total_rew += torch.where(alive, base_rew, 0.0)
                extra_total_rew += torch.where(alive, extra_rew, 0.0)
                term = get_terminate(self.cfg, self.env, ref_step) & alive
                if term.any():
                    done |= term
                    term_reason_id_buf[term] = 1.0
                completed = (self._sim_step_buf >= self._window_end_step) & ~done
                done |= completed
                term_reason_id_buf[completed] = 3.0
                if bool(done.all()):
                    break

            self._episode_ctrl_steps_buf += 1
            self.episode_length_buf += 1
            timeout = self._episode_ctrl_steps_buf >= self.max_episode_length
            newly_timeout = timeout & ~done
            done |= timeout
            term_reason_id_buf[newly_timeout] = 2.0

            # A terminated world has zero reward for the rest of this control
            # tick. Its reward must not depend on when other worlds finish.
            denom = max(1, self.cfg.ctrl_steps)
            self.rew_buf = total_rew / denom
            base_rew_buf = base_total_rew / denom
            extra_rew_buf = extra_total_rew / denom
            self.reset_buf = done.clone()
            self.episode_return_buf += self.rew_buf
            self.episode_base_return_buf += base_rew_buf
            self.episode_extra_return_buf += extra_rew_buf
            rew_out = self.rew_buf.clone()
            done_out = done.clone()

            if done.any():
                episode_info = {
                    "r": float(self.episode_return_buf[done].mean().item()),
                    "base_r": float(self.episode_base_return_buf[done].mean().item()),
                    "extra_r": float(self.episode_extra_return_buf[done].mean().item()),
                    "l": float(self._episode_ctrl_steps_buf[done].to(torch.float32).mean().item()),
                    "sim_step": float(self._sim_step_buf[done].to(torch.float32).mean().item()),
                    "term_reason_id": float(term_reason_id_buf[done].mean().item()),
                }
                self._reset_done(done)
                next_obs = self._get_obs()
                self.extras = {
                    "time_outs": torch.zeros_like(done),
                    "episode": episode_info,
                }
            else:
                next_obs = self._get_obs()
                self._sim_step = int(self._sim_step_buf[0].item())
                self._start_sim_step = int(self._start_sim_step_buf[0].item())
                self._episode_ctrl_steps = int(self._episode_ctrl_steps_buf[0].item())
                self.extras = {"time_outs": torch.zeros_like(done)}
            if not torch.isfinite(next_obs).all():
                next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
            self.obs_buf = next_obs
            return self._obs_tensordict(), rew_out, done_out, self.extras

    if spec.rl_iters_allocated <= 0:
        raise ValueError(f"RL window {spec.window_id} received non-positive iteration budget")

    snapshot = _snapshot_env_state(main_env)
    local_max_episode_ctrl_steps = max(
        1,
        min(
            int(args.rl_max_episode_ctrl_steps),
            int(math.ceil((spec.end_step - spec.start_step) / max(1, rl_cfg.ctrl_steps))),
        ),
    )
    train_env = WindowResidualEnv(
        rl_cfg,
        ref_data,
        MJWPRLEnvCfg(
            future_ctrl_steps=0,
            max_episode_ctrl_steps=local_max_episode_ctrl_steps,
            lifting_rew_scale=args.lifting_rew_scale,
            root_position_scale=args.rl_root_position_scale,
            root_rotation_scale=args.rl_root_rotation_scale,
            finger_scale=args.rl_finger_scale,
        ),
        window_snapshot=snapshot,
        window_start_step=spec.start_step,
        window_end_step=spec.end_step,
    )
    run_dir = Path(rl_cfg.output_dir) / "_mixed_rl" / f"window_{spec.window_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = _make_rsl_train_cfg(
        num_steps_per_env=int(args.rl_num_steps_per_env),
        save_interval=max(1, spec.rl_iters_allocated),
        seed=int(args.seed + spec.window_id),
    )
    train_cfg = adapt_mlp_config(train_cfg)
    runner = OnPolicyRunner(train_env, train_cfg, log_dir=str(run_dir), device=rl_cfg.device)
    if previous_checkpoint is not None:
        # Match run_aria_rl_chunk.sh's --load-ckpt: continue the accepted
        # models/normalizers and optimizer, but give this window a fresh counter.
        runner.load(previous_checkpoint, load_cfg={
            "actor": True, "critic": True, "optimizer": True,
            "iteration": False, "rnd": True,
        }, map_location=rl_cfg.device)
        runner.current_learning_iteration = 0
    native_save = runner.save
    runner.save = lambda *a, **k: None  # type: ignore[method-assign]
    policy = runner.alg.get_policy()
    ledger = main_cfg.work_ledger
    main_cfg.work_mode = "rl"
    clock = OptimizationClock(lambda: _synchronize(main_cfg.device))
    progress = FarthestProgress()
    legacy_time = LegacyTrainingTime()
    best = None
    best_policy_state = None
    learning_trace = []
    used = 0
    original_log = runner.logger.log

    class WindowSolved(Exception):
        pass

    def evaluate():
        with clock.exclude():
            was_training = policy.training
            policy.eval()
            try:
                with ledger.evaluating(), torch.no_grad():
                    return _evaluate_rl_window(
                        main_cfg, main_env, ref_data, spec, train_env, policy, snapshot
                    )
            finally:
                policy.train(was_training)
                _restore_env_state(main_cfg, main_env, snapshot)

    def wrapped_log(*log_args, **kwargs):
        nonlocal best, used, best_policy_state
        # Preserve ordinary logger work; exclude the evaluation/export hook.
        with redirect_stdout(io.StringIO()):
            original_log(*log_args, **kwargs)
        fallback = (
            log_args[0].get("it", 0)
            if log_args and isinstance(log_args[0], dict)
            else log_args[0]
            if log_args
            else runner.current_learning_iteration
        )
        used = max(used, int(kwargs.get("it", fallback)) + 1)
        elapsed = legacy_time.at_hook(clock.elapsed())
        with clock.exclude():
            trial = evaluate()
            valid_steps = _valid_window_steps(trial)
            progress.observe(valid_steps, elapsed)
            learning_trace.append({
                "iteration": used, "valid_steps": valid_steps,
                "total_reward": trial.summary["total_reward"],
                "feasible": _window_succeeded(trial, spec),
                "legacy_training_elapsed": elapsed,
            })
            if used == 1 or used % 20 == 0:
                cprint(f"[RL] chunk={spec.window_id} iteration={used} "
                       f"valid_steps={valid_steps}/{spec.end_step - spec.start_step} "
                       f"first_farthest_time={progress.elapsed:.3f}s", "cyan")
                (run_dir / "learning_trace.json").write_text(json.dumps(learning_trace, indent=2))
            if (
                best is None
                or valid_steps > _valid_window_steps(best)
                or (
                    valid_steps == _valid_window_steps(best)
                    and trial.summary["total_reward"] > best.summary["total_reward"]
                )
            ):
                best = trial
                best_policy_state = {key: value.detach().cpu().clone()
                                     for key, value in policy.state_dict().items()}
            if _window_succeeded(trial, spec):
                best = trial
                best_policy_state = {key: value.detach().cpu().clone()
                                     for key, value in policy.state_dict().items()}
                raise WindowSolved()

    runner.logger.log = wrapped_log
    stopped = False
    try:
        runner.learn(
            num_learning_iterations=int(spec.rl_iters_allocated), init_at_random_ep_len=False
        )
    except WindowSolved:
        stopped = True
    finally:
        runner.logger.log = original_log
        if stopped and getattr(runner.logger, "writer", None) is not None:
            with clock.exclude():
                runner.logger.stop_logging_writer()
    if not stopped:
        used = int(spec.rl_iters_allocated)
    elapsed = clock.elapsed()
    if best is None:
        best = evaluate()
        progress.observe(_valid_window_steps(best), legacy_time.elapsed(elapsed))
    with clock.exclude():
        (run_dir / "learning_trace.json").write_text(json.dumps(learning_trace, indent=2))
        if best_policy_state is not None:
            torch.save({"actor_state_dict": best_policy_state, "train_cfg": train_cfg,
                        "observation": "joint_tip_object_goals.v1", "num_obs": train_env.num_obs,
                        "num_actions": train_env.num_actions, "window_start": spec.start_step,
                        "window_end": spec.end_step, "feasible": _window_succeeded(best, spec),
                        "embodiment_type": args.embodiment_type, "active_hand": args.active_hand,
                        "residual_scales": {"root_position": args.rl_root_position_scale,
                                            "root_rotation": args.rl_root_rotation_scale,
                                            "finger": args.rl_finger_scale}},
                       run_dir / "policy_best.pt")
        continuation_checkpoint = None
        if stopped:
            # The first fully feasible evaluation stops learn immediately, so
            # the native runner state here is exactly the accepted checkpoint.
            continuation_checkpoint = str(run_dir / "policy_continue.pt")
            native_save(continuation_checkpoint)
    best.summary.update(
        optimization_elapsed=elapsed,
        time_to_farthest=progress.elapsed,
        benchmark_elapsed=(
            progress.elapsed if _window_succeeded(best, spec) else legacy_time.elapsed(elapsed)
        ),
        warmup_elapsed=legacy_time.warmup or 0.0,
        rl_iterations_used=used,
        hook_excluded_time=clock.excluded,
        policy_checkpoint=str(run_dir / "policy_best.pt"),
        learning_trace=learning_trace,
        warm_start_checkpoint=previous_checkpoint,
        warm_started=previous_checkpoint is not None,
        continuation_checkpoint=continuation_checkpoint,
    )
    train_env.close()
    return best


def _json_ready(value):
    """Export portable JSON, including empty or nonfinite failed-attempt diagnostics."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_result(result: MixedResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = result.window_summaries
    np.savez(
        path,
        success=np.asarray(result.success),
        statistics_json=np.asarray(json.dumps(_json_ready(result.statistics), allow_nan=False)),
        attempts_json=np.asarray(json.dumps(_json_ready(result.attempts), allow_nan=False)),
        window_plan_end=np.asarray([s["plan_end_step"] for s in summaries], dtype=np.int64),
        qpos=result.qpos if result.qpos is not None else np.empty((0, 0)),
        qvel=result.qvel if result.qvel is not None else np.empty((0, 0)),
        initial_qpos=result.initial_qpos if result.initial_qpos is not None else np.empty(0),
        initial_qvel=result.initial_qvel if result.initial_qvel is not None else np.empty(0),
        ctrl=result.ctrl,
        reward=result.reward,
        error=result.error,
        method_id=result.method_id,
        window_start=np.asarray([s["start_step"] for s in summaries], dtype=np.int64),
        window_end=np.asarray([s["end_step"] for s in summaries], dtype=np.int64),
        window_method_id=np.asarray([s["method_id"] for s in summaries], dtype=np.int64),
        window_steps_executed=np.asarray([s["steps_executed"] for s in summaries], dtype=np.int64),
        window_terminated=np.asarray([s["terminated"] for s in summaries], dtype=np.bool_),
        window_mean_error=np.asarray([s["mean_error"] for s in summaries], dtype=np.float32),
        window_final_error=np.asarray([s["final_error"] for s in summaries], dtype=np.float32),
        window_total_reward=np.asarray([s["total_reward"] for s in summaries], dtype=np.float32),
        window_rl_iters_allocated=np.asarray(
            [s["rl_iters_allocated"] for s in summaries], dtype=np.int64
        ),
    )


def _valid_window_steps(result: WindowResult) -> int:
    """The terminal failing/nonfinite step is computation, not valid progress."""
    valid = int(result.steps_executed)
    if valid:
        finite = (
            np.isfinite(result.ctrl).all(axis=1)
            & np.isfinite(result.reward)
            & np.isfinite(result.error)
        )
        if result.qpos_states:
            state_finite = torch.isfinite(torch.stack(result.qpos_states)).all(dim=1)
            state_finite &= torch.isfinite(torch.stack(result.qvel_states)).all(dim=1)
            finite &= state_finite.detach().cpu().numpy()
        bad = np.flatnonzero(~finite)
        if len(bad):
            valid = min(valid, int(bad[0]))
    if result.terminated:
        valid = min(valid, max(0, result.steps_executed - 1))
    return valid


def _window_succeeded(result: WindowResult, spec: WindowSpec) -> bool:
    return bool(
        not result.terminated
        and _valid_window_steps(result) >= int(spec.end_step - spec.start_step)
        and int(result.steps_executed) >= int(spec.end_step - spec.start_step)
        and np.isfinite(result.ctrl).all()
        and np.isfinite(result.reward).all()
        and np.isfinite(result.error).all()
    )


def main(args: MixedArgs) -> None:
    import time

    pipeline_start = time.perf_counter()
    methods = _ordered_methods(_parse_methods(args.methods))
    if args.rl_total_learning_iterations <= 0 and "rl" in methods:
        raise ValueError("RL iteration budget must be positive")
    if args.rl_window_learning_iterations <= 0 and "rl" in methods:
        raise ValueError("Per-window RL iteration budget must be positive")
    main_cfg = _build_config(args, num_samples=1)
    ref_data = load_data(main_cfg, main_cfg.data_path)
    total_steps = int(ref_data[0].shape[0] - main_cfg.horizon_steps - main_cfg.ctrl_steps)
    windows = planning_windows(
        total_steps, main_cfg.ctrl_steps, args.window_size, args.lookahead_chunks
    )
    if not windows:
        raise RuntimeError(f"No valid mixed-tracking steps for task={args.task}")
    ledger = WorkLedger()
    main_env = setup_env(main_cfg, ref_data)
    initial_qpos = get_qpos(main_cfg, main_env)[0].detach().cpu().numpy().copy()
    initial_qvel = get_qvel(main_cfg, main_env)[0].detach().cpu().numpy().copy()
    main_cfg.work_ledger, main_cfg.work_mode = ledger, "replay"
    mpc_cfg = mpc_env = mpc_optimize = None
    if "mpc" in methods:
        mpc_cfg = _build_config(
            args,
            num_samples=args.mpc_num_samples,
            max_num_iterations=args.mpc_max_iters,
            temperature=args.mpc_temperature,
            improvement_threshold=args.mpc_improve_thresh,
        )
        mpc_env = setup_env(mpc_cfg, ref_data)
        mpc_cfg.work_ledger, mpc_cfg.work_mode = ledger, "mpc"
        rollout = make_rollout_fn(
            step_env,
            save_state,
            load_state,
            get_reward,
            get_terminal_reward,
            get_terminate,
            get_trace,
            save_env_params,
            load_env_params,
            copy_sample_state,
        )
        mpc_optimize = make_optimize_fn(make_optimize_once_fn(rollout))
    rl_cfg = None
    if "rl" in methods:
        rl_cfg = _build_config(args, num_samples=args.rl_num_envs)
        rl_cfg.work_ledger, rl_cfg.work_mode = ledger, "rl"
    chunks, summaries, attempts = [], [], []
    qpos_states, qvel_states = [], []
    remaining = int(args.rl_total_learning_iterations)
    previous_rl_checkpoint = None
    optimization_elapsed = 0.0
    benchmark_elapsed = 0.0
    progress = FarthestProgress()
    for index, window in enumerate(windows):
        boundary = _snapshot_env_state(main_env)
        spec = WindowSpec(
            window.index,
            window.start,
            window.plan_end,
            methods[0],
            execute_end_step=window.execute_end,
        )

        def reset():
            _restore_env_state(main_cfg, main_env, boundary)

        def solve(method):
            nonlocal remaining, optimization_elapsed, benchmark_elapsed
            selected = replace(spec, method_name=method)
            before = ledger.snapshot()
            clock = OptimizationClock(lambda: _synchronize(args.device))
            if method == "replay":
                main_cfg.work_mode = method
                trial = _run_replay_window(main_cfg, main_env, ref_data, selected)
            elif method == "mpc":
                _restore_env_state(mpc_cfg, mpc_env, boundary)
                trial = _run_mpc_window(mpc_cfg, mpc_env, mpc_optimize, ref_data, selected)
            elif remaining > 0:
                selected.rl_iters_allocated = min(remaining, args.rl_window_learning_iterations)
                trial = _run_rl_window(
                    args, main_cfg, main_env, rl_cfg, ref_data, selected,
                    previous_checkpoint=(previous_rl_checkpoint if args.rl_continue_from_previous else None),
                )
                remaining -= trial.summary["rl_iterations_used"]
            else:
                trial = _finalize_window_result(
                    main_cfg,
                    selected,
                    [],
                    [],
                    [],
                    [],
                    terminated=True,
                    termination_step=window.start,
                )
                trial.summary.update(rl_iterations_used=0, failure_reason="rl_budget_exhausted")
            elapsed = (
                trial.summary["optimization_elapsed"]
                if "optimization_elapsed" in trial.summary
                else clock.elapsed()
            )
            optimization_elapsed += elapsed
            reported = trial.summary.get("benchmark_elapsed", elapsed)
            benchmark_elapsed += reported
            after = ledger.snapshot()
            trial.summary.update(
                task=args.task,
                execute_end_step=window.execute_end,
                plan_end_step=window.plan_end,
                optimization_elapsed=elapsed,
                benchmark_elapsed=reported,
                cumulative_optimization_elapsed=optimization_elapsed,
                cumulative_benchmark_elapsed=benchmark_elapsed,
                generation_sim_steps=after["generation_sim_steps"] - before["generation_sim_steps"],
                evaluation_sim_steps=after["evaluation_sim_steps"] - before["evaluation_sim_steps"],
                feasible=_window_succeeded(trial, selected),
            )
            attempts.append(dict(trial.summary))
            return trial

        def feasible(trial):
            return trial.summary["feasible"] and trial.commit_state is not None

        def commit(trial):
            _restore_env_state(main_cfg, main_env, trial.commit_state)

        trial, _ = select_mode(methods, reset, solve, feasible, commit)
        if trial is None:
            break
        if trial.summary["method_name"] == "rl":
            previous_rl_checkpoint = trial.summary["continuation_checkpoint"]
        width = window.execute_end - window.start
        chunks.append(
            (trial.ctrl[:width], trial.reward[:width], trial.error[:width], trial.method_id[:width])
        )
        qpos_states.extend(trial.qpos_states[:width])
        qvel_states.extend(trial.qvel_states[:width])
        summary = dict(trial.summary)
        summary.update(
            start_step=window.start,
            end_step=window.execute_end,
            plan_steps_executed=trial.steps_executed,
            steps_executed=width,
            terminated=False,
            termination_step=-1,
            mean_error=float(np.mean(trial.error[:width])),
            final_error=float(trial.error[width - 1]),
            total_reward=float(np.sum(trial.reward[:width])),
        )
        summaries.append(summary)
        progress.observe(window.execute_end, benchmark_elapsed)
        cprint(
            f"[SWITCH] chunk={index} mode={summary['method_name']} "
            f"execute={window.start}:{window.execute_end} plan_end={window.plan_end}",
            "cyan",
        )
    ctrl = (
        np.concatenate([c[0] for c in chunks]) if chunks else np.empty((0, main_cfg.nu), np.float32)
    )

    def cat(i, dtype):
        return np.concatenate([c[i] for c in chunks]) if chunks else np.empty(0, dtype)

    success = len(ctrl) == total_steps
    reference_ratio = main_cfg.ref_dt / main_cfg.sim_dt
    steps_actual = len(ctrl) / reference_ratio
    steps_max = total_steps / reference_ratio
    work = ledger.snapshot()
    statistics = dict(
        work,
        success=success,
        steps_actual=steps_actual,
        steps_max=steps_max,
        committed_sim_steps=len(ctrl),
        control_steps_per_chunk=args.window_size,
        sim_steps_per_control=main_cfg.ctrl_steps,
        sim_steps_per_reference=reference_ratio,
        lookahead_chunks=args.lookahead_chunks,
        optimization_elapsed=optimization_elapsed,
        training_elapsed=progress.elapsed,
        time_to_farthest=progress.elapsed,
        fps=(
            steps_actual / progress.elapsed
            if progress.elapsed > 0
            else math.inf
            if steps_actual > 0
            else 0.0
        ),
        pipeline_wall_elapsed=time.perf_counter() - pipeline_start,
        cost=work["generation_sim_steps"] / steps_actual if success else None,
        cost_including_evaluation=(work["generation_sim_steps"] + work["evaluation_sim_steps"])
        / steps_actual
        if success
        else None,
        rl_iterations_used=int(args.rl_total_learning_iterations) - remaining,
        rl_window_learning_iterations=args.rl_window_learning_iterations,
        rl_continue_from_previous=args.rl_continue_from_previous,
        rl_warm_started_windows=sum(bool(s.get("warm_started")) for s in summaries),
    )
    result = MixedResult(
        ctrl,
        cat(1, np.float32),
        cat(2, np.float32),
        cat(3, np.int64),
        summaries,
        success,
        attempts,
        statistics,
    )
    # Transfer state history after optimization timing ends; serialization is an export hook.
    if qpos_states:
        result.qpos = torch.stack(qpos_states).detach().cpu().numpy()
        result.qvel = torch.stack(qvel_states).detach().cpu().numpy()
    result.initial_qpos = initial_qpos
    result.initial_qvel = initial_qvel
    if args.save_npz:
        name = args.output_name if args.output_name.endswith(".npz") else args.output_name + ".npz"
        path = Path(main_cfg.output_dir) / name
        _save_result(result, path)
        path.with_suffix(".json").write_text(
            json.dumps(
                _json_ready({"statistics": statistics, "windows": summaries, "attempts": attempts}),
                indent=2,
                allow_nan=False,
            )
        )
        cprint(f"save results to: {path}", "cyan")
    counts = Counter(s["method_name"] for s in summaries)
    print(
        f"[MIXED] task={args.task} steps_actual={steps_actual:.2f} steps_max={steps_max:.2f} "
        f"success={int(success)} fps={statistics['fps']:.6f} "
        f"training_elapsed={progress.elapsed:.6f} time_to_farthest={progress.elapsed:.6f} "
        f"optimization_elapsed={optimization_elapsed:.6f} "
        f"generation_sim_steps={work['generation_sim_steps']} evaluation_sim_steps={work['evaluation_sim_steps']} "
        f"replay_windows={counts['replay']} mpc_windows={counts['mpc']} rl_windows={counts['rl']} "
        f"rl_budget_used={statistics['rl_iterations_used']}/{args.rl_total_learning_iterations}"
    )


if __name__ == "__main__":
    main(tyro.cli(MixedArgs))
