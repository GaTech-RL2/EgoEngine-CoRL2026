from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from spider.config import Config
from spider.rl.goal_observation import GoalObservation
from spider.rl.state_layout import bimanual_object_slice
from spider.simulators.mjwp import (
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    load_env_params,
    reset_inactive_side_to_ref,
    setup_env,
    setup_mj_model,
    step_env,
    sync_env_mujoco,
)


def _quat_wxyz_to_rotmat_torch(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) to rotation matrix. quat shape (..., 4)."""
    w, x, y, z = quat.unbind(dim=-1)
    two = torch.tensor(2.0, device=quat.device, dtype=quat.dtype)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - two * (yy + zz)
    r01 = two * (xy - wz)
    r02 = two * (xz + wy)
    r10 = two * (xy + wz)
    r11 = 1 - two * (xx + zz)
    r12 = two * (yz - wx)
    r20 = two * (xz - wy)
    r21 = two * (yz + wx)
    r22 = 1 - two * (xx + yy)
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def _quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    rot = _quat_wxyz_to_rotmat_torch(quat_wxyz)
    return rot[..., :, :2].reshape(*rot.shape[:-2], 6)


def _build_action_scale(config: Config, env_cfg=None) -> torch.Tensor:
    """Residual correction capacity is independent of MPC sampling noise."""
    env_cfg = env_cfg or MJWPRLEnvCfg()
    scale = torch.ones(config.nu, device=config.device, dtype=torch.float32)
    if config.embodiment_type in ["bimanual", "right", "left"]:
        scale[:3] = float(env_cfg.root_position_scale)
        scale[3:6] = float(env_cfg.root_rotation_scale)
        if config.embodiment_type == "bimanual":
            half = config.nu // 2
            scale[6:half] = float(env_cfg.finger_scale)
            scale[half : half + 3] = float(env_cfg.root_position_scale)
            scale[half + 3 : half + 6] = float(env_cfg.root_rotation_scale)
            scale[half + 6 :] = float(env_cfg.finger_scale)
            if config.active_hand == "right":
                scale[half:] = 0.0
            elif config.active_hand == "left":
                scale[:half] = 0.0
        else:
            scale[6:] = float(env_cfg.finger_scale)
    else:
        scale[:] = float(env_cfg.finger_scale)
    return scale


@dataclass
class MJWPRLEnvCfg:
    root_position_scale: float = 0.10
    root_rotation_scale: float = 0.30
    finger_scale: float = 0.30
    future_ctrl_steps: int = 0
    max_episode_ctrl_steps: int = 64
    action_clip: float = 1.0
    lifting_rew_scale: float = 0.0
    contact_rew_scale: float = 0.0
    action_smoothness_rew_scale: float = 0.0


class MJWPRslResidualEnv:
    """Single-task residual RL env on top of MJWP.

    Parallel residual RL env on top of MJWP.

    Each world keeps its own reference phase / episode clock, which is important
    for PPO sample diversity and to avoid "one world terminates => all reset".
    """

    def __init__(
        self,
        config: Config,
        ref_data: tuple[torch.Tensor, ...],
        env_cfg: MJWPRLEnvCfg,
    ) -> None:
        self.cfg = config
        self.ref_data = ref_data
        self.env_cfg = env_cfg
        self.device = config.device
        self.num_envs = int(config.num_samples)
        self.num_privileged_obs = 0
        self.max_episode_length = int(env_cfg.max_episode_ctrl_steps)

        self.env = setup_env(config, ref_data)
        self.mj_model = setup_mj_model(config)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.qpos_ref, self.qvel_ref, self.ctrl_ref, self.contact_ref, self.contact_pos_ref = (
            ref_data
        )
        self.max_ref_idx = int(self.qpos_ref.shape[0] - 1)
        self.nq_obj = int(config.nq_obj)
        self.nv_obj = 12 if config.embodiment_type == "bimanual" else 6
        self._single_hand_mode = bool(
            config.embodiment_type == "bimanual"
            and config.active_hand in ["right", "left"]
        )
        self._half_nu = int(config.nu // 2)
        full_action_scale = _build_action_scale(config, env_cfg)
        if self._single_hand_mode:
            if config.active_hand == "right":
                self._ctrl_slice = slice(0, self._half_nu)
            else:
                self._ctrl_slice = slice(self._half_nu, int(config.nu))
            self._obj_qpos_slice = bimanual_object_slice(self.mj_model, config.active_hand)
            self.action_scale = full_action_scale[self._ctrl_slice].clone()
            self.num_actions = int(self.action_scale.numel())
        else:
            self._ctrl_slice = slice(0, int(config.nu))
            self._obj_qpos_slice = slice(0, self.nq_obj)
            self.action_scale = full_action_scale
            self.num_actions = int(config.nu)

        self.goal_observation = GoalObservation(
            self.mj_model, self.qpos_ref, config.embodiment_type, config.active_hand
        )
        limits = self.mj_model.actuator_ctrlrange.copy()
        limited = self.mj_model.actuator_ctrllimited.astype(bool)
        limits[~limited] = [-np.inf, np.inf]
        self._ctrl_lower = torch.as_tensor(limits[:, 0], device=self.device, dtype=torch.float32)
        self._ctrl_upper = torch.as_tensor(limits[:, 1], device=self.device, dtype=torch.float32)

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.episode_return_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.episode_base_return_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.episode_extra_return_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.obs_buf: torch.Tensor | None = None
        self.extras: dict = {}

        self._sim_step = 0
        self._start_sim_step = 0
        self._episode_ctrl_steps = 0
        self._sim_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._start_sim_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._episode_ctrl_steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._dr_xy_offset = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._dr_object_mass_scale = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._dr_pair_margin = 0.0
        self._prev_actions = torch.zeros(
            self.num_envs, self.num_actions, device=self.device, dtype=torch.float32
        )
        self._has_prev_action = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._contact_thumb_geom_ids, self._contact_other_finger_geom_ids = (
            self._find_contact_pinch_geom_ids()
        )
        self._contact_object_geom_ids = self._find_contact_object_geom_ids()
        self.obs_buf = self._reset_all()
        self.num_obs = int(self.obs_buf.shape[1])

    def action_scale_for_ref_idx(self, ref_idx: torch.Tensor | int) -> torch.Tensor:
        """Use the configured residual scale uniformly across tasks and timesteps."""
        return self.action_scale.unsqueeze(0)

    def residual_actions_to_ctrl(
        self, actions: torch.Tensor, ref_idx: torch.Tensor | int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert normalized residual policy actions into MuJoCo controls."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if isinstance(ref_idx, torch.Tensor):
            ref_idx_t = torch.clamp(
                ref_idx.to(self.device, dtype=torch.long),
                0,
                self.max_ref_idx,
            )
            if ref_idx_t.ndim == 0:
                base_ctrl = (
                    self.ctrl_ref[int(ref_idx_t.item())]
                    .unsqueeze(0)
                    .repeat(actions.shape[0], 1)
                )
            else:
                base_ctrl = self.ctrl_ref.index_select(0, ref_idx_t.reshape(-1))
        else:
            base_ctrl = (
                self.ctrl_ref[self._clamp_ref_idx(int(ref_idx))]
                .unsqueeze(0)
                .repeat(actions.shape[0], 1)
            )

        action_scale = self.action_scale_for_ref_idx(ref_idx)
        scaled_residual = actions * action_scale
        if self._single_hand_mode:
            ctrl = base_ctrl.clone()
            ctrl[:, self._ctrl_slice] = ctrl[:, self._ctrl_slice] + scaled_residual
        else:
            ctrl = base_ctrl + scaled_residual
        ctrl = torch.maximum(torch.minimum(ctrl, self._ctrl_upper), self._ctrl_lower)
        return ctrl, scaled_residual, action_scale

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""

    def _find_contact_pinch_geom_ids(self) -> tuple[torch.Tensor, torch.Tensor]:
        thumb_ids = []
        other_ids = []
        for geom_id in range(self.mj_model.ngeom):
            name = self._geom_name(geom_id).lower()
            if name.startswith("collision_hand_right_thumb_"):
                thumb_ids.append(geom_id)
            elif (
                name.startswith("collision_hand_right_index_")
                or name.startswith("collision_hand_right_middle_")
                or name.startswith("collision_hand_right_ring_")
                or name.startswith("collision_hand_right_pinky_")
            ):
                other_ids.append(geom_id)
        return (
            torch.as_tensor(thumb_ids, device=self.device, dtype=torch.long),
            torch.as_tensor(other_ids, device=self.device, dtype=torch.long),
        )

    def _find_contact_object_geom_ids(self) -> torch.Tensor:
        ids = []
        for geom_id in range(self.mj_model.ngeom):
            name = self._geom_name(geom_id).lower()
            if name.startswith("right_object_") or name.startswith("left_object_"):
                ids.append(geom_id)
        return torch.as_tensor(ids, device=self.device, dtype=torch.long)

    def _sample_uniform(self, lo: float, hi: float) -> float:
        lo_f, hi_f = float(lo), float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        return float(np.random.uniform(lo_f, hi_f))

    def _sample_and_apply_dr(
        self,
        done_mask: torch.Tensor | None = None,
        *,
        resample_pair_margin: bool,
    ) -> None:
        if done_mask is None:
            done_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            done_mask = done_mask.to(device=self.device, dtype=torch.bool)
        if not bool(done_mask.any()):
            return

        prev_xy_offset = self._dr_xy_offset.clone()

        if int(self.cfg.num_dr) == 0:
            self._dr_xy_offset.zero_()
            self._dr_object_mass_scale.fill_(float(getattr(self.cfg, "object_mass_scale", 1.0)))
            if resample_pair_margin:
                self._dr_pair_margin = 0.0
        else:
            n_done = int(done_mask.sum().item())
            self._dr_xy_offset[done_mask] = torch.empty(
                n_done, device=self.device, dtype=torch.float32
            ).uniform_(
                float(self.cfg.xy_offset_range[0]), float(self.cfg.xy_offset_range[1])
            )
            self._dr_object_mass_scale[done_mask] = torch.empty(
                n_done, device=self.device, dtype=torch.float32
            ).uniform_(
                float(self.cfg.object_mass_scale_range[0]),
                float(self.cfg.object_mass_scale_range[1]),
            )
            if resample_pair_margin:
                self._dr_pair_margin = self._sample_uniform(
                    self.cfg.pair_margin_range[0], self.cfg.pair_margin_range[1]
                )

        xy_offset_delta = self._dr_xy_offset - prev_xy_offset
        env_param = {
            "pair_margin": float(self._dr_pair_margin),
            "xy_offset": xy_offset_delta,
            "object_mass_scale": self._dr_object_mass_scale,
        }
        load_env_params(self.cfg, self.env, env_param)

    def _clamp_ref_idx(self, idx: int) -> int:
        return int(max(0, min(idx, self.max_ref_idx)))

    def _ref_at(self, sim_step: int):
        idx = self._clamp_ref_idx(sim_step)
        return tuple(r[idx] for r in self.ref_data), idx

    def _ref_at_indices(self, sim_steps: torch.Tensor):
        idx = torch.clamp(sim_steps.to(torch.long), 0, self.max_ref_idx)
        ref = tuple(r.index_select(0, idx) for r in self.ref_data)
        return ref, idx

    def _set_state_from_ref(self, sim_step: int) -> None:
        idx = self._clamp_ref_idx(sim_step)
        qpos_np = self.qpos_ref[idx].detach().cpu().numpy()
        qvel_np = self.qvel_ref[idx].detach().cpu().numpy()
        ctrl_np = self.ctrl_ref[idx].detach().cpu().numpy()
        self.mj_data.qpos[:] = qpos_np
        self.mj_data.qvel[:] = qvel_np
        if self.mj_model.nu > 0:
            self.mj_data.ctrl[:] = ctrl_np
        self.mj_data.time = idx * float(self.cfg.sim_dt)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        sync_env_mujoco(self.cfg, self.env, self.mj_data)
        self._sim_step = idx
        self._start_sim_step = idx
        self._episode_ctrl_steps = 0
        self._sim_step_buf.fill_(idx)
        self._start_sim_step_buf.fill_(idx)
        self._episode_ctrl_steps_buf.zero_()

    def _refresh_mj_data_from_env0(self) -> None:
        qpos0 = get_qpos(self.cfg, self.env)[0].detach().cpu().numpy()
        qvel0 = get_qvel(self.cfg, self.env)[0].detach().cpu().numpy()
        self.mj_data.qpos[:] = qpos0
        self.mj_data.qvel[:] = qvel0
        self.mj_data.time = float(int(self._sim_step_buf[0].item()) * self.cfg.sim_dt)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _set_done_envs_from_ref(self, done_mask: torch.Tensor, sim_steps: torch.Tensor) -> None:
        if not bool(done_mask.any()):
            return
        done_mask = done_mask.to(torch.bool)
        idx = torch.clamp(sim_steps.to(torch.long), 0, self.max_ref_idx)
        qpos_all = get_qpos(self.cfg, self.env).detach().clone()
        qvel_all = get_qvel(self.cfg, self.env).detach().clone()
        qpos_all[done_mask] = self.qpos_ref.index_select(0, idx[done_mask])
        qvel_all[done_mask] = self.qvel_ref.index_select(0, idx[done_mask])
        wp.copy(self.env.data_wp.qpos, wp.from_torch(qpos_all))
        wp.copy(self.env.data_wp.qvel, wp.from_torch(qvel_all))
        if hasattr(self.env.data_wp, "ctrl"):
            ctrl_all = wp.to_torch(self.env.data_wp.ctrl).clone()
            ctrl_all[done_mask] = self.ctrl_ref.index_select(0, idx[done_mask])
            wp.copy(self.env.data_wp.ctrl, wp.from_torch(ctrl_all))
        if hasattr(self.env.data_wp, "time"):
            time_all = wp.to_torch(self.env.data_wp.time).clone()
            time_all[done_mask] = idx[done_mask].to(torch.float32) * float(self.cfg.sim_dt)
            wp.copy(self.env.data_wp.time, wp.from_torch(time_all))
        self._refresh_mj_data_from_env0()

    def _sync_scalar_clocks_from_env0(self) -> None:
        self._sim_step = int(self._sim_step_buf[0].item())
        self._start_sim_step = int(self._start_sim_step_buf[0].item())
        self._episode_ctrl_steps = int(self._episode_ctrl_steps_buf[0].item())

    def _reset_all(self) -> torch.Tensor:
        max_start = max(
            0,
            self.max_ref_idx
            - self.cfg.horizon_steps
            - self.cfg.ctrl_steps
            - self.env_cfg.max_episode_ctrl_steps * self.cfg.ctrl_steps
            - 2,
        )
        if max_start > 0:
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            starts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._start_sim_step_buf = starts.clone()
        self._sim_step_buf = starts.clone()
        self._episode_ctrl_steps_buf.zero_()
        self._set_done_envs_from_ref(
            torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
            self._sim_step_buf,
        )
        self._sample_and_apply_dr(resample_pair_margin=True)
        self._sync_scalar_clocks_from_env0()
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()
        self.episode_return_buf.zero_()
        self.episode_base_return_buf.zero_()
        self.episode_extra_return_buf.zero_()
        self._prev_actions.zero_()
        self._has_prev_action.zero_()
        return self._get_obs()

    def _reset_done(self, done_mask: torch.Tensor) -> None:
        if not bool(done_mask.any()):
            return
        max_start = max(
            0,
            self.max_ref_idx
            - self.cfg.horizon_steps
            - self.cfg.ctrl_steps
            - self.env_cfg.max_episode_ctrl_steps * self.cfg.ctrl_steps
            - 2,
        )
        n_done = int(done_mask.sum().item())
        if max_start > 0:
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(n_done,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            starts = torch.zeros(n_done, device=self.device, dtype=torch.long)
        self._start_sim_step_buf[done_mask] = starts
        self._sim_step_buf[done_mask] = starts
        self._episode_ctrl_steps_buf[done_mask] = 0
        self.episode_length_buf[done_mask] = 0
        self.episode_return_buf[done_mask] = 0.0
        self.episode_base_return_buf[done_mask] = 0.0
        self.episode_extra_return_buf[done_mask] = 0.0
        self.rew_buf[done_mask] = 0.0
        self.reset_buf[done_mask] = False
        self._prev_actions[done_mask] = 0.0
        self._has_prev_action[done_mask] = False
        self._set_done_envs_from_ref(done_mask, self._sim_step_buf)
        self._sample_and_apply_dr(done_mask, resample_pair_margin=False)
        self._sync_scalar_clocks_from_env0()

    def _get_obs(self) -> torch.Tensor:
        return self.observation_for(self.cfg, self.env, self._sim_step_buf)

    def observation_for(self, config, env, phase):
        # MuJoCo step integrates qpos after computing derived poses. Refresh FK so
        # tips and joint state describe the same instant, including after resets.
        with wp.ScopedDevice(env.device):
            mjwarp.kinematics(env.model_wp, env.data_wp)
        return self.goal_observation.build(
            get_qpos(config, env).detach(), wp.to_torch(env.data_wp.site_xpos), phase
        )

    def _obs_tensordict(self) -> TensorDict:
        assert self.obs_buf is not None
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self._obs_tensordict()

    def _active_object_z_tracking_reward(self, qpos_ref: torch.Tensor) -> torch.Tensor:
        if float(self.env_cfg.lifting_rew_scale) == 0.0:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        qpos_sim = get_qpos(self.cfg, self.env)
        if qpos_ref.dim() == 1:
            qpos_ref_b = qpos_ref.unsqueeze(0).repeat(qpos_sim.shape[0], 1)
        else:
            qpos_ref_b = qpos_ref

        if self.cfg.embodiment_type == "bimanual":
            if self.cfg.active_hand == "right":
                sim_z = qpos_sim[:, -12]
                ref_z = qpos_ref_b[:, -12]
            elif self.cfg.active_hand == "left":
                sim_z = qpos_sim[:, -5]
                ref_z = qpos_ref_b[:, -5]
            else:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        elif self.cfg.embodiment_type in ["right", "left"]:
            sim_z = qpos_sim[:, -5]
            ref_z = qpos_ref_b[:, -5]
        else:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        z_error = torch.abs(sim_z - ref_z)
        z_reward = float(self.cfg.object_pos_threshold) - z_error
        return z_reward * float(self.env_cfg.lifting_rew_scale)

    def _active_hand_object_contact_reward(self) -> torch.Tensor:
        if float(self.env_cfg.contact_rew_scale) == 0.0:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        if (
            self._contact_thumb_geom_ids.numel() == 0
            or self._contact_other_finger_geom_ids.numel() == 0
            or self._contact_object_geom_ids.numel() == 0
        ):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        contact_geom = wp.to_torch(self.env.data_wp.contact.geom).to(torch.long)
        contact_dist = wp.to_torch(self.env.data_wp.contact.dist)
        contact_world = wp.to_torch(self.env.data_wp.contact.worldid).to(torch.long)
        geom0 = contact_geom[..., 0].reshape(-1)
        geom1 = contact_geom[..., 1].reshape(-1)
        contact_dist = contact_dist.reshape(-1)
        contact_world = contact_world.reshape(-1)
        valid_contact = (geom0 >= 0) & (geom1 >= 0) & (contact_dist <= 0.0)

        geom0_thumb = torch.isin(geom0, self._contact_thumb_geom_ids)
        geom1_thumb = torch.isin(geom1, self._contact_thumb_geom_ids)
        geom0_other = torch.isin(geom0, self._contact_other_finger_geom_ids)
        geom1_other = torch.isin(geom1, self._contact_other_finger_geom_ids)
        geom0_object = torch.isin(geom0, self._contact_object_geom_ids)
        geom1_object = torch.isin(geom1, self._contact_object_geom_ids)

        thumb_object_contact = valid_contact & (
            (geom0_thumb & geom1_object) | (geom1_thumb & geom0_object)
        )
        other_object_contact = valid_contact & (
            (geom0_other & geom1_object) | (geom1_other & geom0_object)
        )
        valid_world = (contact_world >= 0) & (contact_world < self.num_envs)
        thumb_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        other_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if (thumb_object_contact & valid_world).any():
            thumb_env[contact_world[thumb_object_contact & valid_world]] = True
        if (other_object_contact & valid_world).any():
            other_env[contact_world[other_object_contact & valid_world]] = True
        pinch_contact = thumb_env & other_env
        return pinch_contact.to(torch.float32) * float(self.env_cfg.contact_rew_scale)

    def reset(self):
        self.obs_buf = self._reset_all()
        return self._obs_tensordict()

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = torch.clamp(actions, -self.env_cfg.action_clip, self.env_cfg.action_clip)
        if actions.shape[1] != self.num_actions:
            raise ValueError(
                f"Expected action dim {self.num_actions}, got {actions.shape[1]}"
            )

        total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        base_total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        extra_total_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        term_reason_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        base_ctrl_idx = torch.clamp(self._sim_step_buf, 0, self.max_ref_idx)
        ctrl, _, _ = self.residual_actions_to_ctrl(actions, base_ctrl_idx)

        if float(self.env_cfg.action_smoothness_rew_scale) == 0.0:
            smoothness_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        else:
            action_delta = actions - self._prev_actions
            action_delta_cost = torch.mean(action_delta.square(), dim=1)
            smoothness_rew = -torch.expm1(action_delta_cost) * float(
                self.env_cfg.action_smoothness_rew_scale
            )
            smoothness_rew = torch.where(
                self._has_prev_action,
                smoothness_rew,
                torch.zeros_like(smoothness_rew),
            )
        self._prev_actions = actions.detach().clone()
        self._has_prev_action.fill_(True)

        for sub in range(self.cfg.ctrl_steps):
            step_env(self.cfg, self.env, ctrl)
            self._sim_step_buf += 1
            reset_inactive_side_to_ref(
                self.cfg,
                self.env,
                self.qpos_ref.index_select(0, torch.clamp(self._sim_step_buf, 0, self.max_ref_idx)),
                self.qvel_ref.index_select(0, torch.clamp(self._sim_step_buf, 0, self.max_ref_idx)),
            )
            ref_step, _ = self._ref_at_indices(self._sim_step_buf)
            base_rew, _ = get_reward(self.cfg, self.env, ref_step)
            extra_rew = (
                self._active_object_z_tracking_reward(ref_step[0])
                + self._active_hand_object_contact_reward()
                + smoothness_rew
            )
            rew = base_rew + extra_rew
            # NaN/Inf safety: prevent invalid rewards from contaminating PPO updates.
            if not torch.isfinite(rew).all():
                bad = ~torch.isfinite(rew)
                rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
                base_rew = torch.nan_to_num(base_rew, nan=0.0, posinf=0.0, neginf=0.0)
                extra_rew = torch.nan_to_num(extra_rew, nan=0.0, posinf=0.0, neginf=0.0)
                done[bad] = True
                term_reason_id_buf[bad] = 1.0
            total_rew += rew
            base_total_rew += base_rew
            extra_total_rew += extra_rew
            if sub == self.cfg.ctrl_steps - 1:
                term_rew, _ = get_terminal_reward(self.cfg, self.env, ref_step)
                total_rew += term_rew
                base_total_rew += term_rew
        # Per-env terminate on final state of this control step (keeps all worlds stepped equally).
        term = get_terminate(self.cfg, self.env, ref_step)
        if term.any():
            done |= term
            term_reason_id_buf[term] = 1.0

        self._episode_ctrl_steps_buf += 1
        self.episode_length_buf += 1
        timeout = self._episode_ctrl_steps_buf >= self.max_episode_length
        out_of_ref = self._sim_step_buf >= (self.max_ref_idx - 2)
        newly_timeout = timeout & ~done
        newly_ref_end = out_of_ref & ~done
        done |= timeout | out_of_ref
        term_reason_id_buf[newly_timeout] = 2.0
        term_reason_id_buf[newly_ref_end] = 3.0

        self.rew_buf = total_rew / max(1, self.cfg.ctrl_steps)
        base_rew_buf = base_total_rew / max(1, self.cfg.ctrl_steps)
        extra_rew_buf = extra_total_rew / max(1, self.cfg.ctrl_steps)
        if not torch.isfinite(self.rew_buf).all():
            bad = ~torch.isfinite(self.rew_buf)
            self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
            base_rew_buf = torch.nan_to_num(base_rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
            extra_rew_buf = torch.nan_to_num(extra_rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
            done[bad] = True
            term_reason_id_buf[bad] = 1.0
        self.reset_buf = done.clone()
        self.episode_return_buf += self.rew_buf
        self.episode_base_return_buf += base_rew_buf
        self.episode_extra_return_buf += extra_rew_buf
        # Preserve outputs before any reset path mutates internal buffers.
        rew_out = self.rew_buf.clone()
        done_out = done.clone()

        episode_info = {}
        if done.any():
            done_mask = done.clone()
            episode_return_done = self.episode_return_buf[done_mask]
            episode_base_return_done = self.episode_base_return_buf[done_mask]
            episode_extra_return_done = self.episode_extra_return_buf[done_mask]
            episode_length_done = self._episode_ctrl_steps_buf[done_mask].to(torch.float32)
            sim_step_done = self._sim_step_buf[done_mask].to(torch.float32)
            term_reason_done = term_reason_id_buf[done_mask]
            train_success_target = getattr(self, "train_success_target_sim_step", None)
            if train_success_target is None:
                sim_step_success_count = 0
            else:
                sim_step_success_count = int(
                    (sim_step_done >= float(train_success_target)).sum().item()
                )
            episode_info = {
                "r": float(episode_return_done.mean().item()),
                "r_std": float(episode_return_done.std(unbiased=False).item()),
                "r_min": float(episode_return_done.min().item()),
                "r_max": float(episode_return_done.max().item()),
                "base_r": float(episode_base_return_done.mean().item()),
                "base_r_std": float(episode_base_return_done.std(unbiased=False).item()),
                "base_r_min": float(episode_base_return_done.min().item()),
                "base_r_max": float(episode_base_return_done.max().item()),
                "extra_r": float(episode_extra_return_done.mean().item()),
                "extra_r_std": float(episode_extra_return_done.std(unbiased=False).item()),
                "extra_r_min": float(episode_extra_return_done.min().item()),
                "extra_r_max": float(episode_extra_return_done.max().item()),
                "l": float(episode_length_done.mean().item()),
                "l_std": float(episode_length_done.std(unbiased=False).item()),
                "sim_step": float(sim_step_done.mean().item()),
                "sim_step_std": float(sim_step_done.std(unbiased=False).item()),
                "sim_step_min": float(sim_step_done.min().item()),
                "sim_step_max": float(sim_step_done.max().item()),
                "sim_step_success_count": float(sim_step_success_count),
                "done_count": float(done_mask.sum().item()),
                "term_reason_id": float(term_reason_done.mean().item()),
            }

        if done.any():
            self._reset_done(done)
            next_obs = self._get_obs()
        else:
            next_obs = self._get_obs()
        if not torch.isfinite(next_obs).all():
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.obs_buf = next_obs
        self._sync_scalar_clocks_from_env0()

        # Preserve outputs before _reset_all() clears internal buffers.
        time_outs = (term_reason_id_buf == 2.0) | (term_reason_id_buf == 3.0)
        if episode_info:
            self.extras = {
                "time_outs": time_outs,
                "log": {
                    "/episode_reward": float(episode_info["r"]),
                    "/episode_reward_std": float(episode_info["r_std"]),
                    "/episode_reward_min": float(episode_info["r_min"]),
                    "/episode_reward_max": float(episode_info["r_max"]),
                    "/episode_reward_base": float(episode_info["base_r"]),
                    "/episode_reward_base_std": float(episode_info["base_r_std"]),
                    "/episode_reward_base_min": float(episode_info["base_r_min"]),
                    "/episode_reward_base_max": float(episode_info["base_r_max"]),
                    "/episode_reward_extra": float(episode_info["extra_r"]),
                    "/episode_reward_extra_std": float(episode_info["extra_r_std"]),
                    "/episode_reward_extra_min": float(episode_info["extra_r_min"]),
                    "/episode_reward_extra_max": float(episode_info["extra_r_max"]),
                    "/episode_length": float(episode_info["l"]),
                    "/episode_length_std": float(episode_info["l_std"]),
                    "/sim_step": float(episode_info["sim_step"]),
                    "/sim_step_std": float(episode_info["sim_step_std"]),
                    "/sim_step_min": float(episode_info["sim_step_min"]),
                    "/sim_step_max": float(episode_info["sim_step_max"]),
                    "/sim_step_success_count": float(episode_info["sim_step_success_count"]),
                    "/done_count": float(episode_info["done_count"]),
                    "/term_reason_id": float(episode_info["term_reason_id"]),
                },
                "episode": episode_info,
            }
        else:
            self.extras = {"time_outs": time_outs}
        return self._obs_tensordict(), rew_out, done_out, self.extras

    def close(self):
        return None
