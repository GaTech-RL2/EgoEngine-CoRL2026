import numpy as np
from termcolor import cprint
from robosuite.utils.transform_utils import quat2mat
from robosuite import make
import gymnasium
from gymnasium import spaces
from .custom_tasks import *
from .rl_utils import _dense_reward, _early_termination

import matplotlib.pyplot as plt
import time

DEBUG = False
SET_CFG = False

def remove_prefixes(s, prefixes):
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s  # return unchanged if no prefix matche

def rotz(theta: float):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def set_flat_yaw(T: np.ndarray, yaw_abs: float) -> np.ndarray:
    """Set orientation to lie flat on desk with absolute yaw."""
    Tn = T.copy()
    Tn[:3, :3] = rotz(yaw_abs)
    return Tn

def add_flat_yaw(T: np.ndarray, yaw_delta: float) -> np.ndarray:
    """Keep current yaw (from T), add yaw_delta, and enforce flat (no roll/pitch)."""
    R = T[:3, :3]
    # current yaw from x-axis projection (equivalently, atan2(R[1,0], R[0,0]))
    yaw0 = np.arctan2(R[1, 0], R[0, 0])
    return set_flat_yaw(T, yaw0 + yaw_delta)

class RobosuiteFuncRetarWrapper(gymnasium.Env):
    def __init__(self, env_name, robot_name,
                 control_cfg, obs_keys, obj_info,
                 retarget_data, track_budget, use_early_term,
                 use_camera_obs=False, camera_names="cameraview", camera_segmentations=False, camera_heights=1080,
                 camera_widths=1920):

        # ===== tracking task stuff =====
        self.sides = ["left", "right"]
        self.track_data = retarget_data["traj"]
        self.init_state = retarget_data["init_state"]

        # ---- objects are optional now ----
        self.obj_info = obj_info if obj_info is not None else {}
        self.has_objects = ("target" in self.obj_info) and ("tool" in self.obj_info)

        self.track_index = 0
        self.track_budget = track_budget
        self.use_early_term = use_early_term

        init_desk_pose = np.array(self.init_state["desk"])
        self.init_q = self.init_state["robot_cfg"]

        # robot base shift used by your camera math
        robot_base_pos = np.array([-0.7, 0.0, 0.0])

        # ===== (OPTIONAL) rotate initial target/tool yaw if objects exist =====
        if self.has_objects:
            delta1 = 0.0  # radians
            self.obj_info["target"]["pose"] = add_flat_yaw(self.obj_info["target"]["pose"], delta1)

            delta2 = -0.0  # radians
            self.obj_info["tool"]["pose"] = add_flat_yaw(self.obj_info["tool"]["pose"], delta2)

        # table offset (independent of objects)
        table_offset = robot_base_pos + init_desk_pose[:3, 3] + np.array([0.0, 0.0, 0.025])

        # ===== create env =====
        env_obj_info = self.obj_info if self.has_objects else None
        self.env = make(
            env_name,
            robots=robot_name,
            controller_configs=control_cfg,
            obj_info=env_obj_info,
            use_camera_obs=use_camera_obs,
            has_offscreen_renderer=use_camera_obs,
            camera_names=[camera_names],
            has_renderer=False,
            reward_shaping=True,
            control_freq=20,
            table_offset=table_offset,
            ignore_done=True,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
        )

        robot = self.env.robots[0]
        robot.print_action_info()

        self.camera_names = camera_names
        obs = self.env.reset()


        self.obs_keys = []
        obs_dim = 0
        for k in obs_keys:
            if k in obs:
                self.obs_keys.append(k)
                obs_dim += obs[k].shape[0]
        if len(self.obs_keys) == 0:
            raise RuntimeError(f"No valid obs keys from {obs_keys}. Available keys: {list(obs.keys())}")
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ===== action space =====
        rs_act_spec = self.env.action_spec
        act_low = np.ones(rs_act_spec[0].shape, dtype=np.float32) * -1.0
        act_high = np.ones(rs_act_spec[0].shape, dtype=np.float32)

        grip_idx, grip_low, grip_high = self.env._get_ctrl_limit()
        act_low[grip_idx] = grip_low
        act_high[grip_idx] = grip_high

        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # ===== optional debug (only if objects exist and keys present) =====
        if 'DEBUG' in globals() and DEBUG and self.has_objects:
            if f"{self.camera_names}_image" in obs:
                obs_img = obs[f"{self.camera_names}_image"][::-1]
                plt.imshow(obs_img); plt.title(f"min max: {obs_img.min()} {obs_img.max()}"); plt.show()

    def _set_full_config(self, conf):
        obs = self.env._set_robot_config(conf)
        return obs

    def _get_actuator_mapping(self):
        actor_map = self.env._get_actuator_mapping()
        return actor_map

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        obs = self.env.reset()

        # optional debug prints only if objects exist and keys there
        if 'DEBUG' in globals() and DEBUG and self.has_objects:
            if "target_quat" in obs and "target_pos" in obs:
                target_mat = quat2mat(obs["target_quat"])
                print("target mat:", target_mat)
                print("target pos:", obs["target_pos"])
            if "tool_quat" in obs and "tool_pos" in obs:
                tool_mat = quat2mat(obs["tool_quat"])
                print("tool mat:", tool_mat)
                print("tool pos:", obs["tool_pos"])
            if f"{self.camera_names}_image" in obs:
                obs_img = obs[f"{self.camera_names}_image"][::-1]
                plt.imshow(obs_img); plt.title(f"min max: {obs_img.min()} {obs_img.max()}"); plt.show()

        self.track_index = 0
        obs = self._set_full_config(self.init_q)

        use_obs = [obs[k] for k in self.obs_keys]
        use_obs = np.concatenate(use_obs, axis=0, dtype=np.float32)
        return (use_obs, {})

    def step(self, action):
        if 'DEBUG' in globals() and DEBUG:
            print(f"action: {len(action)}", action)

        if 'SET_CFG' in globals() and SET_CFG:
            # debug: set desired config directly
            info = {}
            subgoal_index = self.track_index // self.track_budget
            subgoal_index = min(subgoal_index, len(self.track_data))
            obs = self._set_full_config(self.track_data[subgoal_index]["robot_cfg"])
        else:
            obs, _, _, info = self.env.step(action)

        if 'DEBUG' in globals() and DEBUG:
            if f"{self.camera_names}_image" in obs:
                obs_img = obs[f"{self.camera_names}_image"][::-1]
                plt.imshow(obs_img); plt.title(f"min max: {obs_img.min()} {obs_img.max()}"); plt.show()

        if self.has_objects:
            dense_reward = self._dense_reward(obs=obs)
        else:
            dense_reward=None
        done = self._early_term(obs=obs)

        filt_obs = [obs[k] for k in self.obs_keys]
        filt_obs = np.concatenate(filt_obs, axis=0, dtype=np.float32)

        self.track_index += 1
        if 'cprint' in globals():
            cprint(f"track index: {self.track_index}", "red")

        return filt_obs, dense_reward, done, done, info

    def _dense_reward(self, obs):
        subgoal_index = self.track_index // self.track_budget
        subgoal_index = min(subgoal_index, len(self.track_data))

        next_robot_config = self.track_data[subgoal_index]["robot_cfg"]
        next_tool_pose = np.array(self.track_data[subgoal_index]["tool"])
        next_target_pose = np.array(self.track_data[subgoal_index]["target"])
        next_desk_pose = np.array(self.track_data[subgoal_index]["desk"])
        at_the_end = subgoal_index == len(self.track_data) - 1


        next_target_pose[:3, 3] -= next_desk_pose[:3, 3]
        next_tool_pose[:3, 3] -= next_desk_pose[:3, 3]

        dense_reward, info = _dense_reward(
            obs=obs,
            next_robot_cfg=next_robot_config,
            next_tool_pose=next_tool_pose,
            next_target_pose=next_target_pose,
            at_the_end=at_the_end,
            sides=self.sides
        )
        return dense_reward

    def _early_term(self, obs):
        subgoal_index = self.track_index // self.track_budget
        subgoal_index = min(subgoal_index, len(self.track_data))

        if not self.use_early_term:
            return subgoal_index == len(self.track_data) - 1

        next_robot_config = self.track_data[subgoal_index]["robot_cfg"]
        next_tool_pose = np.array(self.track_data[subgoal_index]["tool"])
        next_target_pose = np.array(self.track_data[subgoal_index]["target"])
        next_desk_pose = np.array(self.track_data[subgoal_index]["desk"])

        next_target_pose[:3, 3] -= next_desk_pose[:3, 3]
        next_tool_pose[:3, 3] -= next_desk_pose[:3, 3]

        done, info = _early_termination(
            obs=obs,
            next_robot_cfg=next_robot_config,
            next_tool_pose=next_tool_pose,
            next_target_pose=next_target_pose,
            sides=self.sides
        )

        if subgoal_index == len(self.track_data) - 1:
            done = True
        return done

    def render(self, mode="rgb_array"):
        observations = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        return observations[f"{self.camera_names}_image"][::-1]

    @property
    def render_mode(self):
        return "rgb_array"

    def close(self):
        self.env.close()
