import os
import json
from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
import random
from gymnasium.spaces import Box
from PIL import Image
import torch
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import imageio
from datetime import datetime
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import matplotlib.pyplot as plt

now = datetime.now()

class robot_model_pybullet:
    def __init__(self, urdf_path, base_pos, base_quat, robot_dof_names_ordered, gui=False):

        if gui:
            cid = p.connect(p.GUI)  # or p.DIRECT for headless
        else:
            cid = p.connect(p.DIRECT)
        self.cid = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to find common URDFs

        plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        self.robot_id = p.loadURDF(urdf_path,
                                   basePosition=base_pos,
                                   baseOrientation=base_quat,
                                   useFixedBase=True,
                                   physicsClientId=cid
                                   )

        self.link_name2id = {}
        self.link_name2id[-1] = "base"

        self.dof_names_ordered = robot_dof_names_ordered
        self.dof_name2id = {}

        # Loop through all joints
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i, )

            joint_name = joint_info[1].decode("utf-8")
            self.dof_name2id[joint_name] = i

            link_name = joint_info[12].decode("utf-8")  # child link name
            self.link_name2id[link_name] = i

        self.debug_items = []

    def get_link_poses(self, joint_positions, link_names):

        for idx, pos in enumerate(joint_positions):
            dof_id = self.dof_name2id[self.dof_names_ordered[idx]]
            p.resetJointState(self.robot_id, dof_id, targetValue=pos, physicsClientId=self.cid)

        ret = {}
        for link_name in link_names:
            link_id = self.link_name2id[link_name]
            info = p.getLinkState(self.robot_id,
                                  linkIndex=link_id,
                                  physicsClientId=self.cid)
            world_pos = info[4]
            world_ori = info[5]
            mat = p.getMatrixFromQuaternion(world_ori)
            mat = np.array(mat).reshape((3, 3))

            pose = np.eye(4)
            pose[:3, 3] = world_pos
            pose[:3, :3] = mat

            ret[link_name] = pose

        return ret

    def clear_debug_items(self):
        for item in self.debug_items:
            p.removeUserDebugItem(item)
        self.debug_items = []

    def draw_frame(self, pose):
        self.clear_debug_items()

        t = pose[:3, 3]
        R = pose[:3, :3]
        line_len = 0.5
        x_axis = p.addUserDebugLine(lineFromXYZ=t,
                                    lineToXYZ=R[:3, 0] * line_len + t,
                                    lineColorRGB=[1, 0, 0],
                                    lineWidth=0.1,
                                    lifeTime=0,
                                    )
        y_axis = p.addUserDebugLine(lineFromXYZ=t,
                                    lineToXYZ=R[:3, 1] * line_len + t,
                                    lineColorRGB=[0, 1, 0],
                                    lineWidth=0.1,
                                    lifeTime=0,
                                    )
        z_axis = p.addUserDebugLine(lineFromXYZ=t,
                                    lineToXYZ=R[:3, 2] * line_len + t,
                                    lineColorRGB=[0, 0, 1],
                                    lineWidth=0.1,
                                    lifeTime=0,
                                    )
        self.debug_items.extend([x_axis, y_axis, z_axis])

class RBY1TaskEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg["num_envs"]
        self.device = "cuda"

        self._seed = cfg.get("seed", 0)
        self.verbose = cfg.get("verbose", False)

        self.proc_act = cfg.get("proc_act", True)
        self.proc_obs = cfg.get("proc_obs", True)

        obs_dim = 0
        self.obs_keys = cfg["obs_keys"]
        self.obs_key_ord = list(self.obs_keys.keys())
        for k, v in self.obs_keys.items():
            obs_dim += v
        self.obs_dim = obs_dim

        self.act_keys = cfg["act_keys"]
        self.act_dim = len(self.act_keys)

        self.single_observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.single_action_space = Box(-np.inf, np.inf, shape=(self.act_dim,), dtype=np.float32)

        headless = cfg.get("headless", False)
        self.render_mode = "rgb_array"
        self.cam_h = cfg["cam_h"]
        self.cam_w = cfg["cam_w"]
        self.cam_x_fov = cfg.get("x_fov", 85.)
        self.use_depth = cfg["use_depth"]
        self.use_segm = cfg["use_segm"]
        self.max_episode_length = cfg["max_episode_length"]
        self.success_reward_th = cfg.get("success_reward_th", 0.99)

        self.control_type = cfg["control_type"]

        self.req_dep = cfg.get("use_depth", False)
        self.req_segm = cfg.get("use_segm", False)
        self.req_rgb = cfg.get("use_rgb", False)

        assert self.control_type in [
            "joint_position", "joint_force",
            "joint_position_to_force"], self.control_type

        self.Kp = 100.0
        self.Kd = 10.0


        compute_device_id, graphics_device_id = 0, 0
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1

        # set PhysX-specific parameters
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # Flex params (unused but left here for completeness)
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5

        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        assert self.sim

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z‑up
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

        self.viewer = None
        if not headless:
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )

        self.prog_buf = np.zeros(self.num_envs)

        self.envs_handles = []
        self.camera_handles = []
        self.robot_handles = []
        
        self.actor_names = [] # this contains all entity names
        self.local_dof_names = [] # this contains all joint names, including ones from articulated objects, not just robot
        self.robot_dof_names = None

        self._create_envs()

        robot_dof_local_indices = [self.local_dof_names.index(dn) for dn in self.robot_dof_names]
        self.robot_dof_local_indices = np.array(robot_dof_local_indices, dtype=np.int32) # dof indices in one env

        assert len(self.envs_handles) == len(
            self.robot_handles
        ), "Each env only supports one robot!"

        self.seed(self._seed)

    # ---------------- ENV CREATION ---------------- #
    def _create_envs(self):
        spacing = 2
        asset_root = "./assets"

        # Robot
        robot_asset_file = "rbyxhand/model_modified.urdf"
        robot_opts = gymapi.AssetOptions()
        robot_opts.fix_base_link = True
        robot_opts.flip_visual_attachments = True
        robot_opts.disable_gravity = False
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_opts)
        self.robot_name = "rby1_robot"

        self.robot_eef_names = ["left_xhand_base_link",
                                "left_xhand_palm",
                                "left_xhand_back_cover",
                                "right_xhand_base_link",
                                "right_xhand_palm",
                                "right_xhand_back_cover",
                                ]

        # Desk
        desk_asset_file = "desk/desk.urdf"
        desk_opts = gymapi.AssetOptions()
        desk_opts.fix_base_link = True
        desk_opts.disable_gravity = True
        desk_asset = self.gym.load_asset(self.sim, asset_root, desk_asset_file, desk_opts)

        # Object
        object_asset_file = "ycb/tomato_soup_can.urdf"
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 500
        obj_opts.linear_damping = 0.01
        obj_opts.angular_damping = 0.01
        obj_opts.max_linear_velocity = 100
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, obj_opts)
        self._base_reset_ranges = {
            self.robot_name: [-1, 0, 0, 0, 0, 0, 1],
            # "desk": [-0.3, 0, 0.724, 0, 0, 0, 1],
            "desk": [-0.3, 0, 0.524, 0, 0, 0, 1],
            "object": [-0.6, -0.1, 0.6, 0, 0, 0, 1],
        }



        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        grid = int(np.ceil(np.sqrt(self.num_envs)))
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, grid)
            assert env

            ####################################################################################################
            # Robot
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(*self._base_reset_ranges[self.robot_name][:3])
            robot_actor = self.gym.create_actor(env, robot_asset, robot_pose, self.robot_name, i, 1)
            self.robot_handles.append(robot_actor)

            robot_dof_names = self.gym.get_actor_dof_names(env, robot_actor)

            if i == 0:
                self.actor_names.append(self.robot_name)
                self.local_dof_names += robot_dof_names
                self.robot_dof_names = robot_dof_names

                base_quat = [robot_pose.r.x, robot_pose.r.y, robot_pose.r.z, robot_pose.r.w]
                base_pos = [robot_pose.p.x, robot_pose.p.y, robot_pose.p.z]

                self.robot_model = robot_model_pybullet(f"./assets/{robot_asset_file}",
                                                        base_pos=base_pos,
                                                        base_quat=base_quat,
                                                        robot_dof_names_ordered=robot_dof_names.copy())

            assert robot_dof_names == self.robot_dof_names, f"{robot_dof_names} vs. {self.robot_dof_names}"

            # 1. get dof
            dof_props = self.gym.get_actor_dof_properties(env, robot_actor)
            body_indices = [0,1,2,3,4,5,6,7,8]

            arm_xhand_indices = [idx for idx in range(len(robot_dof_names)) if idx not in body_indices]

            # Position initialization
            desired_body = {}
            for idx in body_indices:
                name = robot_dof_names[idx]
                # default nominal pose = 0.0
                val = 0.0
                # Wheels: keep at zero so robot remains flat
                if name in ["right_wheel", "left_wheel"]:
                    val = 0.0
                # Torso joints: example offsets (in radians)
                elif name == "torso_0":
                    # small upward/backward bend, e.g. +5°
                    val = -0.6 # 50.0 * np.pi / 180.0
                elif name == "torso_1":
                    # main forward tilt, e.g. -10°
                    val = 1.2 # -100.0 * np.pi / 180.0
                elif name == "torso_2":
                    # slight counter-bend, e.g. +5°
                    val = -0.6 # 50.0 * np.pi / 180.0
                elif name == "torso_3":
                    # maybe keep zero or small: here zero
                    val = 0
                elif name == "torso_4":
                    # keep zero
                    val = 0.0
                # Head joints: keep nominal so head stays level
                elif name.startswith("head_"):
                    val = 0.

                # If any other “body” DOF not covered, keep zero or adjust similarly
                else:
                    val = 0.0

                desired_body[idx] = val


            if self.control_type in ["joint_force", "joint_position_to_force"]:

                for idx in arm_xhand_indices:
                    dof_props["driveMode"][idx] = gymapi.DOF_MODE_EFFORT
                for idx in body_indices:
                    dof_props["driveMode"][idx] = gymapi.DOF_MODE_NONE
            elif self.control_type == "joint_position":
                for idx in arm_xhand_indices:
                    dof_props["driveMode"][idx] = gymapi.DOF_MODE_POS
                for idx in body_indices:
                    dof_props["driveMode"][idx] = gymapi.DOF_MODE_NONE
            else:
                raise NotImplementedError(f"Unknown control_type {self.control_type}")

            for idx in body_indices:
                val = desired_body[idx]
                dof_props["lower"][idx] = val
                dof_props["upper"][idx] = val
                dof_props["stiffness"][idx] = 0.0
                dof_props["damping"][idx] = 0.0

            # bimannual mode
            for idx in arm_xhand_indices:
                dof_props["stiffness"][idx] = self.Kp
                dof_props["damping"][idx] = self.Kd

            self.gym.set_actor_dof_properties(env, robot_actor, dof_props)

            ####################################################################################################



            ####################################################################################################
            # Desk
            desk_pose = gymapi.Transform()
            desk_pose.p = gymapi.Vec3(*self._base_reset_ranges["desk"][:3])
            desk_actor = self.gym.create_actor(env, desk_asset, desk_pose, "desk", i, 0)
            if i == 0:
                self.actor_names.append("desk")
                self.local_dof_names += self.gym.get_actor_dof_names(env, desk_actor)
            ####################################################################################################



            ####################################################################################################
            # Object
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(*self._base_reset_ranges["object"][:3])
            obj_actor = self.gym.create_actor(env, object_asset, obj_pose, "object", i, 2)
            if i == 0:
                self.actor_names.append("object")
                self.local_dof_names += self.gym.get_actor_dof_names(env, obj_actor)
            ####################################################################################################

            self.envs_handles.append(env)

        self.gym.prepare_sim(self.sim)

        cprint("Create Env Successfully!", "yellow")
        cprint(f"Total Actors: {self.gym.get_sim_actor_count(self.sim)}", "cyan")

        self._setup_sensor_cameras()

        self._setup_world_camera()
        cprint("Create Cameras Successfully!", "yellow")

        for _ in range(10):
            self._step_sim()

        # State tensors
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.default_dof_state_tensor = self.dof_state_tensor.clone()

    # ---------------- CAMERA ---------------- #
    def _setup_sensor_cameras(self):
        for i, env in enumerate(self.envs_handles):
            cam_props = gymapi.CameraProperties()
            cam_props.width, cam_props.height = self.cam_w, self.cam_h
            cam_props.enable_tensors = True
            cam_props.horizontal_fov = self.cam_x_fov

            cam_handle = self.gym.create_camera_sensor(env, cam_props)
            cam_pose = gymapi.Transform()
            cam_pose.p = gymapi.Vec3(0.5, 0, 1.25)
            # cam_pose.r = gymapi.Quat.from_euler_zyx(0, 30./180*np.pi, np.pi)
            cam_pose.r = gymapi.Quat.from_euler_zyx(0, np.deg2rad(30), np.pi)

            
            self.gym.set_camera_transform(cam_handle, env, cam_pose)
            self.camera_handles.append(cam_handle)
    
    def _setup_world_camera(self, distance=8.0, height=5, focus_height=1):
        if not self.viewer:
            return

        origins = np.array([[p.x, p.y] for p in (self.gym.get_env_origin(e) for e in self.envs_handles)])
        centre_xy = origins.mean(axis=0)       # mid-point over y
        max_x = origins[:, 0].max()            # frontmost robot

        centre = np.array([centre_xy[0], centre_xy[1], focus_height], dtype=np.float32)
        cam_pos = np.array([max_x + distance, centre_xy[1], height], dtype=np.float32)

        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            gymapi.Vec3(*cam_pos),
            gymapi.Vec3(*centre)
        )

    # ---------------- SIM STEP WRAPPERS ---------------- #
    def _step_sim(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _process_actions(self, pred_actions):
        env_ids = list(range(self.num_envs))
        robot_dof_states = self._get_robot_dof_states(env_ids=env_ids)

        if self.control_type in ["joint_position", "joint_position_to_force"]:
            actions = robot_dof_states[:, 0]
        else:
            raise NotImplementedError

        dof_cnt = len(self.local_dof_names)

        # print(actions.shape, pred_actions.shape)

        for k, dof_name in enumerate(self.act_keys):
            dof_idx = self.robot_dof_names.index(dof_name)
            act_idx_for_all_envs = [i*dof_cnt+dof_idx for i in range(self.num_envs)]
            actions[act_idx_for_all_envs] = pred_actions[:, k]

        return actions

    def _process_observations(self, obs):
        proc_obs = []
        for k in self.obs_key_ord:
            proc_obs.append(obs[k])



        return np.concatenate(proc_obs, axis=1)

    # ---------------- MAIN PUBLIC API ---------------- #
    def step(self, actions):



        if self.proc_act:
            actions = self._process_actions(actions)


        if self.control_type == "joint_position_to_force":
            self._apply_actions_via_force_control(actions)
        elif self.control_type == "joint_position":
            self._apply_actions_via_joint_position_control(actions)
        else:
            raise NotImplementedError

        self.prog_buf += 1

        pre_reset_obs = self._compute_observations()
        rewards, dones = self._compute_rewards_and_dones(obs=pre_reset_obs)
        reset_ids = np.where(dones)[0]
        if len(reset_ids):
            self._reset_envs(reset_ids)

        obs = self._compute_observations()
        # for environment just reset, the captured obs is after reset

        # --------‑‑‑‑ NEW: draw viewer each physics frame
        if self.viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        # ------------------------------------------------

        infos = []

        if self.proc_obs:
            obs = self._process_observations(obs=obs)
            pre_reset_obs = self._process_observations(pre_reset_obs)
            infos = [{"terminal_observation": pre_reset_obs[env_id]} for env_id in range(self.num_envs)]

        return obs, rewards, dones, infos


    def reset(self):
        """Hard-reset every env and return the initial observation dict."""
        self._reset_envs()
        # all envs
        torch.cuda.synchronize()
        cprint("Reset All Env Successfully!", "yellow")

        obs = self._compute_observations()
        torch.cuda.synchronize()
        cprint("Compute All Obs Successfully!", "yellow")

        if self.proc_obs:
            obs = self._process_observations(obs=obs)

        return obs

    # ---------------------------------------------------------------------------

    def _reset_envs(self, env_ids=None):
        """Put actors and DOFs back to their nominal poses/vels."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # root states -----------------------------------------------------------
        actor_indices = []
        dof_actors_indices = []

        for env_id in env_ids:
            base_index = env_id * len(self.actor_names)

            for k, name in enumerate(self.actor_names):
                root_idx = base_index + k
                actor_indices.append(root_idx)

                num_dof = self.gym.get_actor_dof_count(self.envs_handles[env_id], k)
                if num_dof > 0:
                    dof_actors_indices.append(root_idx)

                # desired pose (world) -----------------------------------------
                pos, quat = (
                    np.array(self._base_reset_ranges[name][:3], dtype=np.float32),  # ← cast to float
                    np.array(self._base_reset_ranges[name][3:], dtype=np.float32),
                )

                self.root_tensor[root_idx, 0:3] = torch.tensor(pos, device=self.device)
                self.root_tensor[root_idx, 3:7] = torch.tensor(quat, device=self.device)
                self.root_tensor[root_idx, 7:] = 0.0  # zero velocities

        actor_indices = torch.tensor(actor_indices, dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(actor_indices),
            actor_indices.shape[0],
        )

        # DOF states ------------------------------------------------------------
        for env_id in env_ids:
            start = env_id * len(self.local_dof_names) # this dof includes articulate objects, not just robot
            end = start + len(self.local_dof_names)
            self.dof_state_tensor[start:end] = self.default_dof_state_tensor[start:end]

        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
        # this will reset all environment, however, we can only reset environment with dones, this is very important!!!!

        dof_actors_indices = torch.tensor(dof_actors_indices, dtype=torch.int32, device=self.device)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state_tensor),
            gymtorch.unwrap_tensor(dof_actors_indices),  # actor indices
            dof_actors_indices.shape[0]
        )
        # we can only reset environment with dones, this is very important!!!!

        self.prog_buf[env_ids] = 0

        # very important
        for _ in range(10):
            self._step_sim()

    # ---------------------------------------------------------------------------

    def _extend_to_array(self, v,):
        robot_dof_names = self.robot_dof_names

        if isinstance(v, (int, float)):
            return np.ones((self.num_envs * len(robot_dof_names)), dtype="float32") * v
        elif isinstance(v, np.ndarray) and v.shape[0] == len(robot_dof_names) and len(v.shape) == 1:
            return np.tile(v, self.num_envs)
        elif isinstance(v, np.ndarray) and v.shape[0] == len(robot_dof_names) * self.num_envs and len(v.shape) == 1:
            return v
        else:
            raise ValueError

    # ---------------------------------------------------------------------------

    def _compute_robot_torques(self, target_joint_pos, env_ids):
        '''
        actions should be target joint positions with shape (batch_size, dofs)
        this is a simple joint space control
        may extend to support osc control
        '''

        # PID control
        robot_states = self._get_robot_dof_states(env_ids=env_ids)
        robot_joint_pos = robot_states[:, 0]
        robot_joint_vel = robot_states[:, 1]

        pos_error = target_joint_pos - robot_joint_pos
        vel_error = 0. - robot_joint_vel

        Kp = self._extend_to_array(self.Kp)
        Kd = self._extend_to_array(self.Kd)

        torques = Kp * pos_error + Kd * vel_error

        cprint(f"pos err: {pos_error}", "light_red")
        cprint(f"vel err: {vel_error}", "light_red")


        torques = torch.tensor(torques, dtype=torch.float32, device=self.device)

        return torques

    # ---------------------------------------------------------------------------

    def _apply_actions_via_force_control(self, target_joint_pos, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        force_tensor = self._compute_robot_torques(target_joint_pos=target_joint_pos,
                                                   env_ids=env_ids)
        # print("check force: ", force_tensor)

        robot_indices = self._get_all_robot_indices(env_ids=env_ids)
        robot_indices = torch.tensor(robot_indices, dtype=torch.int32, device=self.device)

        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(force_tensor),
            gymtorch.unwrap_tensor(robot_indices),
            len(robot_indices)
        )
        self._step_sim()

    def _apply_actions_via_joint_position_control(self, target_joint_pos, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        robot_indices = self._get_all_robot_indices(env_ids=env_ids)
        robot_indices = torch.tensor(robot_indices, dtype=torch.int32, device=self.device)

        target_pos_tensor = torch.tensor(target_joint_pos, dtype=torch.float32, device=self.device)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_pos_tensor),
            gymtorch.unwrap_tensor(robot_indices),
            len(robot_indices)
        )
        self._step_sim()

    # ---------------------------------------------------------------------------

    def _get_all_robot_indices(self, env_ids):

        robot_indices = np.array([self.actor_names.index(self.robot_name)], dtype=np.int32)
        res = []
        for env_id in env_ids:
            res += (env_id * len(self.actor_names) + robot_indices).tolist()
        return np.array(res, dtype=np.int32)
    
    def _get_all_robot_dof_indices(self, env_ids):
        robot_dof_local_indices = self.robot_dof_local_indices
        idxs = []
        for env_id in env_ids:
            base = env_id * len(self.local_dof_names)
            idxs += (base + robot_dof_local_indices).tolist()
        return np.array(idxs, dtype=np.int32)

    def _get_robot_dof_states(self, env_ids):

        dof_state_np = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).cpu().numpy()
        robot_dof_indices = self._get_all_robot_dof_indices(env_ids=env_ids)
        return dof_state_np[robot_dof_indices]

    def _compute_rewards_and_dones(self, obs):

        rewards = _reward_shaping(obs=obs, robot_name=self.robot_name)
        early_term = _early_termination(obs=obs, object_name="object")

        # Reset if object is dropped or time is up
        dones = (rewards > self.success_reward_th) | (self.prog_buf >= self.max_episode_length) | early_term

        # just for debug
        # reset_ids = np.where(self.prog_buf > 70)[0]
        # n = len(reset_ids)
        # dones[reset_ids[:n//2]] = True

        # rewards = torch.from_numpy(rewards).to(self.device)
        # dones = torch.from_numpy(dones).to(self.device)

        return rewards, dones

    # ---------------------------------------------------------------------------
    def _get_camera_intrinsics(self):
        fov_x_rad = np.radians(self.cam_x_fov)
        cam_y_fov = float(self.cam_h)/float(self.cam_w) * self.cam_x_fov
        fov_y_rad = np.radians(cam_y_fov)

        fx = (self.cam_w / 2.) / np.tan(fov_x_rad / 2.)
        fy = (self.cam_h / 2.) / np.tan(fov_y_rad / 2.)
        cx = self.cam_w / 2.
        cy = self.cam_h / 2.

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return K

    def _get_camera_pose(self, env_id):
        isaac_T = self.gym.get_camera_transform(self.sim,
                                             self.envs_handles[env_id],
                                             self.camera_handles[env_id])
        quat = [isaac_T.r.x, isaac_T.r.y, isaac_T.r.z, isaac_T.r.w]
        rot_mat = R.from_quat(quat).as_matrix()

        pos = np.array([isaac_T.p.x, isaac_T.p.y, isaac_T.p.z])

        # Construct full 4x4 homogeneous transform
        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = pos

        return mat

    def _get_camera_observation(self, env_ids):
        self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        images = []
        depths = []
        segms = []
        for env_id in env_ids:

            env_ptr = self.envs_handles[env_id]
            cam_ptr = self.camera_handles[env_id]

            if self.req_rgb:
                rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_COLOR)
                rgb_np = gymtorch.wrap_tensor(rgb_tensor).cpu().numpy()[..., :3]
                # rgb_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_COLOR)[..., :3]
                images.append(rgb_np)

            if self.req_dep:
                dep_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_DEPTH)
                dep_np = gymtorch.wrap_tensor(dep_tensor).cpu().numpy() * -1.
                # dep_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_DEPTH)
                depths.append(dep_np)

            if self.req_segm:
                seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_SEGMENTATION)
                seg_np = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                # seg_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_SEGMENTATION)

                segms.append(seg_np)

        self.gym.end_access_image_tensors(self.sim)

        depths = None if not depths else np.array(depths)
        segms = None if not segms else np.array(segms)
        images = None if not images else np.array(images)

        ret = {"images": images, "depths": depths, "segms": segms}
        return ret
    
    def _get_low_dim_observation(self, env_ids):

        actor_base_poses = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))[:, :7].cpu().numpy()

        ret = {}
        for k, actor_name in enumerate(self.actor_names):
            actor_idx = np.array([env_id * len(self.actor_names) + k for env_id in env_ids])
            actor_pos = np.array(actor_base_poses[actor_idx, :3])
            actor_quat = np.array(actor_base_poses[actor_idx, 3:7])
            ret[f"{actor_name}_pos"] = actor_pos.reshape((len(env_ids), 3))
            ret[f"{actor_name}_quat"] = actor_quat.reshape((len(env_ids), 4))

        robot_dof_states = self._get_robot_dof_states(env_ids=env_ids)
        joint_pos = np.array(robot_dof_states[:, 0]).reshape((len(env_ids), len(self.robot_dof_names)))
        joint_vel = np.array(robot_dof_states[:, 1]).reshape((len(env_ids), len(self.robot_dof_names)))

        ret[f"{self.robot_name}_joint_pos"] = joint_pos
        ret[f"{self.robot_name}_joint_vel"] = joint_vel

        all_eef_poses = [self.robot_model.get_link_poses(joint_positions=ret[f"{self.robot_name}_joint_pos"][i],
                                                         link_names=self.robot_eef_names) for i in range(len(env_ids))]


        for eef_name in self.robot_eef_names:
            eef_pose = np.array([ith[eef_name] for ith in all_eef_poses])
            ret[f"{self.robot_name}_{eef_name}_pose"] = eef_pose

        return ret

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        ret = {}

        if self.req_rgb | self.req_dep | self.req_segm:
            cam_obs = self._get_camera_observation(env_ids=env_ids)
            # torch.cuda.synchronize()
            if self.verbose:
                cprint("Capture Cam Obs Successfully!", "cyan")
            ret.update(cam_obs)

        low_dim_obs = self._get_low_dim_observation(env_ids=env_ids)
        # torch.cuda.synchronize()
        if self.verbose:
            cprint("Capture Low Dim Obs Successfully!", "cyan")
        ret.update(low_dim_obs)

        # ret = self._debug(ret, env_ids=env_ids)

        return ret

    def get_env_info(self):
        return {
            'observation_space': {'shape': (self.obs_dim,)},
            'action_space': {'shape': (self.act_dim,)},
            'agents': self.num_envs
        }

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)


    def render(self):
        # kept for external calls if desired
        if self.viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            return None
        else:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            env_ptr = self.envs_handles[0]
            cam_ptr = self.camera_handles[0]
            rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_COLOR)
            rgb_np = gymtorch.wrap_tensor(rgb_tensor).cpu().numpy()[..., :3]
            # rgb_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_COLOR)[..., :3]
            return rgb_np


    def close(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _debug(self, obs, env_ids):
        K_mats = []
        pose_mats = []

        view_mats = []
        proj_mats = []

        for env_id in env_ids:

            env_ptr = self.envs_handles[0]
            cam_ptr = self.camera_handles[0]
            view_matrix = self.gym.get_camera_view_matrix(self.sim, env_ptr, cam_ptr)
            proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env_ptr, cam_ptr)

            view_mats.append(view_matrix)
            proj_mats.append(proj_matrix)

            K_mats.append(self._get_camera_intrinsics())
            pose_mats.append(self._get_camera_pose(env_id))

        obs["K_mats"] = np.array(K_mats)
        obs["pose_mats"] = np.array(pose_mats)
        obs["proj_mats"] = np.array(proj_mats)
        obs["view_mats"] = np.array(view_mats)

        return obs

def _reward_shaping(obs, robot_name):
    # this reward is task specific, need to redefine for each task
    obj_pos = obs["object_pos"]
    r_hand_pose = obs[f"{robot_name}_right_xhand_back_cover_pose"]
    r_pos = r_hand_pose[:, :3, 3]

    alpha = 5
    beta = 10

    dist = np.linalg.norm(r_pos - obj_pos, axis=1)
    r1 = np.exp(-alpha * dist)

    lift_height = obj_pos[:, 2] - 0.9
    lift_height[lift_height > 0] = 0
    r2 = np.exp(beta * lift_height)

    rewards = r1 * 0.5 + r2 * 0.5

    return rewards

def _early_termination(obs, object_name):
    # this reward is task specific, need to redefine for each task

    obj_pos = obs[f"{object_name}_pos"]
    n_envs = obj_pos.shape[0]
    term = np.zeros((n_envs))

    term[obj_pos[:, 2] < 0.1] = 1 # object fall over
    return term.astype(np.bool_)


if __name__ == "__main__":


    if not hasattr(gymtorch, "wrap_tensor"):
        setattr(gymtorch, "wrap_tensor", lambda x: x)
    if not hasattr(gymtorch, "unwrap_tensor"):
        setattr(gymtorch, "unwrap_tensor", lambda x: x)


    cfg = {
        "num_envs": 5,
        "headless": True,
        "control_type": "joint_position",
        "max_episode_length": 3000,
        "cam_h": 480,
        "cam_w": 640,
        "proc_obs": False,
        "proc_act": True,
        "use_depth": True,
        "use_segm": False,
        "use_rgb": True,
        "obs_keys": {"object_pos": 3, "object_quat": 4, "rby1_robot_joint_pos": 47, "rby1_robot_pos": 3,
                     "images": 0},
        "act_keys": ["right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
                     "right_arm_5", "right_arm_6", "right_xhand_index_q0", "right_xhand_index_q1", "right_xhand_index_q2",
                     "right_xhand_middle_q1", "right_xhand_middle_q2", "right_xhand_pinky_q1", "right_xhand_pinky_q2",
                     "right_xhand_ring_q1", "right_xhand_ring_q2", "right_xhand_thumb_q0",
                     "right_xhand_thumb_q1", "right_xhand_thumb_q2"]
    }
    env = RBY1TaskEnv(cfg)


    env0 = env.envs_handles[0]
    robot_handle0 = env.robot_handles[0]
    robot_dof_names = env.gym.get_actor_dof_names(env0, robot_handle0)
    num_envs = cfg["num_envs"]
    dof_per_env = len(robot_dof_names)
    print("Robot DOF count:", dof_per_env)

    ctrl_idx = [env.robot_dof_names.index(dof_name) for dof_name in cfg["act_keys"]]


    body_indices = [0,1,2,3,4,5,6,7,8]
    arm_xhand_indices = [i for i in range(dof_per_env) if i not in body_indices]
    print("Body indices (fixed):", body_indices)
    print("Arm/Xhand indices:", arm_xhand_indices)
    print("Arm/Xhand names:", [robot_dof_names[i] for i in arm_xhand_indices])



    dof_props = env.gym.get_actor_dof_properties(env0, robot_handle0)

    lower_limits = dof_props["lower"].copy()  # shape (dof_per_env,)
    upper_limits = dof_props["upper"].copy()  # shape (dof_per_env,)


    # env.reset()
    # raw = env._get_robot_dof_states(env_ids=None)  # shape (num_envs * dof_per_env,)
    # jpos_all = raw.reshape(num_envs, dof_per_env)
    # body_initial = jpos_all[:, body_indices]  # shape (num_envs, len(body_indices))




    lower_arm = lower_limits[arm_xhand_indices]    # shape (len(arm_xhand_indices),)
    upper_arm = upper_limits[arm_xhand_indices]    # shape (len(arm_xhand_indices),)
    mid_arm = 0.5 * (lower_arm + upper_arm)
    half_range_arm = 0.5 * (upper_arm - lower_arm)



    init_obs = env.reset()
    obs_buff = [init_obs]

    try:
        robot_states = env._get_robot_dof_states(env_ids=None)  # shape (num_envs * dof_per_env,)
        jpos_all = robot_states.reshape(num_envs, dof_per_env)
    except Exception:
        # fallback
        flat = init_obs[f"{env.robot_name}_joint_pos"]
        jpos_all = flat.reshape(num_envs, dof_per_env)
    body_initial = jpos_all[:, body_indices].copy()  # shape (num_envs, len(body_indices))
    print("Recorded initial body joint positions:", body_initial.shape)


    rng = np.random.default_rng()
    amplitudes = half_range_arm[np.newaxis, :].repeat(num_envs, axis=0)  # shape (num_envs, len(arm_xhand_indices))


    frequencies = rng.uniform(0.1, 1.0, size=(num_envs, len(arm_xhand_indices)))

    phases = rng.uniform(0.0, 2*np.pi, size=(num_envs, len(arm_xhand_indices)))

    step_count = 0
    dt = 1.0/60.0

    cur_time = now.strftime("%H-%M-%S")
    save_root = f"./debug_env_{cur_time}"

    try:
        for _ in range(500):

            print(f"step {_} ...")

            t = step_count * dt

            action = np.zeros((num_envs, dof_per_env), dtype=np.float32)

            for env_i in range(num_envs):
                action[env_i, body_indices] = body_initial[env_i]

            #    target = mid_arm + half_range_arm * sin(2π f t + phase)
            sin_vals = np.sin(2 * np.pi * frequencies * t + phases)  # shape (num_envs, len(arm_xhand_indices))
            targets_arm = mid_arm[np.newaxis, :] + amplitudes * sin_vals  # shape (num_envs, len(...))
            for env_i in range(num_envs):
                action[env_i, arm_xhand_indices] = targets_arm[env_i]

            action_sel = action[:, ctrl_idx]

            obs, rewards, dones, infos = env.step(action_sel)
            actor_pos = obs["rby1_robot_pos"]
            cprint(f"robot pos shape: {actor_pos.shape}", "cyan")
            for act_id in range(len(actor_pos)):
                print(actor_pos[act_id])

            obs_buff.append(obs)

            if np.any(dones):
                reset_ids = np.where(dones)[0]
                cprint(f"Reset env ids: {reset_ids}", "yellow")
                for idx in reset_ids:

                    frequencies[idx] = rng.uniform(0.1, 1.0, size=(len(arm_xhand_indices),))
                    phases[idx] = rng.uniform(0.0, 2*np.pi, size=(len(arm_xhand_indices),))

                    # new_states = env._get_robot_dof_states(env_ids=[idx]).reshape(1, dof_per_env)
                    # body_initial[idx] = new_states[0, body_indices]


                cur_prog_buff = env.prog_buf
                cprint(f"Prog buff: {cur_prog_buff}", "cyan")

            step_count += 1

        save_dir = f"{save_root}"
        os.makedirs(save_dir, exist_ok=True)

        for env_id in range(len(obs["images"])):
            with imageio.get_writer(f'{save_dir}/output_video_{env_id}.mp4', fps=5) as writer:
                for obs in obs_buff:
                    writer.append_data(obs["images"][env_id])

        # for env_id in range(len(obs["images"])):
        #     with imageio.get_writer(f'{save_dir}/output_video_{env_id}_w_pose.mp4', fps=5) as writer:
        #         for obs in obs_buff:
        #             writer.append_data(obs["debug_frames"][env_id])
        for ind, obs in enumerate(obs_buff):
            save_dir_at_t = f"{save_dir}/{ind:06d}"
            os.makedirs(save_dir_at_t, exist_ok=True)
            for k in obs:
                np.save(f"{save_dir_at_t}/{k}.npy", obs[k])


    except KeyboardInterrupt:
        print("Terminated")
    finally:
        env.close()

