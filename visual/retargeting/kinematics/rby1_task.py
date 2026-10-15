import os

from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
from PIL import Image
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import imageio
from datetime import datetime
now = datetime.now()

import matplotlib.pyplot as plt


class RBY1TaskEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg["num_envs"]
        self.device = "cuda"

        headless = cfg.get("headless", False)
        self.cam_h = cfg["cam_h"]
        self.cam_w = cfg["cam_w"]
        self.use_depth = cfg["use_depth"]
        self.use_segm = cfg["use_segm"]
        self.max_episode_length = cfg["max_episode_length"]

        self.control_type = cfg["control_type"]

        assert self.control_type in ["joint_position", "joint_force", "joint_position_to_force"], self.control_type

        self.Kp = 100.
        self.Kd = 10.


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
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # set Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5


        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id,
                                       gymapi.SIM_PHYSX, sim_params)
        assert self.sim, self.sim

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

        self.viewer = None
        # For headless mode
        if not headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        self.prog_buf = np.array([0 for _ in range(self.num_envs)])

        self.envs_handles = []
        self.camera_handles = []
        self.robot_handles = []

        self.actor_names = []
        self.local_dof_names = []
        self.local_robot_indices = []

        self._create_envs()

        assert len(self.envs_handles) == len(self.robot_handles), "Each env only support one robot!"

    def _create_envs(self):
        spacing = 2
        asset_root = "./assets"

        # Load robot
        robot_asset_file = "rbyxhand/model.urdf"
        robot_options = gymapi.AssetOptions()
        robot_options.fix_base_link = True
        robot_options.flip_visual_attachments = True
        robot_options.disable_gravity = False
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_options)
        self.robot_name = "rby1_robot"

        desk_asset_file = "desk/desk.urdf"
        desk_asset_options = gymapi.AssetOptions()
        desk_asset_options.fix_base_link = True  # makes it static
        desk_asset_options.disable_gravity = True
        desk_asset_options.flip_visual_attachments = False  # if needed
        desk_asset = self.gym.load_asset(self.sim, asset_root, desk_asset_file, desk_asset_options)

        # Load cube
        object_asset_file = "ycb/adjustable_wrench.urdf"
        object_options = gymapi.AssetOptions()
        object_options.density = 500
        object_options.linear_damping = 0.01
        object_options.angular_damping = 0.01
        object_options.max_linear_velocity = 100
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_options)

        self._base_reset_ranges = {self.robot_name: [-1, 0, 0, 0, 0, 0, 1],
                                   "desk": [0, 0, 0.85, 0, 0, 0, 1],
                                   "object": [0, 0, 0.95, 0, 0, 0, 1],}


        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            assert env, env

            # Franka pose
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(*self._base_reset_ranges[self.robot_name][:3])
            robot_actor = self.gym.create_actor(env, robot_asset, robot_pose, self.robot_name, i, 1)
            self.robot_handles.append(robot_actor)
            if i == 0:
                self.actor_names.append(self.robot_name)
                self.local_dof_names += self.gym.get_actor_dof_names(env, robot_actor)
                self.local_robot_indices.append(0)


            dof_props = self.gym.get_actor_dof_properties(env, robot_actor)

            if self.control_type in ["joint_force", "joint_position_to_force"]:
                dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            elif self.control_type == "joint_position":
                dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            else:
                raise NotImplemented
            dof_props["stiffness"][:] = self.Kp  # Kp
            dof_props["damping"][:] = self.Kd  # Kd
            self.gym.set_actor_dof_properties(env, robot_actor, dof_props)

            desk_pose = gymapi.Transform()
            desk_pose.p = gymapi.Vec3(*self._base_reset_ranges["desk"][:3])  # x, y, z of desk
            desk_actor = self.gym.create_actor(env, desk_asset, desk_pose, "desk", i, 0)
            if i == 0:
                self.actor_names.append("desk")
                self.local_dof_names += self.gym.get_actor_dof_names(env, desk_actor)

            # Cube pose
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(*self._base_reset_ranges["object"][:3])
            object_actor = self.gym.create_actor(env, object_asset, object_pose, "object", i, 2)
            if i == 0:
                self.actor_names.append("object")
                self.local_dof_names += self.gym.get_actor_dof_names(env, object_actor)


            self.envs_handles.append(env)



        self.gym.prepare_sim(self.sim)
        cprint("Create Env Successfully!", "cyan")

        all_actor_ = self.gym.get_sim_actor_count(self.sim)
        cprint(f"Total Actors: {all_actor_}", "cyan")

        self.setup_cameras()
        cprint("Create Cameras Successfully!", "cyan")

        for _ in range(10):
            self._step_sim()

        # DOF state tensor
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # cprint(f"check init dof state: {self.dof_state_tensor}", "yellow")
        # cprint(f"check init pose: {self.root_tensor}", "yellow")
        self.default_dof_state_tensor = self.dof_state_tensor.clone()


    def setup_cameras(self,):
        for i, env in enumerate(self.envs_handles):
            cam_props = gymapi.CameraProperties()
            cam_props.width = self.cam_w
            cam_props.height = self.cam_h
            cam_props.enable_tensors = True

            # Create camera sensor inside the environment
            camera_handle = self.gym.create_camera_sensor(env, cam_props)
            # # Position the camera
            camera_pose = gymapi.Transform()
            env_orig = self.gym.get_env_origin(env)

            camera_pose.p = gymapi.Vec3(
                env_orig.x+ 1.5,
                env_orig.y,
                env_orig.z + 2.0  # 1m above cube
            )
            camera_pose.r = gymapi.Quat.from_euler_zyx(-0, 0.4, 3.14)  # Orientation
            # Set camera pose for env i here
            self.gym.set_camera_transform(camera_handle, env, camera_pose)
            # Attach camera to robot or set fixed transform
            self.camera_handles.append(camera_handle)

    def get_camera_observation(self, env_ids=None, req_dep=False, req_segm=False):
        self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        if env_ids is None:
            env_ids = list(range(self.num_envs))

        images = []
        depths = []
        segms = []
        for env_id in env_ids:

            env_ptr = self.envs_handles[env_id]
            cam_ptr = self.camera_handles[env_id]

            rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_COLOR)
            rgb_np = gymtorch.wrap_tensor(rgb_tensor).cpu().numpy()[..., :3]
            # rgb_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_COLOR)[..., :3]
            images.append(rgb_np)

            if req_dep:
                dep_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_DEPTH)
                dep_np = gymtorch.wrap_tensor(dep_tensor).cpu().numpy()
                # dep_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_DEPTH)

                depths.append(dep_np)

            if req_segm:
                seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_SEGMENTATION)
                seg_np = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                # seg_np = self.gym.get_camera_image(self.sim, env_ptr, cam_ptr, gymapi.IMAGE_SEGMENTATION)

                segms.append(seg_np)

        self.gym.end_access_image_tensors(self.sim)

        depths = None if not depths else np.array(depths)
        segms = None if not segms else np.array(segms)
        images = np.array(images)

        ret = {"images": images, "depths": depths, "segms": segms}
        return ret

    def get_low_dim_observation(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        to_vec = lambda p: np.array([p.x, p.y, p.z])
        env_origs = np.array([to_vec(self.gym.get_env_origin(ev)) for ev in self.envs_handles])
        env_origs = env_origs[np.array(env_ids)]

        actor_base_poses = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))[:, :7].cpu().numpy()

        ret = {}
        for k, actor_name in enumerate(self.actor_names):
            actor_idx = np.array([env_id * len(self.actor_names) + k for env_id in env_ids])
            ret[f"{actor_name}_pos"] = actor_base_poses[actor_idx, :3] - env_origs
            ret[f"{actor_name}_quat"] = actor_base_poses[actor_idx, 3:7]

        robot_dof_states = self._get_robot_dof_states(env_ids=env_ids)
        robot_dof_names = self.gym.get_actor_dof_names(self.envs_handles[0], self.robot_handles[0])
        ret[f"{self.robot_name}_joint_pos"] = robot_dof_states[:, 0]
        ret[f"{self.robot_name}_joint_vel"] = robot_dof_states[:, 1]



        return ret

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        cam_obs = self.get_camera_observation(env_ids=env_ids,
                                              req_dep=self.use_depth,
                                              req_segm=self.use_segm,)

        torch.cuda.synchronize()
        # cprint("Capture Cam Obs Successfully!", "cyan")


        low_dim_obs = self.get_low_dim_observation(env_ids=env_ids)

        torch.cuda.synchronize()
        # cprint("Capture Low Dim Obs Successfully!", "cyan")

        ret = {**cam_obs, **low_dim_obs}

        return ret

    def _step_sim(self):
        # 2. Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 3. Refresh states
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _reset_envs(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        actor_indices = []
        dof_actors_indices = []

        actor_root_states = self.root_tensor  # shape: (num_total_actors, 13)
        for env_id in env_ids:
            base_index = env_id * len(self.actor_names)
            env_orig = self.gym.get_env_origin(self.envs_handles[env_id])

            for name, reset_pose in self._base_reset_ranges.items():
                reset_pos, reset_quat = reset_pose[:3], reset_pose[3:]
                reset_pos = np.array(reset_pos) + np.array([env_orig.x, env_orig.y, env_orig.z])



                local_actor_index = self.actor_names.index(name)
                cur_index = base_index + local_actor_index
                actor_indices.append(cur_index)

                num_dof = self.gym.get_actor_dof_count(self.envs_handles[env_id], local_actor_index)
                if num_dof > 0:
                    dof_actors_indices.append(cur_index)

                # Reset position and orientation
                if local_actor_index >= 2:
                    rad = np.deg2rad(30)
                    delta_yaw = np.random.uniform(-rad, rad, ())
                    delta_rot = R.from_euler('z', delta_yaw)
                    reset_rot = R.from_quat(reset_quat)
                    reset_quat = (delta_rot * reset_rot).as_quat()

                actor_root_states[cur_index, 0:3] = torch.tensor(reset_pos, device=self.device)
                actor_root_states[cur_index, 3:7] = torch.tensor(reset_quat, device=self.device)
                # Reset velocity
                actor_root_states[cur_index, 7:] = 0.0

        dof_states = self.dof_state_tensor  # shape: (num_total_dofs, 2)

        # cprint("check detailed size: ", "light_magenta")
        # print("dof state: ", self.dof_state_tensor.shape)
        # print("num env: ", self.num_envs)
        # print("env ids: ", env_ids)
        # print("actor index: ", actor_indices)
        # print(f"check device: {dof_states.device} {self.default_dof_state_tensor.device}")

        for env_id in env_ids:
            start = env_id * len(self.local_dof_names)
            end = start + len(self.local_dof_names)

            dof_states[start:end, 0] = self.default_dof_state_tensor[start:end, 0]
            dof_states[start:end, 1] = 0.0

        actor_indices = torch.tensor(actor_indices, dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(actor_root_states),
            gymtorch.unwrap_tensor(actor_indices),  # actor indices
            actor_indices.shape[0]
        )

        dof_actors_indices = torch.tensor(dof_actors_indices, dtype=torch.int32, device=self.device)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_states),
            gymtorch.unwrap_tensor(dof_actors_indices),  # actor indices
            dof_actors_indices.shape[0]
        )

        for env_id in env_ids:
            self.prog_buf[env_id] = 0

        for _ in range(10):
            self._step_sim()

    def reset(self):
        self._reset_envs(env_ids=None)
        torch.cuda.synchronize()
        # cprint("Reset Partial Envs Successfully!", "cyan")

        obs = self.compute_observations(env_ids=None)

        return obs

    def _extend_to_array(self, v,):
        robot_dof_names = self.gym.get_actor_dof_names(self.envs_handles[0], self.robot_handles[0])
        if isinstance(v, (int, float)):
            return np.ones((self.num_envs * len(robot_dof_names)), dtype="float32") * v
        elif isinstance(v, np.ndarray) and v.shape[0] == len(robot_dof_names) and len(v.shape) == 1:
            return np.tile(v, self.num_envs)
        elif isinstance(v, np.ndarray) and v.shape[0] == len(robot_dof_names) * self.num_envs and len(v.shape) == 1:
            return v
        else:
            raise ValueError

    def _compute_robot_torques(self, target_joint_pos, ):
        '''
        actions should be target joint positions with shape (batch_size, dofs)
        this is a simple joint space control
        may extend to support osc control
        '''

        # PID control
        robot_states = self._get_robot_dof_states()
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

    def _apply_actions_via_force_control(self, target_joint_pos):
        '''

        '''

        force_tensor = self._compute_robot_torques(target_joint_pos=target_joint_pos)
        # print("check force: ", force_tensor)

        robot_indices = self._get_all_robot_indices()
        robot_indices = torch.tensor(robot_indices, dtype=torch.int32, device=self.device)

        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(force_tensor),
            gymtorch.unwrap_tensor(robot_indices),
            len(robot_indices)
        )
        self._step_sim()

    def _apply_actions_via_joint_position_control(self, target_joint_pos):

        robot_indices = self._get_all_robot_indices()
        robot_indices = torch.tensor(robot_indices, dtype=torch.int32, device=self.device)

        target_pos_tensor = torch.tensor(target_joint_pos, dtype=torch.float32, device=self.device)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_pos_tensor),
            gymtorch.unwrap_tensor(robot_indices),
            len(robot_indices)
        )
        self._step_sim()


    def _get_all_robot_indices(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        robot_indices = np.array(self.local_robot_indices)
        robot_indices_extend = []
        for env_id in env_ids:
            robot_indices_extend += (env_id * len(self.actor_names) + robot_indices).tolist()
        return np.array(robot_indices_extend)

    def _get_all_robot_dof_indices(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        robot_dof_names = self.gym.get_actor_dof_names(self.envs_handles[0], self.robot_handles[0])
        # print("robot dof names: ", robot_dof_names)
        robot_dof_index = np.array([self.local_dof_names.index(dn) for dn in robot_dof_names])
        robot_dof_index_extend = []
        for env_id in env_ids:
            robot_dof_index_extend += (env_id * len(self.local_dof_names) + robot_dof_index).tolist()
        return np.array(robot_dof_index_extend)

    def _get_robot_dof_states(self, env_ids=None):
        dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).cpu().numpy()
        robot_dof_index = self._get_all_robot_dof_indices(env_ids=env_ids)
        robot_dof_states = dof_state_tensor[robot_dof_index]
        return robot_dof_states


    def _reward_shaping(self,):

        return np.zeros((self.num_envs, ))

    def compute_rewards_and_dones(self):
        rewards = self._reward_shaping()
        # Reset if object is dropped or time is up
        dones = (rewards > 0.99) | (self.prog_buf >= self.max_episode_length)
        return rewards, dones

    def step(self, actions):
        # 1. Apply actions to actors
        if self.control_type == "joint_position_to_force":
            self._apply_actions_via_force_control(target_joint_pos=actions)
        elif self.control_type == "joint_position":
            self._apply_actions_via_joint_position_control(target_joint_pos=actions)
        else:
            raise NotImplemented

        for i in range(self.num_envs):
            self.prog_buf[i] += 1.

        # 4. rewards, and dones
        rewards, dones = self.compute_rewards_and_dones()

        # 5. Reset if needed
        env_ids = np.where(dones > 0)[0]

        if len(env_ids) > 0:
            self._reset_envs(env_ids=env_ids)

        obs = self.compute_observations(env_ids=None)

        # 6. Return
        return obs, rewards, dones, {}


    def render(self):
        if self.viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

    def close(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == '__main__':
    cfg = {
        "num_envs": 2,
        "headless": False,
        "control_type": "joint_position",
        "max_episode_length": 300,
        "cam_h": 480,
        "cam_w": 640,
        "use_depth": True,
        "use_segm": True,
    }

    env = RBY1TaskEnv(cfg)
    cur_time = now.strftime("%H-%M-%S")
    save_root = f"./debug_env_{cur_time}"

    def save_obs(cam_obs, save_dir, ):
        imgs = cam_obs["images"]
        deps = cam_obs["depths"]
        segms = cam_obs["segms"]
        for j in range(len(imgs)):
            Image.fromarray(imgs[j]).save(f"{save_dir}/image_{j}.png")
            np.save(f"{save_dir}/depth_{j}.npy", deps[j])
            np.save(f"{save_dir}/segm_{j}.npy", segms[j])

    obs = env.reset()
    init_jpos = obs["rby1_robot_joint_pos"]
    obs_buff = [obs]
    for t in range(300000000000000000000000000000):

        actions = init_jpos + t * np.ones(init_jpos.shape) * 0.01

        obs, _, _, _ = env.step(actions)
        # print("all keys: ", obs.keys())
        obs_buff.append(obs)

    '''
    save_dir = f"{save_root}"
    os.makedirs(save_dir, exist_ok=True)

    with imageio.get_writer(f'{save_dir}/output_video.mp4', fps=10) as writer:
        for obs in obs_buff:
            writer.append_data(obs["images"][0])
    '''
    env.close()
