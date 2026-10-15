from collections import OrderedDict

import numpy as np
from itertools import chain

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.utils import transform_utils as T  
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat, mat2quat

from .samplers import UniformRandomPoseSampler
from .taco import TACOObject


def remove_prefixes(s, prefixes):
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s  # return unchanged if no prefix matche

def move_camera(env, robot_base_pos, cam_pose, cam_K, cam_body_id, cam_obj_id):

    mjc_cam_pose = np.eye(4)
    cam_rot = cam_pose[:3, :3] @ np.array([[1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])
    mjc_cam_pose[:3, :3] = cam_rot
    mjc_cam_pose[:3, 3] = robot_base_pos + cam_pose[:3, 3]

    camera_pos = mjc_cam_pose[:3, 3]
    camera_rot = mjc_cam_pose[:3, :3]

    env.sim.model.body_pos[cam_body_id] = camera_pos
    env.sim.model.body_quat[cam_body_id] = T.convert_quat(T.mat2quat(camera_rot), to='wxyz')

    intrisics = [cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]]
    env.sim.model.cam_intrinsic[cam_obj_id] = np.array(intrisics)  # not sure the correct way to set cam
    env.sim.forward()

def move_camera_v2(env, cam_pose, cam_K, cam_body_id=None, cam_name=None):
    # Always use your single camera
    cam_name = "cameraview"

    # Desired pose
    camera_pos = cam_pose[:3, 3].astype(float)
    camera_rot = cam_pose[:3, :3]@ np.array([[1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])
    camera_quat = convert_quat(mat2quat(camera_rot), to="wxyz")

    # Locate camera + owner body
    cam_id = env.sim.model.camera_name2id(cam_name)
    body_id = int(env.sim.model.cam_bodyid[cam_id])  # 0 == worldbody

    if body_id != 0:
        # Camera sits on a body → move the body
        env.sim.model.body_pos[body_id]  = camera_pos
        env.sim.model.body_quat[body_id] = camera_quat
    else:
        # Camera is directly under worldbody → set camera pose directly
        env.sim.model.cam_pos[cam_id]  = camera_pos
        env.sim.model.cam_quat[cam_id] = camera_quat

    # Apply intrinsics from retarget data.
    try:
        fx = float(cam_K[0][0])
        fy = float(cam_K[1][1])
        cx = float(cam_K[0][2])
        cy = float(cam_K[1][2])
        H = env.camera_heights if isinstance(env.camera_heights, int) else int(env.camera_heights[0])
        fovy_deg = 2.0 * np.degrees(np.arctan(0.5 * H / fy))
        env.sim.model.cam_fovy[cam_id] = fovy_deg
        if hasattr(env.sim.model, "cam_intrinsic"):
            env.sim.model.cam_intrinsic[cam_id] = np.array([fx, fy, cx, cy], dtype=np.float64)
    except Exception:
        raise Exception("Camera intrinsics or extrinsics are missing")  # ignore if anything is missing

    env.sim.forward()

class CustomTask(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        obj_info,
        table_offset=(),
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(1, 2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="cameraview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        print(camera_widths)
        print(camera_heights)
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        self.table_offset = np.array(table_offset)

        self.obj_info = obj_info

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        self.camera_names = camera_names

        # robot_model = self.robots[0].robot_model
        # xpos = self.sim.data.get_body_xpos(robot_model.root_body)
        # print("robot pos: ", xpos)

    def update_cam(self, cam_pose, cam_K):
        """
        Update cameramover body pose + intrinsics every frame.
        Assumes initialize_cam() has been called once.
        """
        move_camera_v2(
            env=self,
            cam_name=None, # [self.camera_names],
            cam_body_id=None, # self.camera_names,
            cam_K=cam_K,
            cam_pose=cam_pose,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        return 0.

    def _set_robot_config(self, robot_config):
        robot = self.robots[0]
        robot_jnames = [remove_prefixes(k, ["robot0_",]) for k in robot.robot_joints]
        robot_jpos = np.array([robot_config[k] for k in robot_jnames])

        robot.set_robot_joint_positions(robot_jpos)

        grip_jnames = []
        for side in robot.arms:
            grip_jnames += [remove_prefixes(k, ["gripper0_left_", "gripper0_right_"])
                            for k in robot.gripper_joints[side]]

        grip_jpos = np.array([robot_config[k] for k in grip_jnames])

        gripper_index = robot._ref_gripper_joint_pos_indexes
        gripper_index = list(chain.from_iterable([gripper_index[side] for side in robot.arms]))
        self.sim.data.qpos[gripper_index] = grip_jpos

        self.sim.forward()

        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        observations = self._extend_obs(observations)

        return observations

    def _get_robot_config(self, ):
        robot = self.robots[0]
        robot_jnames = [remove_prefixes(k, ["robot0_", ]) for k in robot.robot_joints]
        robot_jpos = np.array([self.sim.data.qpos[x] for x in robot._ref_joint_pos_indexes])

        config = dict(zip(robot_jnames, robot_jpos))

        for side in robot.arms:
            grip_jnames = [remove_prefixes(k, ["gripper0_left_", "gripper0_right_"])
                            for k in robot.gripper_joints[side]]
            grip_jpos = np.array([self.sim.data.qpos[x] for x in robot._ref_gripper_joint_pos_indexes[side]])
            config.update(dict(zip(grip_jnames, grip_jpos)))
        return config

    def _get_ctrl_limit(self):
        indexes = []
        act_low = []
        act_high = []
        robot = self.robots[0]

        for side_name in robot.arms:
            for act_idx in robot._ref_joint_gripper_actuator_indexes[side_name]:
                indexes.append(act_idx)
                a_low = self.sim.model.actuator_ctrlrange[act_idx, 0]
                a_high = self.sim.model.actuator_ctrlrange[act_idx, 1]
                act_low.append(a_low)
                act_high.append(a_high)

        return np.array(indexes), np.array(act_low), np.array(act_high)

    def _get_actuator_mapping(self):
        all_name2id = self.sim.model._actuator_name2id.copy()
        ret = {}
        for k, v in all_name2id.items():
            ret[remove_prefixes(k, ["robot0_",
                                    "gripper0_right_",
                                    "gripper0_left_"])] = v
        return ret

    def step(self, action):
        obs, reward, done, info = super().step(action)
        print(obs['cameraview_image'].shape)
        obs = self._extend_obs(obs)
        return obs, reward, done, info

    def _extend_obs(self, obs):
        if self.has_objects:
            hand_obj_contacts = {}
            for grip_side in self.robots[0].gripper:
                grip = self.robots[0].gripper[grip_side]
                if grip_side == "left":
                    contact = self.check_contact(grip, self.target)
                else:
                    contact = self.check_contact(grip, self.tool)
                hand_obj_contacts[grip_side + "_contact"] = contact
            obs.update(hand_obj_contacts)
        obs["robot_config"] = self._get_robot_config()
        obs["table_offset"] = self.table_offset
        return obs

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest

        if self.obj_info is not None:

            target_info = self.obj_info["target"]
            target_scale = target_info.get("scale", 0.01)
            self.target = TACOObject(xml_id=target_info["xml_id"],
                                    name="target",
                                    scale=target_scale)
            tool_info = self.obj_info["tool"]
            tool_scale = tool_info.get("scale", 0.01)
            self.tool = TACOObject(xml_id=tool_info["xml_id"],
                                    name="tool",
                                    scale=tool_scale)

            obj_poses = [target_info["pose"], tool_info["pose"]]

            objects = [self.target, self.tool]
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for i, obj in enumerate(objects):
                pose = obj_poses[i]
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomPoseSampler(
                        name=f"ObjectSampler-obj{i}",
                        mujoco_objects=obj,
                        obj_poses=[pose],
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    ),
                )

            # task includes arena, robot, and objects of interest
            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=[self.target, self.tool],
            )
            self.has_objects = True
        
        else:
            # task includes arena, robot, and objects of interest
            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=None,
            )
            self.has_objects = False

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        if self.has_objects:
            self.target_body_id = self.sim.model.body_name2id(self.target.root_body)
            self.tool_body_id = self.sim.model.body_name2id(self.tool.root_body)


    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def target_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.target_body_id])

            @sensor(modality=modality)
            def target_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.target_body_id]), to="xyzw")

            @sensor(modality=modality)
            def tool_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.tool_body_id])

            @sensor(modality=modality)
            def tool_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.tool_body_id]), to="xyzw")


            sensors = [target_pos, target_quat, tool_pos, tool_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                if not self.has_objects:
                    if name in ["target_pos", "target_body_id", "tool_pos", "tool_body_id", "target_quat", "tool_quat"]:
                        continue
                
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.has_objects:
            return
        
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                val = np.concatenate([np.array(obj_pos).flatten(), np.array(obj_quat).flatten()])
                for j in obj.joints:
                    if self.sim.data.get_joint_qpos(j).shape == (7,):
                        self.sim.data.set_joint_qpos(j, val)
                        break

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)



    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """

        # cube is higher than the table top above a margin
        return False

    def _set_object_poses(self, target_pose, tool_pose):
        def __set_pose(obj, pose):
            if obj is None:
                return
            obj_rot = pose[:3, :3]
            quat = convert_quat(mat2quat(obj_rot), "wxyz")
            pos = pose[:3, 3]

            self.sim.data.set_joint_qpos(obj.joints[0],
                                             np.concatenate([np.array(pos),
                                                             np.array(quat)]))

        if getattr(self, 'has_objects', False):
            __set_pose(getattr(self, 'tool', None), tool_pose)
            __set_pose(getattr(self, 'target', None), target_pose)


        # self.sim.data.set_joint_qvel(self.tool.joints[0], 0)
        # self.sim.data.set_joint_qvel(self.target.joints[0], 0)

        self.sim.forward()
