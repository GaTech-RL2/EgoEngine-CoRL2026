"""Visual arm replacement using original robosuite assets and replay conventions.

The example changes the arm while copying the recorded XHand joints. It does
not change floating-wrist action optimization or the author's metrics.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .data import PROJECT_ROOT, get_case
from .video import video_info, writer


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != "egoengine.visual_robot.v1":
        raise ValueError("Expected egoengine.visual_robot.v1 robot configuration")
    required = (
        "name",
        "robot_module",
        "robot_class",
        "arm_joints",
        "base_position_in_desk",
        "base_quat_wxyz",
        "gripper_name",
        "wrist_body",
        "source_wrist_body",
    )
    if any(key not in config for key in required):
        raise ValueError("Robot configuration is missing required fields")
    position = np.asarray(config["base_position_in_desk"], dtype=float)
    quat = np.asarray(config["base_quat_wxyz"], dtype=float)
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[position, quat]).all():
        raise ValueError("Invalid robot base pose")
    if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
        raise ValueError("Robot base quaternion must be unit wxyz")
    if not config["arm_joints"] or len(set(config["arm_joints"])) != len(config["arm_joints"]):
        raise ValueError("Arm joint names must be nonempty and unique")
    return config


def base_pose(config, retarget):
    # Original visual replay shifts the prepared world by -0.7m on X.
    desk = np.asarray(retarget["init_state"]["desk"], dtype=float).copy()
    desk[0, 3] -= 0.7
    relative = np.eye(4)
    relative[:3, 3] = config["base_position_in_desk"]
    relative[:3, :3] = Rotation.from_quat(
        np.asarray(config["base_quat_wxyz"])[[1, 2, 3, 0]]
    ).as_matrix()
    return desk @ relative


def register_robot(config, placement, scene_lights=None):
    from robosuite.robots import register_robot_class

    parent = getattr(importlib.import_module(config["robot_module"]), config["robot_class"])

    class VisualRobot(parent):
        def __init__(self, idn=0):
            super().__init__(idn=idn)
            self.set_base_ori(Rotation.from_matrix(placement[:3, :3]).as_euler("xyz"))
            if scene_lights is not None:
                # World lights belong to the scene, independent of the replacement arm.
                for light in self.worldbody.findall("light"):
                    self.worldbody.remove(light)
                for light in scene_lights:
                    self.worldbody.append(deepcopy(light))

        @property
        def default_base(self):
            return "NullMount"

        @property
        def default_gripper(self):
            return {"right": config["gripper_name"]}

        @property
        def base_xpos_offset(self):
            return {
                "table": lambda _: placement[:3, 3].copy(),
                "empty": placement[:3, 3].copy(),
                "bins": placement[:3, 3].copy(),
            }

    register_robot_class("FixedBaseRobot")(VisualRobot)
    return VisualRobot


def lighting_state(model):
    """Record compiled scene lights independently of robot geometry and placement."""
    return {
        "headlight": {
            key: np.asarray(getattr(model.vis.headlight, key)).tolist()
            for key in ("active", "ambient", "diffuse", "specular")
        },
        "world_lights": [
            {
                key: np.asarray(getattr(model, "light_" + key)[index]).tolist()
                for key in ("pos", "dir", "ambient", "diffuse", "specular", "castshadow")
            }
            for index in range(model.nlight)
            if model.light_bodyid[index] == 0
        ],
    }


def source_visual_reference(case, original, config):
    args = SimpleNamespace(data_file=str(case.visual / "retarget.json"), cam_name="cameraview")
    env = original.make_tracking_env(args, use_cam_obs=False)()
    poses = []
    try:
        scene_lights = deepcopy(env.env.robots[0].robot_model.worldbody.findall("light"))
        lighting = lighting_state(env.env.sim.model._model)
        for frame in case.retarget_frames():
            env._set_full_config(frame["robot_cfg"])
            pose = np.eye(4)
            pose[:3, 3] = env.env.sim.data.get_body_xpos(config["source_wrist_body"])
            pose[:3, :3] = env.env.sim.data.get_body_xmat(config["source_wrist_body"]).reshape(3, 3)
            poses.append(pose)
    finally:
        env.close()
    return np.asarray(poses), scene_lights, lighting


def map_arm(config, robot_class, placement, targets, frames, hand_joints):
    """Fit only the arm; the recorded hand commands never enter the optimizer."""
    from robosuite.models.bases import robot_base_factory
    from robosuite.models.grippers import gripper_factory

    robot = robot_class(idn=0)
    robot.add_base(robot_base_factory("NullMount", idn=0))
    robot.add_gripper(gripper_factory(config["gripper_name"], idn="0_right"))
    robot.set_base_xpos(placement[:3, 3])
    model = mujoco.MjModel.from_xml_string(robot.get_xml())
    data = mujoco.MjData(model)
    arm_names = config["arm_joints"]
    joints = [model.joint("robot0_" + name) for name in arm_names]
    if any(model.jnt_type[j.id] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
        raise ValueError("This adapter expects hinge arm joints")
    addresses = [int(model.jnt_qposadr[j.id]) for j in joints]
    bounds = np.asarray([j.range for j in joints])
    hand_addresses = [
        int(model.jnt_qposadr[model.joint("gripper0_right_" + name).id]) for name in hand_joints
    ]
    wrist = model.body(config["wrist_body"]).id
    previous = np.clip(np.asarray(robot.init_qpos), bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6)
    home = previous.copy()
    rows, actuals = [], []
    for index, (target, frame) in enumerate(zip(targets, frames)):
        hand = np.asarray([frame["robot_cfg"][name] for name in hand_joints])
        data.qpos[hand_addresses] = hand

        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_kinematics(model, data)
            rotation = Rotation.from_matrix(
                target[:3, :3] @ data.xmat[wrist].reshape(3, 3).T
            ).as_rotvec()
            return np.r_[data.xpos[wrist] - target[:3, 3], rotation * 0.2, (q - previous) * 1e-5]

        best = least_squares(
            residual,
            previous,
            bounds=(bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6),
            max_nfev=int(config.get("ik_max_evaluations", 150)),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if np.linalg.norm(best.fun[:6]) > 0.003:
            alternate = least_squares(
                residual,
                home,
                bounds=(bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6),
                max_nfev=int(config.get("ik_max_evaluations", 150)),
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if np.linalg.norm(alternate.fun[:6]) < np.linalg.norm(best.fun[:6]):
                best = alternate
        previous = best.x.copy()
        residual(previous)
        actual = np.eye(4)
        actual[:3, 3] = data.xpos[wrist]
        actual[:3, :3] = data.xmat[wrist].reshape(3, 3)
        actuals.append(actual)
        rows.append(dict(zip(arm_names + list(hand_joints), np.r_[previous, hand].tolist())))
        if index % 30 == 0:
            print(f"[visual arm] {index + 1}/{len(frames)}", flush=True)
    return rows, np.asarray(actuals)


def make_environment(original, args, config, width, height):
    """Supply the robot to the existing factory in this isolated worker only."""
    from rl_tracking_dev.envs import rl_env_wrapper

    original_wrapper = rl_env_wrapper.RobosuiteFuncRetarWrapper

    class ConfiguredWrapper(original_wrapper):
        def __init__(self, env_name, robot_name, **kwargs):
            arm = deepcopy(kwargs["control_cfg"]["body_parts"]["right"])
            kwargs["control_cfg"] = {"type": "BASIC", "body_parts": {"right": arm}}
            kwargs["obs_keys"] = ["robot0_joint_pos", "robot0_right_gripper_qpos"]
            kwargs["camera_widths"], kwargs["camera_heights"] = width, height
            super().__init__(env_name, "VisualRobot", **kwargs)

    rl_env_wrapper.RobosuiteFuncRetarWrapper = ConfiguredWrapper
    try:
        return original.make_tracking_env(args, use_cam_obs=True)()
    finally:
        rl_env_wrapper.RobosuiteFuncRetarWrapper = original_wrapper


def replay_layers(env, original, args, retarget, output, width, height, fps, wrist_body):
    """Keep the original Aria camera, object, postprocessing and difference-mask steps."""
    model = env.env.sim.model
    native = model._model
    robot_root = native.body(env.env.robots[0].robot_model.root_body).id
    descendants = {robot_root}
    for body in range(native.nbody):
        if int(native.body_parentid[body]) in descendants:
            descendants.add(body)
    arm_ids = np.asarray(
        [
            i
            for i in range(native.ngeom)
            if int(native.geom_bodyid[i]) in descendants and native.geom_group[i] == 1
        ]
    )
    if not len(arm_ids):
        raise ValueError("Configured robot has no visible geometry")
    object_ids = original._collect_object_geom_ids(env, model)
    object_rgba = model.geom_rgba[object_ids].copy()
    fixed = original._capture_fixed_object_poses(env)
    original._disable_object_contacts(model, object_ids)
    paths = {
        key: output / name
        for key, name in (
            ("scene", "scene.mp4"),
            ("foreground", "foreground.mp4"),
            ("mask", "visibility_mask.mp4"),
        )
    }
    writers = {key: writer(path, width, height, fps) for key, path in paths.items()}
    previous, pixels, wrist_poses = None, [], []
    wrist_id = native.body(wrist_body).id
    start = int(retarget.get("warm_up_steps", 0))
    try:
        for index in range(start, len(retarget["traj"])):
            frame = retarget["traj"][index]
            camera = np.asarray(retarget["camera_traj"]["target2camera"][index], float).copy()
            camera[0, 3] -= 0.7
            k = np.asarray(retarget["camera_traj"]["cam_K"], float).copy()
            k[0, :] *= width / args.source_video_width
            k[1, :] *= height / args.source_video_height
            env.env.update_cam(camera, k)
            env._set_full_config(frame["robot_cfg"])
            original._set_tool_pose_from_retarget(env, fixed, frame)
            original._set_object_visibility(model, object_ids, object_rgba, visible=True)
            env.env.sim.forward()
            wrist = np.eye(4)
            wrist[:3, 3] = env.env.sim.data.body_xpos[wrist_id]
            wrist[:3, :3] = env.env.sim.data.body_xmat[wrist_id].reshape(3, 3)
            wrist_poses.append(wrist)
            rgb = env.render()
            motion = original._apply_light_motion_blur(rgb, previous)
            previous = rgb.copy()
            scene = original._camera_like_postprocess(motion)
            rgba, materials = model.geom_rgba.copy(), model.geom_matid.copy()
            try:
                model.geom_matid[:] = -1
                model.geom_rgba[arm_ids, 3] = 0
                env.env.sim.forward()
                background = env.render()
                model.geom_rgba[arm_ids] = [1, 0, 1, 1]
                env.env.sim.forward()
                foreground = env.render()
            finally:
                model.geom_rgba[:] = rgba
                model.geom_matid[:] = materials
                env.env.sim.forward()
            mask = np.abs(foreground.astype(np.int16) - background.astype(np.int16)).max(2) > 2
            pixels.append(int(mask.sum()))
            images = {
                "scene": scene,
                "foreground": motion * mask[..., None],
                "mask": np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2),
            }
            for key, image in images.items():
                writers[key].write(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        for stream in writers.values():
            stream.release()
    return pixels, np.asarray(wrist_poses)


def worker(case, config_path, output, width, height):
    if case.info["dataset"] != "Aria":
        raise ValueError("The first single-arm robot example uses the prepared Aria cases")
    sys.path.insert(0, str(PROJECT_ROOT / "visual"))
    original = importlib.import_module("rl_tracking_dev.replay_trajectory_seg_aria")
    config = load_config(config_path)
    source_path = case.visual / "retarget.json"
    retarget = json.loads(source_path.read_text())
    placement = base_pose(config, retarget)
    targets, scene_lights, source_lighting = source_visual_reference(case, original, config)
    robot_class = register_robot(config, placement, scene_lights)
    commands, actual = map_arm(
        config, robot_class, placement, targets, case.retarget_frames(), original.RIGHT_XHAND_JOINTS
    )
    position_error = np.linalg.norm(targets[:, :3, 3] - actual[:, :3, 3], axis=1)
    orientation_error = Rotation.from_matrix(
        targets[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    names = config["arm_joints"] + list(original.RIGHT_XHAND_JOINTS)
    np.savez(
        output / "robot_commands.npz",
        joint_names=np.asarray(names),
        joint_commands=np.asarray([[row[name] for name in names] for row in commands]),
        target_wrist_pose=targets,
        actual_wrist_pose=actual,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
    )
    print(
        f"[visual wrist] max_position={position_error.max():.6f}m "
        f"max_orientation={orientation_error.max():.6f}rad",
        flush=True,
    )
    start = int(retarget.get("warm_up_steps", 0))
    if len(retarget["traj"]) - start != len(commands):
        raise ValueError("Prepared retarget warmup/reference lengths do not agree")
    retarget["init_state"]["robot_cfg"] = commands[0]
    for index, frame in enumerate(retarget["traj"]):
        frame["robot_cfg"] = commands[max(0, index - start)]
    staged = output / "original_pipeline"
    data_file = staged / "Hand_Poses" / case.name / "1/retarget.json"
    rgb_dir = staged / "Egocentric_RGB_Videos" / case.name / "1"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    data_file.write_text(json.dumps(retarget))
    link = rgb_dir / "cropped_video.mp4"
    if not link.exists():
        link.symlink_to(case.visual / "cropped_video.mp4")
    info = video_info(link)
    width = width or info["width"]
    height = height or int(round(width * info["height"] / info["width"] / 2)) * 2
    args = SimpleNamespace(data_file=str(data_file), cam_name="cameraview")
    env = make_environment(original, args, config, width, height)
    try:
        lighting = lighting_state(env.env.sim.model._model)
        if lighting != source_lighting:
            raise ValueError("Replacement robot changed the original scene lighting")
        pixels, replayed = replay_layers(
            env,
            original,
            args,
            retarget,
            output,
            width,
            height,
            1 / case.ref_dt,
            config["wrist_body"],
        )
        collision_visible = bool(env.env.render_collision_mesh)
    finally:
        env.close()
    np.savez(output / "replay_states.npz", wrist_pose=replayed)
    if not np.allclose(actual, replayed, atol=2e-6):
        raise ValueError("Standalone arm mapping and rendered-scene wrist frames differ")
    hand_path = PROJECT_ROOT / "visual/robosuite/models/assets/grippers/x_right_hand_aria.xml"
    metadata = dict(
        case=case.name,
        renderer="configured_robot",
        robot_name=config["name"],
        backend="original_robosuite_environment_with_robot_adapter",
        frames=len(commands),
        fps=1 / case.ref_dt,
        retarget_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        robot_config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        xhand_xml_sha256=hashlib.sha256(hand_path.read_bytes()).hexdigest(),
        action_source="original_retarget",
        action_sha256=None,
        commands_file="robot_commands.npz",
        xhand_commands="unchanged from original retarget",
        lighting_source="original_retarget_environment",
        scene_lights=[dict(light.attrib) for light in scene_lights],
        lighting=lighting,
        base_to_world=placement.tolist(),
        render_collision_mesh=collision_visible,
        frames_with_visible_robot=sum(value > 0 for value in pixels),
        mask_pixels_min=min(pixels),
        mask_pixels_max=max(pixels),
        max_wrist_position_error_m=float(position_error.max()),
        max_wrist_orientation_error_rad=float(orientation_error.max()),
    )
    (output / "render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def render_robot(case, output, config_path, width=None, height=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    load_config(config_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(PROJECT_ROOT / "visual"), env.get("PYTHONPATH", "")]
    )
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
    env.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cache/numba"))
    command = [
        sys.executable,
        "-m",
        "egoengine.robot_visual",
        "--case",
        case.name,
        "--data-root",
        str(case.root),
        "--output",
        str(output),
        "--robot-config",
        str(Path(config_path).resolve()),
    ]
    for flag, value in (("--width", width), ("--height", height)):
        if value is not None:
            command += [flag, str(value)]
    log = output / "robot_replay.log"
    print(f"[render] {case.name}: configured robot; log: {log}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"Robot visual replay failed: {log}\n{log.read_text()[-5000:]}")
    return {
        key: output / name
        for key, name in (
            ("scene", "scene.mp4"),
            ("foreground", "foreground.mp4"),
            ("mask", "visibility_mask.mp4"),
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    worker(
        get_case(args.case, args.data_root), args.robot_config, args.output, args.width, args.height
    )
