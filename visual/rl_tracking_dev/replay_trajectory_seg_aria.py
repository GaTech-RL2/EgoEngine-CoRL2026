#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation as R
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
np.random.seed(42)
REPLAY_BLUR_KERNEL = 5
REPLAY_NOISE_STD = 3.5
REPLAY_JPEG_QUALITY = 92
REPLAY_WARM_SHIFT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
REPLAY_MOTION_ALPHA = 0.01
REPLAY_PROPRIO_SAVE_OFFSET = np.array([0.7, 0.0, 0.0], dtype=np.float64)


def _rgb_dir_from_data_file(data_file):
    return os.path.dirname(data_file).replace("Hand_Poses", "Egocentric_RGB_Videos")


def _source_video_info(data_file):
    rgb_dir = _rgb_dir_from_data_file(data_file)
    for name in ("cropped_video.mp4", "rgb.mp4", "video_raw.mp4"):
        path = os.path.join(rgb_dir, name)
        if not os.path.isfile(path):
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if width > 0 and height > 0:
            return width, height, fps if fps > 0 else 30.0, path
    return 640, 640, 10.0, None


RIGHT_XHAND_JOINTS = [
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


def _camera_like_postprocess(frame_rgb):
    out = frame_rgb
    if REPLAY_BLUR_KERNEL > 1:
        out = cv2.GaussianBlur(out, (REPLAY_BLUR_KERNEL, REPLAY_BLUR_KERNEL), 0)
    if REPLAY_NOISE_STD > 0:
        noise = np.random.normal(0.0, REPLAY_NOISE_STD, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out = np.clip(out.astype(np.float32) + REPLAY_WARM_SHIFT[None, None, :], 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(
        ".jpg",
        cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), REPLAY_JPEG_QUALITY],
    )
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return out


def _apply_light_motion_blur(frame_rgb, prev_frame_rgb):
    if prev_frame_rgb is None or REPLAY_MOTION_ALPHA <= 0.0:
        return frame_rgb
    a = float(np.clip(REPLAY_MOTION_ALPHA, 0.0, 1.0))
    cur = frame_rgb.astype(np.float32)
    prev = prev_frame_rgb.astype(np.float32)
    blended = cur * (1.0 - a) + prev * a
    return np.clip(blended, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# Env factory — keep objects for rendering / occlusion reference only
# ---------------------------------------------------------
def make_tracking_env(args, use_cam_obs):
    env_name = "CustomTask"
    robot_name = "RBY1ArmsOnlyAria"
    controller_config = load_composite_controller_config(
        controller="RBY1_TORSO_AND_ARMS_ABSOLUTE"
    )
    print("controller:", controller_config)

    use_early_term = False
    cam_name = args.cam_name


    policy_obs_keys = [
        "robot0_joint_pos",
        "robot0_right_gripper_qpos",
        "robot0_left_gripper_qpos",
    ]

    track_budget = 3


    retarget_data = json.load(open(args.data_file, "r"))


    desk_pose = np.array(retarget_data["init_state"]["desk"])
    tool_pose = np.array(retarget_data["init_state"]["tool"]).reshape(4, 4)
    target_pose = np.array(retarget_data["init_state"]["target"]).reshape(4, 4)
    
    is_hammer_task = "hammer" in args.data_file
    # Most Aria tasks have no target object; hammer uses the fixed shelf target.
    if not is_hammer_task:
        target_pose[:3, 3] += np.array([100.0, 100.0, -100.0])

    tool2desk = np.linalg.inv(desk_pose) @ tool_pose
    target2desk = np.linalg.inv(desk_pose) @ target_pose


    obj_name = "mustard_cm"
    if "drawer" in args.data_file:
        obj_name = "cube_cm"
    elif "flower" in args.data_file:
        obj_name = "flower_cm"
    elif "hammer" in args.data_file:
        obj_name = "hammer_cm"
    elif "mustard" in args.data_file:
        obj_name = "mustard_cm"

    obj_info = {
        "target": {"xml_id": "../../aria_objects/hammer_shelf_cm" if is_hammer_task else f"../../aria_objects/{obj_name}", "pose": target2desk, "scale": 1.0},
        "tool":   {"xml_id": f"../../aria_objects/{obj_name}", "pose": tool2desk, "scale": 1.0}
    }

    render_w, render_h, render_fps, source_video = _source_video_info(args.data_file)
    args.source_video_width = render_w
    args.source_video_height = render_h
    args.source_video_fps = render_fps
    args.source_video_path = source_video
    print(
        f"[RENDER_SIZE] {render_w}x{render_h} @ {render_fps:.3f} fps"
        + (f" from {source_video}" if source_video else " (fallback)")
    )

    from .envs.rl_env_wrapper import RobosuiteFuncRetarWrapper
    def _init():
        return RobosuiteFuncRetarWrapper(
            env_name,
            robot_name,
            control_cfg=controller_config,
            obj_info=obj_info,
            obs_keys=policy_obs_keys,
            track_budget=track_budget,
            retarget_data=retarget_data,
            use_camera_obs=use_cam_obs,
            use_early_term=use_early_term,
            camera_names=cam_name,
            camera_heights=render_h,
            camera_widths=render_w,
        )
    return _init

def _maybe_flip(img, do_flip):
    # 90-degree clockwise
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if do_flip else img


def _action_from_robot_cfg(robot_cfg, actor_map):
    ordered = sorted(actor_map.items(), key=lambda kv: kv[1])
    return np.array([robot_cfg[name] for name, _ in ordered], dtype=np.float64)


def _pose7_wxyz_to_pose6(pose7):
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    pos = pose7[:3]
    quat_wxyz = pose7[3:7]
    quat_norm = np.linalg.norm(quat_wxyz)
    if quat_norm <= 1e-12:
        rotvec = np.zeros(3, dtype=np.float64)
    else:
        quat_wxyz = quat_wxyz / quat_norm
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        rotvec = R.from_quat(quat_xyzw).as_rotvec()
    return np.concatenate([pos, rotvec]).astype(np.float64)


def _matrix_to_pose6(mat):
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return np.concatenate([mat[:3, 3], R.from_matrix(mat[:3, :3]).as_rotvec()]).astype(np.float64)


def _body_pose6(sim, body_name_candidates):
    last_error = None
    for name in body_name_candidates:
        try:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 3] = np.asarray(sim.data.get_body_xpos(name), dtype=np.float64)
            mat[:3, :3] = np.asarray(sim.data.get_body_xmat(name), dtype=np.float64).reshape(3, 3)
            return _matrix_to_pose6(mat)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read body pose from candidates {body_name_candidates}: {last_error}")


def _joint_qpos(sim, joint_name):
    candidates = [
        joint_name,
        f"robot0_{joint_name}",
        f"gripper0_right_{joint_name}",
        f"robot0_gripper0_right_{joint_name}",
    ]
    last_error = None
    for name in candidates:
        try:
            qpos_addr = sim.model.get_joint_qpos_addr(name)
            return float(np.asarray(sim.data.qpos[qpos_addr]).reshape(-1)[0])
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read qpos for joint {joint_name}: {last_error}")


def _xhand_action_from_robot_cfg(robot_cfg):
    return np.array([robot_cfg[name] for name in RIGHT_XHAND_JOINTS], dtype=np.float64)


def _xhand_proprio_from_sim(sim):
    return np.array([_joint_qpos(sim, name) for name in RIGHT_XHAND_JOINTS], dtype=np.float64)


def _replay_sample_from_state(sim, dp, frame_index):
    right_arm6_proprio = _body_pose6(
        sim,
        ["link_right_arm_6", "robot0_link_right_arm_6"],
    )
    right_arm6_proprio[:3] += REPLAY_PROPRIO_SAVE_OFFSET
    return {
        "frame_index": int(frame_index),
        "right_arm6": _pose7_wxyz_to_pose6(dp["right_arm_6"]).tolist(),
        "right_arm6_proprio": right_arm6_proprio.tolist(),
        "xhand": _xhand_action_from_robot_cfg(dp["robot_cfg"]).tolist(),
        "xhand_proprio": _xhand_proprio_from_sim(sim).tolist(),
    }


def _collect_object_geom_ids(env, model):
    object_body_ids = set()
    object_geom_ids = set()
    for obj_name in ("target", "tool"):
        obj = getattr(env.env, obj_name, None)
        if obj is None:
            continue
        body_names = list(getattr(obj, "bodies", []))
        root_body = getattr(obj, "root_body", None)
        if root_body is not None:
            body_names.append(root_body)
        for body_name in body_names:
            try:
                object_body_ids.add(model.body_name2id(body_name))
            except ValueError:
                continue
        for geom_name in list(getattr(obj, "visual_geoms", [])) + list(getattr(obj, "contact_geoms", [])):
            try:
                object_geom_ids.add(model.geom_name2id(geom_name))
            except ValueError:
                continue

    object_geom_ids.update(
        gid for gid in range(model.ngeom) if model.geom_bodyid[gid] in object_body_ids
    )
    return np.asarray(sorted(object_geom_ids), dtype=np.int32)


def _model_body_name(model, body_id):
    try:
        return model.body_id2name(int(body_id)) or ""
    except Exception:
        return ""


def _model_geom_name(model, geom_id):
    try:
        return model.geom_id2name(int(geom_id)) or ""
    except Exception:
        return ""


def _is_robot_foreground_body(name):
    n = (name or "").lower()
    if not n:
        return False
    return (
        n.startswith("link_right_arm_")
        or n.startswith("link_left_arm_")
        or n in {"right", "left", "right_eef", "left_eef"}
        or "right_hand" in n
        or "left_hand" in n
        or "x_right_hand" in n
        or "x_left_hand" in n
        or "gripper0_right" in n
        or "gripper0_left" in n
    )


def _is_robot_foreground_geom(name):
    n = (name or "").lower()
    if not n:
        return False
    return (
        "right_arm" in n
        or "left_arm" in n
        or "right_hand" in n
        or "left_hand" in n
        or "x_right_hand" in n
        or "x_left_hand" in n
        or "gripper0_right" in n
        or "gripper0_left" in n
    )


def _collect_robot_foreground_geom_ids(model):
    body_ids = {
        bid
        for bid in range(model.nbody)
        if _is_robot_foreground_body(_model_body_name(model, bid))
    }

    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            parent = int(model.body_parentid[bid]) if bid < len(model.body_parentid) else -1
            if parent in body_ids and bid not in body_ids:
                body_ids.add(bid)
                changed = True

    geom_ids = {
        gid
        for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) in body_ids
        or _is_robot_foreground_geom(_model_geom_name(model, gid))
    }
    return np.asarray(sorted(geom_ids), dtype=np.int32)


def _body_pose4(sim, body_id):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    pose[:3, :3] = np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    return pose


def _capture_fixed_object_poses(env):
    fixed = {}
    sim = env.env.sim
    for obj_name in ("target", "tool"):
        body_id = getattr(env.env, f"{obj_name}_body_id", None)
        obj = getattr(env.env, obj_name, None)
        if body_id is None and obj is not None:
            body_names = list(getattr(obj, "bodies", []))
            root_body = getattr(obj, "root_body", None)
            if root_body is not None:
                body_names.insert(0, root_body)
            for body_name in body_names:
                try:
                    body_id = sim.model.body_name2id(body_name)
                    break
                except ValueError:
                    continue
        if body_id is not None:
            fixed[obj_name] = _body_pose4(sim, int(body_id))
    return fixed


def _zero_object_velocities(env):
    sim = env.env.sim
    for obj_name in ("target", "tool"):
        obj = getattr(env.env, obj_name, None)
        if obj is None:
            continue
        for joint_name in getattr(obj, "joints", []):
            try:
                addr = sim.model.get_joint_qvel_addr(joint_name)
            except Exception:
                continue
            if isinstance(addr, tuple):
                sim.data.qvel[addr[0]:addr[1]] = 0.0
            else:
                sim.data.qvel[addr] = 0.0


def _reset_objects_to_fixed_pose(env, fixed_object_poses):
    if not fixed_object_poses or not hasattr(env.env, "_set_object_poses"):
        return
    target_pose = fixed_object_poses.get("target")
    tool_pose = fixed_object_poses.get("tool")
    if target_pose is None or tool_pose is None:
        return
    env.env._set_object_poses(target_pose=target_pose, tool_pose=tool_pose)
    _zero_object_velocities(env)
    env.env.sim.forward()


def _set_tool_pose_from_retarget(env, fixed_object_poses, sample):
    if not fixed_object_poses or not hasattr(env.env, "_set_object_poses") or "tool" not in sample:
        return
    target_pose = fixed_object_poses.get("target")
    if target_pose is None:
        return
    tool_pose = np.asarray(sample["tool"], dtype=np.float64).reshape(4, 4).copy()
    tool_pose[:3, 3] += np.array([-0.7, 0.0, 0.0], dtype=np.float64)
    env.env._set_object_poses(target_pose=target_pose, tool_pose=tool_pose)
    _zero_object_velocities(env)
    env.env.sim.forward()


def _set_object_visibility(model, object_geom_ids, object_rgba, visible):
    if object_geom_ids.size == 0:
        return
    if visible:
        model.geom_rgba[object_geom_ids] = object_rgba
    else:
        model.geom_rgba[object_geom_ids, 3] = 0.0


def _disable_object_contacts(model, object_geom_ids):
    if object_geom_ids.size == 0:
        return
    if hasattr(model, "geom_contype"):
        model.geom_contype[object_geom_ids] = 0
    if hasattr(model, "geom_conaffinity"):
        model.geom_conaffinity[object_geom_ids] = 0

# ---------------------------------------------------------

# ---------------------------------------------------------
def replaying(args):
    # 1) Load data & create env
    retarget_data = json.load(open(args.data_file, "r"))
    traj = retarget_data["traj"]
    env = make_tracking_env(args, use_cam_obs=True)()

    warm_up_steps = retarget_data.get("warm_up_steps", 0)

    target2cam = [np.array(m) for m in retarget_data["camera_traj"]["target2camera"]]
    K = retarget_data["camera_traj"]["cam_K"]

    # 2) Reset once to get frame size
    _ = env.reset()

    m = env.env.sim.model
    # body ids used only for the original diff-mask visualization path
    right_body_ids = list(range(13, 20)) + [20, 21] + list(range(22, 36))
    left_body_ids  = list(range(36, 43)) + [43, 44] + list(range(45, 59)) + [59, 60, 61]
    both_arm_body_ids = right_body_ids + left_body_ids
    arm_geom_ids = [gid for gid in range(m.ngeom) if m.geom_bodyid[gid] in both_arm_body_ids]
    object_geom_ids = _collect_object_geom_ids(env, m)
    object_rgba = m.geom_rgba[object_geom_ids].copy() if object_geom_ids.size else np.empty((0, 4))
    fixed_object_poses = _capture_fixed_object_poses(env)
    _disable_object_contacts(m, object_geom_ids)
    _reset_objects_to_fixed_pose(env, fixed_object_poses)
    _set_object_visibility(m, object_geom_ids, object_rgba, visible=args.occlusion)
    env.env.sim.forward()

    frame0 = env.render()
    H, W = frame0.shape[:2]

    # 3) Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_dir = _rgb_dir_from_data_file(args.data_file)
    os.makedirs(video_dir, exist_ok=True)

    video_path = os.path.join(video_dir, "video.mp4")
    video_fps = float(getattr(args, "source_video_fps", 10.0) or 10.0)
    writer = cv2.VideoWriter(video_path, fourcc, video_fps, (W, H))
    preview_window = "replay_trajectory_seg_aria"

    # 4) Replay
    actor_map = env._get_actuator_mapping()
    avg_error = []
    mask_writer = None
    mask_bin_writer = None
    prev_frame_full = None
    replay_records = []
    replay_indices = range(warm_up_steps, len(traj))
    iterable = tqdm(replay_indices, desc="Replaying", unit="frame") if tqdm else replay_indices
    for i in iterable:
        dp = traj[i]

        time.sleep(0.1)

        robot_cfg = dp["robot_cfg"]
        cam_pose = np.array(target2cam[i], dtype=np.float64)
        cam_pose[0, 3] -= 0.7

        K_use = np.array(K, dtype=np.float64).copy()
        src_w = float(getattr(args, "source_video_width", W) or W)
        src_h = float(getattr(args, "source_video_height", H) or H)
        K_use[0, 0] *= float(W) / src_w
        K_use[1, 1] *= float(H) / src_h
        K_use[0, 2] *= float(W) / src_w
        K_use[1, 2] *= float(H) / src_h

        env.env.update_cam(cam_pose, K_use)

        if args.reset_replay:
            env._set_full_config(robot_cfg)
        else:
            action = _action_from_robot_cfg(robot_cfg, actor_map)
            obs_array, *_ = env.step(action)
            actual = obs_array[: len(action)]
            avg_error.append(float(np.linalg.norm(actual - action)))
        _set_tool_pose_from_retarget(env, fixed_object_poses, dp)
        _set_object_visibility(m, object_geom_ids, object_rgba, visible=args.occlusion)
        env.env.sim.forward()

        if "right_arm_6" in dp and "robot_cfg" in dp:
            replay_records.append(_replay_sample_from_state(env.env.sim, dp, i))

        frame_full = env.render()
        frame_motion = _apply_light_motion_blur(frame_full, prev_frame_full)
        prev_frame_full = frame_full.copy()
        frame_full_out = _camera_like_postprocess(frame_motion)
        writer.write(cv2.cvtColor(_maybe_flip(frame_full_out, args.flip), cv2.COLOR_RGB2BGR))

        if mask_writer is None:
            mask_path = os.path.join(video_dir, "mask.mp4")
            mask_writer = cv2.VideoWriter(mask_path, fourcc, video_fps, (W, H))
        if mask_bin_writer is None:
            mask_bin_path = os.path.join(video_dir, "mask_bin.mp4")
            mask_bin_writer = cv2.VideoWriter(mask_bin_path, fourcc, video_fps, (W, H))


        # --- Background-only rendering for mask ---
        orig_rgba_step  = m.geom_rgba.copy()
        orig_matid_step = m.geom_matid.copy()
        idx = np.asarray(arm_geom_ids, dtype=np.int32)
        try:
            m.geom_matid[:] = -1
            tmp = orig_rgba_step.copy()

            # 1. Background (arms hidden)
            tmp[idx, 3] = 0.0
            if not args.occlusion and object_geom_ids.size > 0:
                tmp[object_geom_ids, 3] = 0.0

            m.geom_rgba[:] = tmp
            env.env.sim.forward()
            bg_rgb = env.render()

            # 2. Arms visible
            tmp[idx, :] = np.array([1.0, 0.0, 1.0, 1.0])
            if not args.occlusion and object_geom_ids.size > 0:
                tmp[object_geom_ids, 3] = 0.0

            m.geom_rgba[:] = tmp
            env.env.sim.forward()
            arms_rgb = env.render()
        finally:
            m.geom_rgba[:] = orig_rgba_step
            m.geom_matid[:] = orig_matid_step
            _set_tool_pose_from_retarget(env, fixed_object_poses, dp)
            _set_object_visibility(m, object_geom_ids, object_rgba, visible=args.occlusion)
            env.env.sim.forward()

        # Generate mask from render difference.
        diff = np.abs(arms_rgb.astype(np.int16) - bg_rgb.astype(np.int16))
        mask_bool = (diff.max(axis=2) > 2)
        mask_u8 = (mask_bool.astype(np.uint8) * 255)

        masked_rgb = (frame_motion * (mask_u8[..., None] // 255)).astype(np.uint8)
        mask_writer.write(
            cv2.cvtColor(_maybe_flip(np.ascontiguousarray(masked_rgb), args.flip), cv2.COLOR_RGB2BGR)
        )
        mask_bin_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        mask_bin_writer.write(_maybe_flip(np.ascontiguousarray(mask_bin_rgb), args.flip))

        if args.visualize:
            preview_rgb = _maybe_flip(np.ascontiguousarray(frame_motion), args.flip)
            preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                preview_bgr,
                f"frame {i + 1}/{len(traj)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(preview_window, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[INFO] Visualization interrupted by user.")
                break

    if args.reset_replay:
        print("replay mode: reset-style (_set_full_config)")
    else:
        mean_error = float(np.mean(avg_error)) if avg_error else 0.0
        print(f"replay mode: controller stepping enabled (mean action error: {mean_error:.6f})")

    # 5) Clean up
    writer.release()
    if mask_writer is not None:
        mask_writer.release()
    if mask_bin_writer is not None:
        mask_bin_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()
    env.close()

    if replay_records:
        replay_json_path = os.path.join(os.path.dirname(args.data_file), "replay.json")
        replay_payload = {
            "source_retarget": os.path.abspath(args.data_file),
            "video_dir": os.path.abspath(video_dir),
            "warm_up_steps": int(warm_up_steps),
            "num_frames": int(len(replay_records)),
            "fields": {
                "right_arm6": "command pose for link_right_arm_6, [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]",
                "right_arm6_proprio": "replayed MuJoCo body pose for link_right_arm_6, [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]",
                "xhand": f"right XHand command qpos, joints={RIGHT_XHAND_JOINTS}",
                "xhand_proprio": f"replayed right XHand qpos, joints={RIGHT_XHAND_JOINTS}",
            },
            "right_xhand_joints": RIGHT_XHAND_JOINTS,
            "traj": replay_records,
        }
        with open(replay_json_path, "w") as f:
            json.dump(replay_payload, f, indent=2)
        print(f"[REPLAY_JSON] Saved to {replay_json_path}")
    else:
        print("[REPLAY_JSON] skipped: retarget samples do not include right_arm_6 / xhand metadata")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_file",
        type=str,
        default="/home/ycl/projects/datasets/egoengine/Aria/mustard/data/Hand_Poses/901/1/retarget.json"
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=int(6e6))
    parser.add_argument("--ckpt_freq", type=int, default=int(1e5))
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--rl_algm", type=str, default="sac")
    parser.add_argument("--log_video_freq", type=int, default=int(1e5))
    parser.add_argument("--cam_name", type=str, default="cameraview")
    parser.add_argument("--video_len", type=int, default=600)
    parser.add_argument("--flip", action="store_true", help="Save outputs rotated 90° clockwise")
    parser.add_argument("--visualize", action="store_true", help="Show a live OpenCV preview while replaying")
    parser.add_argument("--reset-replay", action="store_true", help="Use direct config reset replay instead of controller stepping")
    parser.add_argument("--occlusion", action="store_true", default=False, help="Render object as visible occluder")
    parser.add_argument("--no-occlusion", action="store_false", dest="occlusion", help="Disable object occlusion in masks")
    args = parser.parse_args()

    # debug knobs
    args.total_timesteps = 10
    args.log_video_freq = 5
    args.ckpt_freq = 5

    save_dir = f"{args.save_dir}/{args.rl_algm}-{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/config.txt", "w+") as f:
        f.writelines(str(args))

    replaying(args)
