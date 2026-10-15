#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal 3D visualization of right hand (wrist + fingertips), tag pose, and camera pose
in the world frame, and a retarget-to-robot pipeline, PLUS a batch mode (`retarget_all`)
that scans all demos under /mnt/data2/dexmimic/datasets/egoengine/Aria/data and runs
retargeting for each ID/session it finds.

Quick batch run (save into targeted Hand_Poses/{ID}/{sess}/retarget.json):
  python -m retargeting.show_aria_new retarget_all \
    --data-root /mnt/data2/dexmimic/datasets/egoengine/Aria/data \
    --tag-world /mnt/data2/dexmimic/workspace/egoengine/retargeting/tag_in_world.txt \
    --front-axis X --first-cam-height 1.4

Optional: visualize every run
  ... --visualize

Optional: save into a central results dir instead of targeted folder
  ... --out-mode results_root --out-root /mnt/data2/dexmimic/datasets/egoengine/Aria/retarget_results

Notes
-----
- Coordinate tweak: hand keypoints in camera frame are rotated +90° around Z (to match your viewer).
- Wrist/tip frames follow your axis construction; wrist (and optionally tips) right-multiply WRIST_ALIGN_R.
"""

import os
import json
import argparse
from typing import Tuple, List, Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------- Static config for names/colors ----------------
FINGER_ORDER  = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLORS = {"thumb":"red", "index":"green", "middle":"blue", "ring":"yellow", "pinky":"black"}
WRIST_COLOR   = "gray"

ARIA = {
    "wrist": 5,
    "thumb_prev": 7,   # thumb: 6,7 before tip(0)
    "index_prev": 10,  # index: 8,9,10 before tip(1)
    "middle_prev": 13, # middle: 11,12,13 before tip(2)
    "ring_prev": 16,   # ring: 14,15,16 before tip(3)
    "pinky_prev": 19,  # pinky: 17,18,19 before tip(4)
    "index1": 8, "pinky1": 17,
    "index3": 10, "pinky3": 19,
    "tips": [0,1,2,3,4],
}
TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

# ====== Relative imports to your repo layout (unchanged) ======
from .pb_visualizer import PybulletVisualizer, Entity
from .rby1_kinematics import MinkSolver, Target

# ---------------- I/O helpers ----------------
def load_w2cam(path: str) -> np.ndarray:
    w2cam = np.load(path)  # (T,4,4)
    assert w2cam.ndim == 3 and w2cam.shape[1:] == (4,4)
    return w2cam.astype(np.float32)

def load_hand_points(path: str) -> np.ndarray:
    """Return (T,21,3) right hand keypoints in camera frame."""
    with open(path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    T = len(frames)
    pts = np.full((T, 21, 3), np.nan, dtype=np.float32)
    for t, fr in enumerate(frames):
        arr = fr.get("right_camera") or fr.get("right_device")
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape == (21,3):
            pts[t] = arr
    return pts

def load_tag_T(path: str) -> np.ndarray:
    """Load 4x4 tag pose in world frame from a plain text file (4 rows)."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    T = np.array(rows, dtype=np.float32)
    assert T.shape == (4,4), f"Expected 4x4, got {T.shape}"
    return T

# ---------------- Math helpers ----------------
def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply batched 4x4 transform(s) T to points P."""
    if T.ndim == 3:
        ones = np.ones((P.shape[0], P.shape[1], 1), dtype=P.dtype)
        Ph = np.concatenate([P, ones], axis=-1)  # (T,N,4)
        out = np.einsum("tij,tkj->tki", T, Ph)  # (T,N,4)
        return out[..., :3]
    else:
        ones = np.ones((P.shape[0], 1), dtype=P.dtype)
        Ph = np.concatenate([P, ones], axis=-1)  # (N,4)
        out = (T @ Ph.T).T  # (N,4)
        return out[:, :3]

def make_axes_segments(T: np.ndarray, length: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 3 segments (each 2x3) for X(red), Y(green), Z(blue) axes of transform T (4x4)."""
    R = T[:3, :3]; t = T[:3, 3]
    X = np.stack([t, t + R[:, 0] * length], axis=0)
    Y = np.stack([t, t + R[:, 1] * length], axis=0)
    Z = np.stack([t, t + R[:, 2] * length], axis=0)
    return X, Y, Z

def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

def _align_world_yaw_to_front(cam2w: np.ndarray, hand_w: np.ndarray, front_axis: str = "X"):
    """Rotate world about Z so camera forward at t=0 points to +X (or +Y)."""
    assert front_axis.upper() in ("X", "Y")
    tgt = np.array([1.0, 0.0, 0.0], dtype=np.float32) if front_axis.upper() == "X" \
          else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd0 = cam2w[0, :3, 2].copy()
    fwd0[2] = 0.0
    if np.linalg.norm(fwd0) < 1e-8:
        return cam2w, hand_w
    a = np.arctan2(tgt[1], tgt[0]); b = np.arctan2(fwd0[1], fwd0[0])
    Rz = _rot_z(a - b)
    cam2w_rot = cam2w.copy()
    cam2w_rot[:, :3, :3] = np.einsum("ij,tjk->tik", Rz, cam2w[:, :3, :3])
    cam2w_rot[:, :3, 3]  = np.einsum("ij,tj->ti",    Rz, cam2w[:, :3, 3])
    hand_w_rot = np.einsum("ij,tnj->tni", Rz, hand_w)
    return cam2w_rot, hand_w_rot

def _shift_world_to_height(cam2w: np.ndarray, hand_w: np.ndarray, height: float):
    delta = np.array([0.0, 0.0, height], dtype=np.float32) - cam2w[0, :3, 3]
    cam2w_s = cam2w.copy()
    cam2w_s[:, :3, 3] += delta
    hand_w_s = hand_w + delta[None, None, :]
    return cam2w_s, hand_w_s

def _plane_normal(a, b, c, eps=1e-12):
    v1 = b - a
    v2 = c - a
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < eps:
        raise ValueError("Hand points are collinear; wrist normal undefined.")
    return n / norm

# Align local axes (right-multiply)
WRIST_ALIGN_R = np.array([[0, -1,  0],
                          [1,  0,  0],
                          [0,  0,  1]], dtype=np.float32)

# ---------------- Hand pose builders ----------------
def _build_hand_poses_from_points_21(pts21: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Given (21,3) points in *camera* frame, build 4x4 SE(3) for wrist and five tips.
    Rotation frames right-multiply WRIST_ALIGN_R (acts in local coords).
    """
    wrist_idx = ARIA["wrist"]
    hand_back_idx = ARIA["pinky3"]
    finger_root_idx = [ARIA["index1"], ARIA["pinky1"]]
    tip_to_last = {
        "thumb_tip":  (0, 7),
        "index_tip":  (1, 10),
        "middle_tip": (2, 13),
        "ring_tip":   (3, 16),
        "pinky_tip":  (4, 19),
    }

    a = pts21[wrist_idx]                      # wrist
    b = pts21[hand_back_idx]                  # back
    c = np.mean(pts21[finger_root_idx], axis=0)

    # Wrist axes (camera frame)
    wrist_y = _plane_normal(a, b, c)          # palm normal
    wrist_z = a - b; wrist_z /= (np.linalg.norm(wrist_z) + 1e-10)
    wrist_x = np.cross(wrist_y, wrist_z)
    if np.linalg.norm(wrist_x) < 1e-10:
        wrist_x = np.cross(wrist_z, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(wrist_x) < 1e-10:
            wrist_x = np.cross(wrist_z, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    wrist_x /= (np.linalg.norm(wrist_x) + 1e-10)

    Tw = np.eye(4, dtype=np.float32)
    Tw[:3, 0] = wrist_x
    Tw[:3, 1] = wrist_y
    Tw[:3, 2] = wrist_z
    Tw[:3, 3] = a
    Tw[:3, :3] = (Tw[:3, :3] @ WRIST_ALIGN_R).astype(np.float32)

    out = {"wrist": Tw}
    wrist_y_aligned = Tw[:3, 1]  # use aligned y for tips

    # Tips
    for name, (tip_id, last_id) in tip_to_last.items():
        if name != "thumb_tip":
            z = pts21[last_id] - pts21[tip_id]
            z /= (np.linalg.norm(z) + 1e-10)
            y = wrist_y_aligned.copy()
            x = np.cross(y, z)
            if np.linalg.norm(x) < 1e-10:
                x = np.cross(y, np.array([1.0, 0.0, 0.0], dtype=np.float32))
                if np.linalg.norm(x) < 1e-10:
                    x = np.cross(y, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            x /= (np.linalg.norm(x) + 1e-10)
        else:
            # Thumb: x from last->tip, y from palm normal, z = -y×x (right-handed)
            x = pts21[tip_id] - pts21[last_id]
            x /= (np.linalg.norm(x) + 1e-10)
            y = wrist_y_aligned.copy()
            z = -1.0 * np.cross(y, x)
            if np.linalg.norm(z) < 1e-10:
                z = np.cross(x, y)
            z /= (np.linalg.norm(z) + 1e-10)

        Tt = np.eye(4, dtype=np.float32)
        Tt[:3, 0] = x
        Tt[:3, 1] = y
        Tt[:3, 2] = z
        Tt[:3, 3] = pts21[tip_id]
        out[name] = Tt

    return out

def _transform_pose_seq_cam2world(seq_cam: List[Dict[str, np.ndarray]], cam2w: np.ndarray) -> List[Dict[str, np.ndarray]]:
    Tn = min(len(seq_cam), cam2w.shape[0])
    out = []
    for t in range(Tn):
        c2w = cam2w[t]
        out.append({k: (c2w @ Tc).astype(np.float32) for k, Tc in seq_cam[t].items()})
    return out

def _apply_yaw_and_shift_to_seq(seq_w: List[Dict[str, np.ndarray]], cam2w: np.ndarray,
                                front_axis: str, height: float):
    Tn = len(seq_w)
    hand_w = np.zeros((Tn, 6, 3), dtype=np.float32)
    order = ["wrist"] + TIP_NAMES
    for t in range(Tn):
        for i, n in enumerate(order):
            hand_w[t, i] = seq_w[t][n][:3, 3]
    cam2w_arr = np.stack([cam2w[t] for t in range(Tn)], axis=0)

    cam2w_rot, hand_w_rot = _align_world_yaw_to_front(cam2w_arr, hand_w, front_axis)
    cam2w_viz, hand_w_viz = _shift_world_to_height(cam2w_rot, hand_w_rot, height)

    seq_viz = []
    for t in range(Tn):
        T_delta = cam2w_viz[t] @ np.linalg.inv(cam2w[t])
        dv = {}
        for i, n in enumerate(order):
            Tw = seq_w[t][n]
            Tn_ = (T_delta @ Tw).astype(np.float32)
            Tn_[:3, 3] = hand_w_viz[t, i]
            dv[n] = Tn_
        seq_viz.append(dv)
    return seq_viz, cam2w_viz

def _zero_out_joints(q_dict: Dict[str, float], names: List[str]):
    for n in names:
        if n in q_dict:
            q_dict[n] = 0.0

# ---------------- GIF visualization ----------------
def make_world_gif(w2cam_path: str,
                   hand_json_path: str,
                   tag_world_txt: str,
                   out_gif: str):
    # Load and map to world
    w2cam = load_w2cam(w2cam_path)
    cam2w = np.linalg.inv(w2cam).astype(np.float32)
    hand_cam = load_hand_points(hand_json_path)        # (T,21,3) in camera
    tag_T = load_tag_T(tag_world_txt)                  # (4,4)

    T = min(len(cam2w), len(hand_cam))
    cam2w = cam2w[:T]
    hand_cam = hand_cam[:T]

    # Rotate hand in cam frame by +90° around Z
    theta = np.pi / 2
    Rz = _rot_z(theta)
    hand_cam = hand_cam @ Rz.T

    # Keep wrist + tips, transform to world
    keep_ids = [ARIA["wrist"]] + ARIA["tips"]
    hand6_cam = hand_cam[:, keep_ids, :]
    hand6_w = transform_points(cam2w, hand6_cam)

    # Camera origins, tag axes
    cam_origins = cam2w[:, :3, 3]
    tag_origin = tag_T[:3, 3]
    tag_axes = make_axes_segments(tag_T, length=0.15)

    # Axis limits
    stack = np.vstack([hand6_w.reshape(-1, 3), cam_origins, tag_origin[None, :]])
    mn = np.nanmin(stack, axis=0)
    mx = np.nanmax(stack, axis=0)
    pad = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    lim_lo = mn - pad
    lim_hi = mx + pad

    # Figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(lim_lo[0], lim_hi[0]); ax.set_ylim(lim_lo[1], lim_hi[1]); ax.set_zlim(lim_lo[2], lim_hi[2])

    # Static tag
    (tag_X, tag_Y, tag_Z) = tag_axes
    tag_x_ln, = ax.plot(tag_X[:,0], tag_X[:,1], tag_X[:,2], c='r', lw=2, label='Tag X')
    tag_y_ln, = ax.plot(tag_Y[:,0], tag_Y[:,1], tag_Y[:,2], c='g', lw=2, label='Tag Y')
    tag_z_ln, = ax.plot(tag_Z[:,0], tag_Z[:,1], tag_Z[:,2], c='b', lw=2, label='Tag Z')
    tag_pt = ax.scatter([tag_origin[0]],[tag_origin[1]],[tag_origin[2]], c='k', s=40)
    tag_txt = ax.text(tag_origin[0], tag_origin[1], tag_origin[2], 'tag', color='k')

    # Dynamic cam + hand
    cam_x_ln, = ax.plot([], [], [], c='r', lw=2)
    cam_y_ln, = ax.plot([], [], [], c='g', lw=2)
    cam_z_ln, = ax.plot([], [], [], c='b', lw=2)
    cam_pt = ax.scatter([0],[0],[0], c='crimson', marker='^', s=60)
    cam_txt = ax.text(0,0,0, 'cam', color='crimson')

    hand_colors = [WRIST_COLOR] + [FINGER_COLORS[f] for f in FINGER_ORDER]
    hand_sc = ax.scatter(np.zeros(6), np.zeros(6), np.zeros(6), c=hand_colors, s=35)
    limb_lines = [ax.plot([], [], [], c=FINGER_COLORS[f], lw=2)[0] for f in FINGER_ORDER]
    name_txts = [ax.text(0,0,0, 'wrist', color=WRIST_COLOR)] + [
        ax.text(0,0,0, f, color=FINGER_COLORS[f]) for f in FINGER_ORDER
    ]

    def update(f):
        Tcw = cam2w[f]
        X, Y, Z = make_axes_segments(Tcw, length=0.2)
        cam_x_ln.set_data(X[:,0], X[:,1]); cam_x_ln.set_3d_properties(X[:,2])
        cam_y_ln.set_data(Y[:,0], Y[:,1]); cam_y_ln.set_3d_properties(Y[:,2])
        cam_z_ln.set_data(Z[:,0], Z[:,1]); cam_z_ln.set_3d_properties(Z[:,2])

        c = Tcw[:3, 3]
        cam_pt._offsets3d = ([c[0]],[c[1]],[c[2]])
        cam_txt.set_position((c[0], c[1])); cam_txt.set_3d_properties(c[2]); cam_txt.set_text('cam')

        pts = hand6_w[f]  # (6,3): [wrist, thumb, index, middle, ring, pinky]
        hand_sc._offsets3d = (pts[:,0], pts[:,1], pts[:,2])

        wrist = pts[0]
        for i in range(5):
            tip = pts[i+1]
            limb_lines[i].set_data([wrist[0], tip[0]], [wrist[1], tip[1]])
            limb_lines[i].set_3d_properties([wrist[2], tip[2]])

        labels = ['wrist'] + FINGER_ORDER
        for i, txt in enumerate(name_txts):
            p = pts[i]
            txt.set_position((p[0], p[1])); txt.set_3d_properties(p[2]); txt.set_text(labels[i])

        ax.set_title(f'World frame – frame {f+1}/{len(cam2w)}')
        return (cam_x_ln, cam_y_ln, cam_z_ln, cam_pt, cam_txt, hand_sc, *limb_lines, *name_txts,
                tag_x_ln, tag_y_ln, tag_z_ln, tag_pt, tag_txt)

    ani = FuncAnimation(fig, update, frames=len(cam2w), interval=50, blit=False)
    try:
        ani.save(out_gif, writer=PillowWriter(fps=20))
        print(f"Saved GIF to {out_gif}")
    except Exception as e:
        try:
            ani.save(out_gif, writer='imagemagick', fps=20)
            print(f"Saved GIF (imagemagick) to {out_gif}")
        except Exception as e2:
            print("Failed to save GIF:", e)
            print("Fallback also failed:", e2)

# ---------------- Retargeting ----------------
def retarget_to_robot(
    w2cam_path: str,
    hand_json_path: str,
    tag_world_txt: str,
    out_path: str | None,
    visualize: bool,
    front_axis: str,
    first_cam_height: float,
    robot_urdf_path: str,
    robot_mjcf_path: str,
    desk_urdf_path: str,
    desk_xyz: Tuple[float,float,float],
) -> str:

    # 1) Load
    w2cam = load_w2cam(w2cam_path)                 # (T,4,4)
    cam2w = np.linalg.inv(w2cam).astype(np.float32)
    hand_cam = load_hand_points(hand_json_path)    # (T,21,3)
    T_tag_w = load_tag_T(tag_world_txt).astype(np.float32)  # noqa: F841

    # 1.1) Apply your 90° Z rotation on *camera-frame* hand points
    Rz = _rot_z(np.pi / 2)
    hand_cam = hand_cam @ Rz.T

    # 2) Build wrist+tips SE(3) per frame in camera frame
    Tn = min(len(cam2w), len(hand_cam))
    seq_cam: List[Dict[str, np.ndarray]] = []
    last_valid = None
    for t in range(Tn):
        pts = hand_cam[t]
        if np.isfinite(pts).all():
            built = _build_hand_poses_from_points_21(pts)
            seq_cam.append(built)
            last_valid = built
        else:
            if last_valid is not None:
                seq_cam.append(last_valid)
            else:
                # search ahead for first valid
                found = None
                for j in range(t+1, Tn):
                    if np.isfinite(hand_cam[j]).all():
                        found = _build_hand_poses_from_points_21(hand_cam[j])
                        break
                seq_cam.append(found if found is not None else _build_hand_poses_from_points_21(np.nan_to_num(pts)))

    # 3) Map to world, then yaw-align + height-shift (viz space)
    seq_world_raw = _transform_pose_seq_cam2world(seq_cam, cam2w)
    seq_world_viz, cam2w_viz = _apply_yaw_and_shift_to_seq(
        seq_world_raw, cam2w, front_axis, first_cam_height
    )

    # 4) IK setup (RB-Y1 with desk; base = identity in viz world)
    desk_pose = np.eye(4, dtype=np.float32)
    desk_pose[:3, 3] = np.array(desk_xyz, dtype=np.float32)

    entities = [
        Entity(name="robot", urdf=robot_urdf_path, pose=np.eye(4, dtype=np.float32), color=None, fix_base=True),
        Entity(name="desk",  urdf=desk_urdf_path,  pose=desk_pose, color=[0.5,0.5,0.5,0.5], fix_base=True),
    ]

    rby1_model = MinkSolver(robot_mjcf_path)
    robot_viz = PybulletVisualizer(entities)
    if visualize:
        robot_viz.run(0.5)

    hand2robot_link_mapping = {
        "thumb_tip":"{}_hand_thumb_rota_tip",
        "index_tip":"{}_hand_index_rota_tip",
        "middle_tip":"{}_hand_mid_tip",
        "ring_tip":"{}_hand_ring_tip",
        "pinky_tip":"{}_hand_pinky_tip",
        "wrist":"{}_hand_link",
    }

    left_hand_joints = ['left_hand_thumb_bend_joint','left_hand_thumb_rota_joint1','left_hand_thumb_rota_joint2',
                        'left_hand_index_bend_joint','left_hand_index_joint1','left_hand_index_joint2',
                        'left_hand_mid_joint1','left_hand_mid_joint2','left_hand_ring_joint1',
                        'left_hand_ring_joint2','left_hand_pinky_joint1','left_hand_pinky_joint2']
    right_hand_joints = ['right_hand_thumb_bend_joint','right_hand_thumb_rota_joint1','right_hand_thumb_rota_joint2',
                         'right_hand_index_bend_joint','right_hand_index_joint1','right_hand_index_joint2',
                         'right_hand_mid_joint1','right_hand_mid_joint2','right_hand_ring_joint1',
                         'right_hand_ring_joint2','right_hand_pinky_joint1','right_hand_pinky_joint2']
    rby1_joints = ["left_wheel","right_wheel",
                   "torso_0","torso_1","torso_2","torso_3","torso_4","torso_5",
                   "head_0","head_1",
                   "left_arm_0","left_arm_1","left_arm_2","left_arm_3","left_arm_4","left_arm_5","left_arm_6",
                   "right_arm_0","right_arm_1","right_arm_2","right_arm_3","right_arm_4","right_arm_5","right_arm_6"]
    init_q = np.array([0] * (len(right_hand_joints+right_hand_joints+rby1_joints)), dtype="float32")
    init_q_dict = dict(zip(left_hand_joints + right_hand_joints + rby1_joints, init_q.tolist()))
    # torso pose like in your long script
    init_q_dict["torso_1"] = 0.5
    init_q_dict["torso_2"] = -1.0
    init_q_dict["torso_3"] = 0.5

    def _retarget_one_frame(rby1_model: MinkSolver, init_q_dict, rh_targets, link_mapping):
        ik_targets = []
        wrist_pose = rh_targets["wrist"]
        for k in TIP_NAMES:
            tip_pose = rh_targets[k]
            link_name = link_mapping[k].format("right")
            # absolute
            ik_targets.append(Target(task_name=f"abs_{link_name}", relative=False,
                                     link_name=link_name, root_name="", link_pose=tip_pose))
            # relative to wrist
            rel_pose = np.linalg.inv(wrist_pose) @ tip_pose
            ik_targets.append(Target(task_name=f"rel_{link_name}", relative=True,
                                     link_name=link_name,
                                     root_name=link_mapping["wrist"].format("right"),
                                     link_pose=rel_pose))
        fixed_links = ["base",'link_torso_0','link_torso_1','link_torso_2','link_torso_3','link_torso_4','link_torso_5']
        res = rby1_model.compute_ik(q_dict=init_q_dict, ik_targets=ik_targets,
                                    fixed_links=fixed_links, solver="quadprog",
                                    num_iter=50, return_err=True)
        return res["q"]

    retarget_traj = []
    for i in range(len(seq_world_viz)):
        rh_targets = seq_world_viz[i]
        q_next = _retarget_one_frame(rby1_model, init_q_dict, rh_targets, hand2robot_link_mapping)

        # freeze left hand & left arm
        _zero_out_joints(q_next, left_hand_joints + ["left_arm_0","left_arm_1","left_arm_2","left_arm_3","left_arm_4","left_arm_5","left_arm_6"])

        if visualize:
            poses_to_show = [rh_targets["wrist"]] + [rh_targets[n] for n in TIP_NAMES]
            robot_viz.visualize_poses(poses_to_show, ax_len=0.10)
            robot_viz.set_joint_configuration(q_next)
            robot_viz.run(duration=0.1)

        init_q_dict = q_next.copy()
        retarget_traj.append({
            "robot_cfg": q_next,
            "tool": np.eye(4, dtype=np.float32).tolist(),
            "target": np.eye(4, dtype=np.float32).tolist(),
            "desk": desk_pose.tolist(),
            # base = I in viz world → record camera world pose
            "target2camera": cam2w_viz[i].tolist(),
            "intrinsic": [[0,0,0],[0,0,0],[0,0,1]],
        })

    # Save JSON (same schema)
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(hand_json_path)) or "."
        out_path = os.path.join(out_dir, "retarget_viz_world_start_t0.json")

    payload = {
        "init_state": {"robot_cfg": retarget_traj[0]["robot_cfg"],
                       "tool": np.eye(4, dtype=np.float32).tolist(),
                       "target": np.eye(4, dtype=np.float32).tolist(),
                       "desk": desk_pose.tolist(),
                       "base_pose_world": np.eye(4, dtype=np.float32).tolist()},
        "traj": retarget_traj,
        "camera_traj": {"target2camera":[step["target2camera"] for step in retarget_traj],
                        "intrinsic": retarget_traj[0]["intrinsic"]},
        "warm_up_steps": 0,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"[RETARGET] Saved to {out_path}")
    return out_path

# ---------------- Batch helpers ----------------
def _iter_numeric_dirs(root: str) -> Iterable[str]:
    if not os.path.isdir(root):
        return []
    for name in sorted(os.listdir(root)):
        if name.isdigit() and os.path.isdir(os.path.join(root, name)):
            yield name

def _iter_sessions(root_id_dir: str) -> Iterable[str]:
    """Yield session subdirs (e.g., '1', '2', ...). Only numeric subdirs considered."""
    if not os.path.isdir(root_id_dir):
        return []
    for name in sorted(os.listdir(root_id_dir)):
        if name.isdigit() and os.path.isdir(os.path.join(root_id_dir, name)):
            yield name

def retarget_all(
    data_root: str,
    tag_world_txt: str,
    out_mode: str,                 # "hand_folder" or "results_root"
    out_root: str,                 # used when out_mode == "results_root"
    visualize: bool,
    front_axis: str,
    first_cam_height: float,
    robot_urdf_path: str,
    robot_mjcf_path: str,
    desk_urdf_path: str,
    desk_xyz: Tuple[float,float,float],
):
    """Scan all IDs and sessions and run retargeting for each pair with valid inputs."""
    w2cam_root = os.path.join(data_root, "Egocentric_Camera_Parameters")
    hand_root  = os.path.join(data_root, "Hand_Poses")
    w2cam_file = "upd_egocentric_frame_extrinsic.npy"
    hand_file  = "all_pose.json"
    out_file   = "retarget.json"

    if not os.path.isfile(tag_world_txt):
        raise FileNotFoundError(f"tag-world not found: {tag_world_txt}")

    id_dirs = sorted(set(_iter_numeric_dirs(w2cam_root)) & set(_iter_numeric_dirs(hand_root)), key=lambda x:int(x))
    if not id_dirs:
        print("[BATCH] No overlapping ID dirs found.")
        return

    print(f"[BATCH] Found {len(id_dirs)} IDs: {', '.join(id_dirs)}")

    total = ok = fail = 0
    for id_str in id_dirs:
        w2_id_dir = os.path.join(w2cam_root, id_str)
        h_id_dir  = os.path.join(hand_root,  id_str)
        sess_dirs = sorted(set(_iter_sessions(w2_id_dir)) & set(_iter_sessions(h_id_dir)), key=lambda x:int(x))
        if not sess_dirs:
            print(f"[{id_str}] No common session subfolders; skipping.")
            continue

        for sess in sess_dirs:
            total += 1
            w2_path = os.path.join(w2_id_dir, sess, w2cam_file)
            hand_js = os.path.join(h_id_dir,  sess, hand_file)
            if not (os.path.isfile(w2_path) and os.path.isfile(hand_js)):
                print(f"[{id_str}/{sess}] Missing inputs; skipping.")
                fail += 1
                continue

            if out_mode == "hand_folder":
                out_path = os.path.join(h_id_dir, sess, out_file)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
            else:
                os.makedirs(out_root, exist_ok=True)
                out_path = os.path.join(out_root, f"{id_str}_{sess}.json")

            print(f"[{id_str}/{sess}] Retarget → {out_path}")
            try:
                retarget_to_robot(
                    w2cam_path=w2_path,
                    hand_json_path=hand_js,
                    tag_world_txt=tag_world_txt,
                    out_path=out_path,
                    visualize=visualize,
                    front_axis=front_axis,
                    first_cam_height=first_cam_height,
                    robot_urdf_path=robot_urdf_path,
                    robot_mjcf_path=robot_mjcf_path,
                    desk_urdf_path=desk_urdf_path,
                    desk_xyz=desk_xyz,
                )
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[{id_str}/{sess}] ERROR: {e}")

    print(f"[BATCH] Done. ok={ok}, fail={fail}, total={total}")

# ---------------- Argument parsing & entry ----------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="World-frame hand/cam/tag GIF and RB-Y1 retargeting (single or batch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Defaults
    DEF_DATA_ROOT = "/mnt/data2/dexmimic/datasets/egoengine/Aria/data"
    DEF_W2CAM = os.path.join(DEF_DATA_ROOT, "Egocentric_Camera_Parameters/201/1/upd_egocentric_frame_extrinsic.npy")
    DEF_HAND  = os.path.join(DEF_DATA_ROOT, "Hand_Poses/201/1/all_pose.json")
    DEF_TAG   = "/mnt/data2/dexmimic/workspace/egoengine/retargeting/tag_in_world.txt"

    # gif subcommand
    pg = sub.add_parser("gif", help="Generate world-frame 3D GIF")
    pg.add_argument("--w2cam", default=DEF_W2CAM, help="Path to world->camera extrinsics .npy (T,4,4)")
    pg.add_argument("--hand-json", default=DEF_HAND, help="Path to Aria hand all_pose.json")
    pg.add_argument("--tag-world", default=DEF_TAG, help="Path to 4x4 world_T_tag txt file")
    pg.add_argument("--out-gif", default=os.path.join(os.path.dirname(__file__), "hand_cam_tag_world.gif"),
                    help="Output GIF path")

    # retarget subcommand (single demo)
    pr = sub.add_parser("retarget", help="Retarget right-hand poses to RB-Y1+XHand (single demo)")
    pr.add_argument("--w2cam", default=DEF_W2CAM, help="Path to world->camera extrinsics .npy (T,4,4)")
    pr.add_argument("--hand-json", default=DEF_HAND, help="Path to Aria hand all_pose.json")
    pr.add_argument("--tag-world", default=DEF_TAG, help="Path to 4x4 world_T_tag txt file")
    pr.add_argument("--out-json", default=None, help="Output retarget JSON (default: alongside hand_json)")
    pr.add_argument("--visualize", action="store_true", help="Enable PyBullet visualization")
    pr.add_argument("--front-axis", choices=["X","Y"], default="X", help="Align camera forward at t=0 to +X or +Y")
    pr.add_argument("--first-cam-height", type=float, default=1.4, help="Shift so cam(0).z = this height (m)")
    pr.add_argument("--desk-x", type=float, default=0.9, help="Desk X in world (m)")
    pr.add_argument("--desk-y", type=float, default=0.0, help="Desk Y in world (m)")
    pr.add_argument("--desk-z", type=float, default=0.724, help="Desk Z in world (m)")
    pr.add_argument("--robot-urdf", default="retargeting/assets/rbyxhand/model.urdf", help="Robot URDF path")
    pr.add_argument("--robot-mjcf", default="retargeting/assets/rbyxhand/model.xml", help="Robot MJCF path")
    pr.add_argument("--desk-urdf", default="retargeting/assets/desk/desk.urdf", help="Desk URDF path")

    # retarget_all subcommand (batch over all demos/sessions)
    pb = sub.add_parser("retarget_all", help="Batch retarget across all IDs and sessions under data-root")
    pb.add_argument("--data-root", default=DEF_DATA_ROOT, help="Root containing Egocentric_Camera_Parameters and Hand_Poses")
    pb.add_argument("--tag-world", default=DEF_TAG, help="Path to 4x4 world_T_tag txt file")
    pb.add_argument("--out-mode", choices=["hand_folder","results_root"], default="hand_folder",
                    help="Where to save outputs: into Hand_Poses/<ID>/<sess>/retarget.json or a central out-root.")
    pb.add_argument("--out-root", default="/mnt/data2/dexmimic/datasets/egoengine/Aria/retarget_results",
                    help="Used only when --out-mode results_root")
    pb.add_argument("--visualize", action="store_true", help="Enable PyBullet visualization for each demo")
    pb.add_argument("--front-axis", choices=["X","Y"], default="X", help="Align camera forward at t=0 to +X or +Y")
    pb.add_argument("--first-cam-height", type=float, default=1.4, help="Shift so cam(0).z = this height (m)")
    pb.add_argument("--desk-x", type=float, default=0.9, help="Desk X in world (m)")
    pb.add_argument("--desk-y", type=float, default=0.0, help="Desk Y in world (m)")
    pb.add_argument("--desk-z", type=float, default=0.724, help="Desk Z in world (m)")
    pb.add_argument("--robot-urdf", default="retargeting/assets/rbyxhand/model.urdf", help="Robot URDF path")
    pb.add_argument("--robot-mjcf", default="retargeting/assets/rbyxhand/model.xml", help="Robot MJCF path")
    pb.add_argument("--desk-urdf", default="retargeting/assets/desk/desk.urdf", help="Desk URDF path")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "gif":
        make_world_gif(
            w2cam_path=args.w2cam,
            hand_json_path=args.hand_json,
            tag_world_txt=args.tag_world,
            out_gif=args.out_gif,
        )

    elif args.cmd == "retarget":
        visualize = args.visualize
        retarget_to_robot(
            w2cam_path=args.w2cam,
            hand_json_path=args.hand_json,
            tag_world_txt=args.tag_world,
            out_path=args.out_json,
            visualize=visualize,
            front_axis=args.front_axis,
            first_cam_height=args.first_cam_height,
            robot_urdf_path=args.robot_urdf,
            robot_mjcf_path=args.robot_mjcf,
            desk_urdf_path=args.desk_urdf,
            desk_xyz=(args.desk_x, args.desk_y, args.desk_z),
        )

    elif args.cmd == "retarget_all":
        retarget_all(
            data_root=args.data_root,
            tag_world_txt=args.tag_world,
            out_mode=args.out_mode,
            out_root=args.out_root,
            visualize=args.visualize,
            front_axis=args.front_axis,
            first_cam_height=args.first_cam_height,
            robot_urdf_path=args.robot_urdf,
            robot_mjcf_path=args.robot_mjcf,
            desk_urdf_path=args.desk_urdf,
            desk_xyz=(args.desk_x, args.desk_y, args.desk_z),
        )

    else:
        parser.error("Unknown subcommand")

if __name__ == "__main__":
    main()

'''
python -m retargeting.parse_aria_data_batch_calibrated retarget_all \
  --data-root /mnt/data2/dexmimic/datasets/egoengine/Aria/data \
  --tag-world /mnt/data2/dexmimic/workspace/egoengine/retargeting/tag_in_world.txt \
  --front-axis X --first-cam-height 1.4
'''