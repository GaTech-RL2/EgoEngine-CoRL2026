#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal 3D visualization of right hand (wrist + fingertips), tag pose, and camera pose
in the world frame, and a retarget-to-robot pipeline.

Usage examples
--------------
# 1) Make GIF (paths shown are defaults; override as needed)
python script.py gif \
  --w2cam /mnt/data2/dexmimic/datasets/egoengine/Aria/data/Egocentric_Camera_Parameters/201/1/upd_egocentric_frame_extrinsic.npy \
  --hand-json /mnt/data2/dexmimic/datasets/egoengine/Aria/data/Hand_Poses/201/1/all_pose.json \
  --tag-world /mnt/data2/dexmimic/workspace/egoengine/retargeting/tag_in_world.txt \
  --out-gif ./hand_cam_tag_world.gif

# 2) Retarget (RB-Y1 + XHand), with PyBullet viz on
python script.py retarget \
  --w2cam /mnt/data2/dexmimic/datasets/egoengine/Aria/data/Egocentric_Camera_Parameters/201/1/upd_egocentric_frame_extrinsic.npy \
  --hand-json /mnt/data2/dexmimic/datasets/egoengine/Aria/data/Hand_Poses/201/1/all_pose.json \
  --tag-world /mnt/data2/dexmimic/workspace/egoengine/retargeting/tag_in_world.txt \
  --front-axis X \
  --first-cam-height 1.4 \
  --out-json /mnt/data2/dexmimic/datasets/egoengine/Aria/data/Hand_Poses/201/1/retarget_viz_world_start_t0.json \
  --visualize

Notes
-----
- This dataset's camera extrinsics are treated as already aligned with Aria convention.
- Hand keypoints are used in the Aria camera frame without the old extra +90° Z rotation.
- Wrist/tip frames follow your axis construction; wrist (and optionally tips) right-multiply WRIST_ALIGN_R.
"""

import os
import json
import argparse
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
import trimesh

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
DEFAULT_OBJECT_CENTRIC_THRESH = 0.0

# ====== Relative imports to your repo layout (unchanged) ======
from .pb_visualizer import PybulletVisualizer, Entity
from .rby1_kinematics import MinkSolver, Target


def _ensure_debug_box_urdf(path: str, size_xyz: Tuple[float, float, float] = (0.04, 0.04, 0.10)) -> str:
    """Create a simple visible box URDF for object visualization if not exists."""
    if os.path.exists(path):
        return path
    sx, sy, sz = size_xyz
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urdf = f"""<?xml version="1.0"?>
<robot name="debug_tool_box">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(urdf)
    return path


def _ensure_mesh_urdf(path: str, mesh_path: str, scale_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> str:
    """Create a URDF wrapper for a mesh file for PyBullet visualization."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sx, sy, sz = scale_xyz
    urdf = f"""<?xml version="1.0"?>
<robot name="mesh_object">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{sx} {sy} {sz}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{sx} {sy} {sz}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(urdf)
    return path

# ---------------- I/O helpers ----------------
def load_w2cam(path: str) -> np.ndarray:
    w2cam = np.load(path)  # (T,4,4)
    assert w2cam.ndim == 3 and w2cam.shape[1:] == (4,4)
    return w2cam.astype(np.float32)


def load_object_poses(path: str) -> np.ndarray:
    poses = np.load(path)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), f"Bad object pose shape: {poses.shape}"
    return poses.astype(np.float32)


def _resolve_camera_dir_from_inputs(w2cam_path: str, hand_json_path: str) -> str:
    candidates = [
        os.path.dirname(os.path.abspath(w2cam_path)),
        os.path.dirname(os.path.abspath(hand_json_path)).replace(
            os.sep + "Hand_Poses" + os.sep,
            os.sep + "Egocentric_Camera_Parameters" + os.sep,
        ),
        os.path.dirname(os.path.abspath(hand_json_path)).replace(
            os.sep + "Hand_Poses" + os.sep,
            os.sep + "Egocentric_Camera_Parameters" + os.sep + "Egocentric_Camera_Parameters" + os.sep,
        ),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "egocentric_intrinsic.txt")):
            return candidate
    return candidates[0]


TASK_NAME_CHOICES = ("mustard", "drawer", "flower", "hammer")


def _infer_task_name_from_path(path: str) -> str | None:
    for part in os.path.abspath(path).split(os.sep):
        if part in TASK_NAME_CHOICES:
            return part
        if part.startswith(tuple(f"{name}_" for name in TASK_NAME_CHOICES)):
            return part.split("_", 1)[0]
    return None


def _object_mesh_path_for_task(task_name: str | None, aria_obj_dir: str) -> str | None:
    mesh_name = {
        "mustard": "mustard.obj",
        "drawer": "cube_convex.obj",
        "flower": "windex.obj",
        "hammer": "hammer_cut_1.5x.stl",
    }.get(task_name or "")
    if mesh_name is None:
        return None
    return os.path.join(aria_obj_dir, "meshes", mesh_name)


def _pybullet_tool_urdf_for_task(task_name: str | None, aria_obj_dir: str) -> str | None:
    if task_name == "mustard":
        return os.path.join(os.path.dirname(__file__), "assets", "aria_mustard_tool.urdf")
    if task_name == "drawer":
        return _ensure_debug_box_urdf(
            os.path.join(os.path.dirname(__file__), "assets", "aria_drawer_tool.urdf"),
            size_xyz=(0.05, 0.05, 0.05),
        )
    if task_name == "flower":
        return _ensure_mesh_urdf(
            os.path.join(os.path.dirname(__file__), "assets", "aria_flower_tool.urdf"),
            os.path.join(aria_obj_dir, "meshes", "windex.obj"),
        )
    if task_name == "hammer":
        return _ensure_mesh_urdf(
            os.path.join(os.path.dirname(__file__), "assets", "aria_hammer_tool.urdf"),
            os.path.join(aria_obj_dir, "meshes", "hammer_cut_1.5x.stl"),
        )
    return None


def _pybullet_target_urdf_for_task(task_name: str | None, aria_obj_dir: str) -> str | None:
    if task_name == "hammer":
        return _ensure_mesh_urdf(
            os.path.join(os.path.dirname(__file__), "assets", "aria_hammer_shelf_target.urdf"),
            os.path.join(aria_obj_dir, "meshes", "hammer_shelf.obj"),
        )
    return None


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

def _plane_normal(a, b, c, eps=1e-12):
    v1 = b - a
    v2 = c - a
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < eps:
        raise ValueError("Hand points are collinear; wrist normal undefined.")
    return n / norm

# Align local axes (right-multiply)
WRIST_ALIGN_R = np.array([[-1, 0,  0],
                          [0,  -1,  0],
                          [0,  0,  1]], dtype=np.float32)

# Match visualize_cam_and_tag_in_ref.py:
# this dataset's camera extrinsics already align with Aria convention,
# so keep both corrections as identity to avoid double-rotation.
R_CAM_TO_ARIA = np.eye(3, dtype=np.float32)
CAM_TO_HAND_R = np.eye(3, dtype=np.float32)

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

def _transform_obj_seq_cam2world(seq_obj: np.ndarray, cam2w: np.ndarray) -> np.ndarray:
    """
    Transform object poses from camera frame to world frame.

    Args:
        seq_obj: (T, 4, 4) object poses in camera frame
        cam2w:   (T, 4, 4) camera-to-world transforms

    Returns:
        seq_obj_w: (T, 4, 4) object poses in world frame
    """
    # assert seq_obj.shape == cam2w.shape, "Shape mismatch, both must be (T, 4, 4)"
    Tn = min(len(seq_obj), cam2w.shape[0])
    seq_obj_w = np.empty_like(seq_obj, dtype=np.float32)

    for t in range(Tn):
        seq_obj_w[t] = cam2w[t] @ seq_obj[t]

    return seq_obj_w


def _weighted_average_rotations(rot_mats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    quats = R.from_matrix(rot_mats).as_quat()
    ref = quats[len(quats) // 2]
    signs = np.where(np.sum(quats * ref[None, :], axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    quats = quats * signs
    accum = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quats, weights):
        accum += float(w) * np.outer(q, q)
    _, vecs = np.linalg.eigh(accum)
    quat = vecs[:, -1]
    if np.dot(quat, ref) < 0.0:
        quat = -quat
    return R.from_quat(quat).as_matrix()


def _smooth_pose_sequence(seq_pose: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1 or len(seq_pose) <= 2:
        return seq_pose.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    base_weights = np.arange(1, radius + 2, dtype=np.float64)
    base_weights = np.concatenate([base_weights, base_weights[-2::-1]])
    out = seq_pose.astype(np.float64).copy()
    for i in range(len(seq_pose)):
        lo = max(0, i - radius)
        hi = min(len(seq_pose), i + radius + 1)
        w_lo = lo - (i - radius)
        w_hi = w_lo + (hi - lo)
        weights = base_weights[w_lo:w_hi]
        weights = weights / weights.sum()
        out[i, :3, 3] = np.sum(seq_pose[lo:hi, :3, 3] * weights[:, None], axis=0)
        out[i, :3, :3] = _weighted_average_rotations(seq_pose[lo:hi, :3, :3], weights)
        out[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return out.astype(np.float32)

def _transform_pose_seq_cam2world(seq_cam: List[Dict[str, np.ndarray]], cam2w: np.ndarray) -> List[Dict[str, np.ndarray]]:
    Tn = min(len(seq_cam), cam2w.shape[0])
    out = []
    for t in range(Tn):
        c2w = cam2w[t]
        out.append({k: (c2w @ Tc).astype(np.float32) for k, Tc in seq_cam[t].items()})
    return out

def _zero_out_joints(q_dict: Dict[str, float], names: List[str]):
    for n in names:
        if n in q_dict:
            q_dict[n] = 0.0


def _mesh_from_obj(obj_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, process=False)
    return mesh


def _transform_mesh(mesh: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    m = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces, process=False)
    m.apply_transform(T)
    return m


def _nearest_surface_point(mesh: trimesh.Trimesh, pt: np.ndarray) -> tuple[np.ndarray, float]:
    q = np.asarray(pt, dtype=np.float64).reshape(1, 3)
    closest, distance, _ = trimesh.proximity.closest_point(mesh, q)
    return np.asarray(closest[0], dtype=np.float64), float(distance[0])


def _snap_tip_to_surface(
    tip_pose: np.ndarray, mesh: trimesh.Trimesh, threshold: float
) -> tuple[np.ndarray, bool]:
    if threshold <= 0:
        return tip_pose, False
    pt = tip_pose[:3, 3]
    try:
        closest_surface_pt, min_dist = _nearest_surface_point(mesh, pt)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing package for mesh-surface distance: rtree. "
            "Install it in `egoengine` env, e.g. `pip install rtree` "
            "or `conda install -c conda-forge rtree`."
        ) from e
    if min_dist > threshold:
        return tip_pose, False
    out = tip_pose.copy()
    out[:3, 3] = closest_surface_pt
    return out, True


def _object_centric_targets(
    targets: Dict[str, np.ndarray], mesh: trimesh.Trimesh, threshold: float
) -> tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    out = {k: v.copy() for k, v in targets.items()}
    contact = {name: False for name in TIP_NAMES}
    for name in TIP_NAMES:
        out[name], contact[name] = _snap_tip_to_surface(out[name], mesh, threshold)
    return out, contact

# ---------------- GIF visualization ----------------
def make_world_gif(w2cam_path: str,
                   hand_json_path: str,
                   tag_world_txt: str,
                   out_gif: str):
    # Load and map to world
    w2cam = load_w2cam(w2cam_path)
    cam2w = np.linalg.inv(w2cam).astype(np.float32)
    cam2w[:, :3, :3] = np.einsum("tij,jk->tik", cam2w[:, :3, :3], R_CAM_TO_ARIA)
    hand_cam = load_hand_points(hand_json_path)        # (T,21,3) in camera
    tag_T = load_tag_T(tag_world_txt)                  # (4,4)

    T = min(len(cam2w), len(hand_cam))
    cam2w = cam2w[:T]
    hand_cam = hand_cam[:T]

    # Keep Aria camera-frame hand points as-is unless CAM_TO_HAND_R is changed.
    hand_cam = hand_cam @ CAM_TO_HAND_R.T

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
    object_centric: float,
    task_name: str | None,
    start_frame: int,
    # first_cam_height: float,
    robot_urdf_path: str,
    robot_mjcf_path: str,
    desk_urdf_path: str,
    desk_xyz: Tuple[float,float,float],
    target_pose_path: str | None = None,
) -> str:
    if task_name is not None and task_name not in TASK_NAME_CHOICES:
        raise ValueError(f"task_name must be one of {TASK_NAME_CHOICES}, got {task_name!r}")

    # 1) Load
    w2cam = load_w2cam(w2cam_path)                 # (T,4,4)
    cam2w = np.linalg.inv(w2cam).astype(np.float32)
    cam2w[:, :3, :3] = np.einsum("tij,jk->tik", cam2w[:, :3, :3], R_CAM_TO_ARIA)
    cam_dir = _resolve_camera_dir_from_inputs(w2cam_path, hand_json_path)
    obj_path = os.path.join(
        os.path.dirname(os.path.abspath(hand_json_path)).replace(
            os.sep + "Hand_Poses" + os.sep,
            os.sep + "Object_Poses" + os.sep,
        ),
        "object.npy",
    )
    cam_K = np.loadtxt(os.path.join(cam_dir, "egocentric_intrinsic.txt")).astype(np.float32)
    hand_cam = load_hand_points(hand_json_path)    # (T,21,3)
    obj_cam = load_object_poses(obj_path)          # (T,4,4)

    # 1.1) Keep hand points in the Aria camera frame by default.
    hand_cam = hand_cam @ CAM_TO_HAND_R.T

    start_frame = max(0, int(start_frame))
    if start_frame >= len(cam2w) or start_frame >= len(hand_cam) or start_frame >= len(obj_cam):
        raise ValueError(
            f"start_frame={start_frame} is out of range for cam2w={len(cam2w)}, hand={len(hand_cam)}, obj={len(obj_cam)}"
        )
    if start_frame > 0:
        cam2w = cam2w[start_frame:]
        hand_cam = hand_cam[start_frame:]
        obj_cam = obj_cam[start_frame:]

    # 2) Build wrist+tips SE(3) per frame in camera frame
    Tn = min(len(cam2w), len(hand_cam), len(obj_cam))
    obj_cam = obj_cam[:Tn]
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
    seq_world = _transform_pose_seq_cam2world(seq_cam, cam2w)
    seq_obj_world = _transform_obj_seq_cam2world(obj_cam, cam2w)
    seq_obj_world = _smooth_pose_sequence(seq_obj_world, window=7)
    # Object-centric refinement: per-fingertip snap to closest mesh-surface point.
    task_name = task_name or _infer_task_name_from_path(hand_json_path)
    # Preserve measured object motion; task names select assets only.
    aria_obj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_tracking_dev/aria_objects")
    object_mesh_path = _object_mesh_path_for_task(task_name, aria_obj_dir)
    if object_centric and object_centric > 0:
        if object_mesh_path is None:
            print(f"[WARN] object-centric snapping is not configured for task={task_name}; skipping")
        else:
            mesh_base = _mesh_from_obj(object_mesh_path)
            for i in range(len(seq_world)):
                mesh_w = _transform_mesh(mesh_base, seq_obj_world[i])
                seq_world[i], _ = _object_centric_targets(seq_world[i], mesh_w, object_centric)


    # 4) IK setup (RB-Y1 with desk; base = identity in viz world)
    desk_pose = np.eye(4, dtype=np.float32)
    desk_pose[:3, 3] = np.array(desk_xyz, dtype=np.float32)
    table_surface_z = float(desk_xyz[2]) + 0.025
    target_pose = np.eye(4, dtype=np.float32)
    target_urdf = _pybullet_target_urdf_for_task(task_name, aria_obj_dir)
    if target_urdf is not None:
        if target_pose_path is None:
            raise ValueError("A prepared target object requires --target-pose (world-frame 4x4 matrix).")
        target_pose = np.loadtxt(target_pose_path).astype(np.float32)
        if target_pose.shape != (4, 4) or not np.isfinite(target_pose).all():
            raise ValueError("--target-pose must contain a finite 4x4 matrix")

    entities = [
        Entity(name="robot", urdf=robot_urdf_path, pose=np.eye(4, dtype=np.float32), color=None, fix_base=True),
        Entity(name="desk",  urdf=desk_urdf_path,  pose=desk_pose, color=[0.5,0.5,0.5,0.5], fix_base=True),
    ]
    tool_urdf = _pybullet_tool_urdf_for_task(task_name, aria_obj_dir)
    if tool_urdf is not None and os.path.isfile(tool_urdf):
        entities.append(Entity(
            name="tool",
            urdf=tool_urdf,
            pose=seq_obj_world[0].astype(np.float32),
            color=[0.1, 0.9, 0.2, 1.0],
            fix_base=True,
        ))
    if target_urdf is not None and os.path.isfile(target_urdf):
        entities.append(Entity(
            name="target",
            urdf=target_urdf,
            pose=target_pose,
            color=[0.9, 0.15, 0.1, 1.0],
            fix_base=True,
        ))

    rby1_model = MinkSolver(robot_mjcf_path)
    robot_viz = PybulletVisualizer(entities) if visualize else None
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

    # Spider xhand right_palm site under right_hand_link:
    # right_palm local quat (wxyz) = [0, 1, 1, 0].
    right_palm_q_wxyz = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    right_palm_q_wxyz /= np.linalg.norm(right_palm_q_wxyz)
    right_palm_q_xyzw = np.array(
        [right_palm_q_wxyz[1], right_palm_q_wxyz[2], right_palm_q_wxyz[3], right_palm_q_wxyz[0]],
        dtype=np.float32,
    )
    T_right_hand_palm = np.eye(4, dtype=np.float32)
    T_right_hand_palm[:3, :3] = R.from_quat(right_palm_q_xyzw).as_matrix().astype(np.float32)


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
            position_weight = 3.0 if k == "thumb_tip" else 1.0
            link_name = link_mapping[k].format("right")
            # absolute
            ik_targets.append(Target(task_name=f"abs_{link_name}", relative=False,
                                     link_name=link_name, root_name="", link_pose=tip_pose,
                                     position_weight=position_weight))
            # relative to wrist
            rel_pose = np.linalg.inv(wrist_pose) @ tip_pose
            ik_targets.append(Target(task_name=f"rel_{link_name}", relative=True,
                                     link_name=link_name,
                                     root_name=link_mapping["wrist"].format("right"),
                                     link_pose=rel_pose,
                                     position_weight=position_weight))
        R_fix = np.array([
            [0, 1,  0],
            [-1,  0,  0],
            [0,  0,  1],
        ], dtype=np.float32)

        wrist_pose_fixed = wrist_pose.copy()
        wrist_pose_fixed[:3, :3] = wrist_pose[:3, :3] @ R_fix

        ik_targets.append(Target(
            task_name="ori_link_right_arm_6",   # special name we'll key on
            relative=False,
            root_name="",                      # unused for absolute task
            link_name="link_right_arm_6",
            link_pose=wrist_pose_fixed,         # use Tw: only its orientation will matter
            orientation_weight=1.0,
        ))
        fixed_links = ["base",'link_torso_0','link_torso_1','link_torso_2','link_torso_3','link_torso_4','link_torso_5']
        res = rby1_model.compute_ik(q_dict=init_q_dict, ik_targets=ik_targets,
                                    fixed_links=fixed_links, solver="quadprog",
                                    num_iter=50, return_err=True)
        return res["q"]

    retarget_traj = []
    for i in range(len(seq_world)):
        rh_targets = seq_world[i]
        q_next = _retarget_one_frame(rby1_model, init_q_dict, rh_targets, hand2robot_link_mapping)
        # freeze left hand & left arm
        _zero_out_joints(q_next, left_hand_joints + ["left_arm_0","left_arm_1","left_arm_2","left_arm_3","left_arm_4","left_arm_5","left_arm_6"])

        if visualize:
            poses_to_show = [rh_targets["wrist"]] + [rh_targets[n] for n in TIP_NAMES]
            robot_viz.visualize_poses(poses_to_show, ax_len=0.10)
            if "tool" in robot_viz.all_ids:
                robot_viz.set_object_poses({"tool": seq_obj_world[i]})
            if "target" in robot_viz.all_ids:
                robot_viz.set_object_poses({"target": target_pose})
            robot_viz.set_joint_configuration(q_next)
            robot_viz.run(duration=0.1)

        # Save 5 fingertip positions as reference (world + tool-local).
        tool_T = seq_obj_world[i]
        tool_R = tool_T[:3, :3]
        tool_p = tool_T[:3, 3]
        fingertips_world = {}
        fingertips_tool = {}
        for tip_name in TIP_NAMES:
            p_w = rh_targets[tip_name][:3, 3].astype(np.float32)
            p_l = (tool_R.T @ (p_w - tool_p)).astype(np.float32)
            fingertips_world[tip_name] = p_w.tolist()
            fingertips_tool[tip_name] = p_l.tolist()

        fk_links = rby1_model.compute_fk(q_next, ["right_hand_link", "link_right_arm_6"])

        right_arm_6_T = fk_links["link_right_arm_6"]
        right_arm_6_q_xyzw = R.from_matrix(right_arm_6_T[:3, :3]).as_quat().astype(np.float32)
        right_arm_6_q_wxyz = np.array(
            [right_arm_6_q_xyzw[3], right_arm_6_q_xyzw[0], right_arm_6_q_xyzw[1], right_arm_6_q_xyzw[2]],
            dtype=np.float32,
        )
        right_arm_6_7d = np.concatenate([right_arm_6_T[:3, 3].astype(np.float32), right_arm_6_q_wxyz]).tolist()

        right_eef_T = fk_links["right_hand_link"] @ T_right_hand_palm
        right_eef_q_xyzw = R.from_matrix(right_eef_T[:3, :3]).as_quat().astype(np.float32)
        right_eef_q_wxyz = np.array(
            [right_eef_q_xyzw[3], right_eef_q_xyzw[0], right_eef_q_xyzw[1], right_eef_q_xyzw[2]],
            dtype=np.float32,
        )
        right_eef_7d = np.concatenate([right_eef_T[:3, 3].astype(np.float32), right_eef_q_wxyz]).tolist()

        init_q_dict = q_next.copy()
        retarget_traj.append({
            "robot_cfg": q_next,
            "right_arm_6": right_arm_6_7d,
            "right_eef": right_eef_7d,
            "tool": seq_obj_world[i].tolist() if seq_obj_world is not None else np.eye(4, dtype=np.float32).tolist(),
            "target": target_pose.tolist(),
            "fingertips_world": fingertips_world,
            "fingertips_tool": fingertips_tool,
            "desk": desk_pose.tolist(),
            # base = I in viz world → record camera world pose
            "target2camera": cam2w[i].tolist(),
            "intrinsic": cam_K.tolist(),
        })

    # Save JSON (same schema)
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(hand_json_path)) or "."
        out_path = os.path.join(out_dir, "retarget.json")

    refined_tool_traj = [step["tool"] for step in retarget_traj]
    refined_target_traj = [step["target"] for step in retarget_traj]

    payload = {
        "info": {
                "tool_id": "none",
                "target_id": "hammer" if task_name == "hammer" else "none",
                "task_name": task_name,
            },
        "init_state": {
            "robot_cfg": retarget_traj[0]["robot_cfg"],
            "tool": refined_tool_traj[0],
            "target": refined_target_traj[0],
            "fingertips_world": retarget_traj[0]["fingertips_world"],
            "fingertips_tool": retarget_traj[0]["fingertips_tool"],
            "desk": desk_pose.tolist(),
            "robot_base_pose": np.eye(4, dtype=np.float32).tolist()
        },
        "traj": retarget_traj,
        "camera_traj": {"target2camera":[step["target2camera"] for step in retarget_traj],
                        "cam_K": retarget_traj[0]["intrinsic"]},
        "warm_up_steps": 0,
        "cam2tool_Ts": [(np.linalg.inv(cam2w[i]) @ np.asarray(refined_tool_traj[i], dtype=np.float32)).tolist() for i in range(Tn)]
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"[RETARGET] Saved to {out_path}")
    return out_path

# ---------------- Argument parsing & entry ----------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="World-frame hand/cam/tag GIF and RB-Y1 retargeting")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Defaults
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ego_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_dir)))
    DEF_W2CAM = os.path.join(ego_dir, "egoengine/Aria/mustard/data/Egocentric_Camera_Parameters/801/1/upd_egocentric_frame_extrinsic.npy")
    DEF_HAND  = os.path.join(ego_dir, "egoengine/Aria/mustard/data/Hand_Poses/801/1/all_pose.json")
    DEF_TAG   = os.path.join(curr_dir, "tag_in_world.txt")

    # gif subcommand
    pg = sub.add_parser("gif", help="Generate world-frame 3D GIF")
    pg.add_argument("--w2cam", default=DEF_W2CAM, help="Path to world->camera extrinsics .npy (T,4,4)")
    pg.add_argument("--hand-json", default=DEF_HAND, help="Path to Aria hand all_pose.json")
    pg.add_argument("--tag-world", default=DEF_TAG, help="Path to 4x4 world_T_tag txt file")
    pg.add_argument("--out-gif", default=os.path.join(os.path.dirname(__file__), "hand_cam_tag_world.gif"),
                    help="Output GIF path")

    # retarget subcommand
    pr = sub.add_parser("retarget", help="Retarget right-hand poses to RB-Y1+XHand")
    pr.add_argument("--w2cam", default=DEF_W2CAM, help="Path to world->camera extrinsics .npy (T,4,4)")
    pr.add_argument("--hand-json", default=DEF_HAND, help="Path to Aria hand all_pose.json")
    pr.add_argument("--tag-world", default=DEF_TAG, help="Path to 4x4 world_T_tag txt file")
    pr.add_argument("--out-json", default=None, help="Output retarget JSON (default: alongside hand_json)")
    pr.add_argument("--visualize", action="store_true", help="Enable PyBullet visualization")
    pr.add_argument(
        "--object-centric",
        type=float,
        default=DEFAULT_OBJECT_CENTRIC_THRESH,
        help="Threshold distance (m) to snap fingertip to closest object mesh surface point (0 = disabled).",
    )
    pr.add_argument(
        "--task-name",
        "--task_name",
        choices=TASK_NAME_CHOICES,
        default=None,
        help="Aria task name for selecting scene assets; trajectories are not adjusted by task.",
    )
    pr.add_argument("--target-pose", default=None, help="Prepared fixed target world pose as a 4x4 text matrix (required for hammer shelf)")
    pr.add_argument("--start-frame", type=int, default=0, help="First frame index to use for retargeting")
    pr.add_argument("--desk-x", type=float, default=0.9, help="Desk X in world (m)")
    pr.add_argument("--desk-y", type=float, default=0.0, help="Desk Y in world (m)")
    pr.add_argument("--desk-z", type=float, default=0.699, help="Desk Z in world (m)")
    # assets
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pr.add_argument("--robot-urdf", default=os.path.join(curr_dir, "assets/rbyxhand_modified/model_modified.urdf"), help="Robot URDF path")
    pr.add_argument("--robot-mjcf", default=os.path.join(curr_dir, "assets/rbyxhand_modified/model_modified.xml"), help="Robot MJCF path")
    pr.add_argument("--desk-urdf",  default="retargeting/assets/desk/desk.urdf", help="Desk URDF path")

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
        # resolve visualize flag precedence
        visualize = args.visualize
        retarget_to_robot(
            w2cam_path=args.w2cam,
            hand_json_path=args.hand_json,
            tag_world_txt=args.tag_world,
            out_path=args.out_json,
            visualize=visualize,
            object_centric=args.object_centric,
            task_name=args.task_name,
            target_pose_path=args.target_pose,
            start_frame=args.start_frame,
            robot_urdf_path=args.robot_urdf,
            robot_mjcf_path=args.robot_mjcf,
            desk_urdf_path=args.desk_urdf,
            desk_xyz=(args.desk_x, args.desk_y, args.desk_z),
        )
    else:
        parser.error("Unknown subcommand")

if __name__ == "__main__":
    main()
