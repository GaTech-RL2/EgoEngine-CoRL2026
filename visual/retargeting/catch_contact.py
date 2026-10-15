import json
import sys
import time

import numpy as np
import pickle
import torch
import mano
import tqdm

# from mano.utils import Mesh
from scipy.spatial.transform import Rotation as R

# import open3d as o3d
from glob import glob
import os
import os.path as osp
import argparse
import trimesh
import matplotlib.pyplot as plt
from .pb_visualizer import PybulletVisualizer, Entity
from .rby1_kinematics import MinkSolver, Target
from manotorch.manolayer import ManoLayer, MANOOutput
from trimesh import Trimesh
from .hand_pose_loader import mano_params_to_hand_info

_alias_map = [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
]
for _name, _type in _alias_map:
    if not hasattr(np, _name):
        setattr(np, _name, _type)

DESK_X = 0.9
DESK_Z = 0.724
DESK_Y = 0.0

HUMAN_BASE_X_OFFSET_TO_OBJ = -(DESK_X - 0.3)
HUMAN_BASE_Z_OFFSET_TO_OBJ = -(DESK_Z + 0.03)
HUMAN_BASE_Y_OFFSET_TO_OBJ = -(DESK_Y - 0.04)

INIT_WRIST_Z_OFFSET_IN_W = 0.15  # initial wrist height offset in world frame
INIT_WRIST_X_OFFSET_IN_W = -0.1  # -0.3 # initial wrist x offset in world frame
WRIST_Z_OFFSET = 0.00
OBJ_SCALE = 0.01
EPS = 1e-10
WARM_UP_STEPS = 0
DEFAULT_SMOOTH_N = 0
TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


def get_target2camera(target_world_poses, world2cam):
    assert target_world_poses.shape == world2cam.shape
    return [world2cam[i] @ target_world_poses[i] for i in range(len(target_world_poses))]


def get_camera_world_pose(target_world_poses, target2camera):
    cam2w, w2c = [], []
    for Tw, Tc in zip(target_world_poses, target2camera):
        Twc = Tc @ np.linalg.inv(Tw)  # world→cam
        w2c.append(Twc)
        cam2w.append(np.linalg.inv(Twc))  # cam→world
    return cam2w, w2c


def _kernel_avg_se3(mats: np.ndarray, half_win: int) -> np.ndarray:
    """Simple kernel average for SE(3) with window 2N+1.
    Translation: mean. Orientation: average rotation matrices + re-orthonormalize.
    """
    if half_win <= 0:
        return mats
    mats = np.asarray(mats)
    out = mats.copy()
    T = mats.shape[0]
    for i in range(T):
        s = max(0, i - half_win)
        e = min(T, i + half_win + 1)
        window = mats[s:e]
        # translation average
        out[i, :3, 3] = window[:, :3, 3].mean(axis=0)
        # rotation average via SVD of mean rotation matrix
        R_mean = window[:, :3, :3].mean(axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        out[i, :3, :3] = R_ortho
    return out


def filter_hand_seq(seq: list, half_win: int) -> list:
    if half_win <= 0 or not seq:
        return seq
    keys = list(seq[0].keys())
    out = [dict() for _ in range(len(seq))]
    for k in keys:
        mats = np.stack([frame[k] for frame in seq], axis=0)
        mats_f = _kernel_avg_se3(mats, half_win)
        for i in range(len(seq)):
            out[i][k] = mats_f[i]
    return out


def _mesh_from_obj(obj_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, process=False)
    mesh.vertices *= OBJ_SCALE
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.rezero()
        mesh.compute_vertex_normals()
    return mesh


def _transform_mesh(mesh: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    m = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces, process=False)
    m.apply_transform(T)
    return m


def _nearest_vertex_point(mesh: trimesh.Trimesh, pt: np.ndarray):
    v = mesh.vertices
    idx = np.argmin(np.linalg.norm(v - pt[None, :], axis=1))
    n = mesh.vertex_normals[idx] if mesh.vertex_normals is not None else np.array([0, 0, 1.0])
    return v[idx], n


def _snap_tip_to_surface(tip_pose: np.ndarray, mesh: trimesh.Trimesh, threshold: float) -> tuple[np.ndarray, bool]:
    pt = tip_pose[:3, 3]
    # Find the closest mesh vertex to the tip
    v = mesh.vertices
    dists = np.linalg.norm(v - pt[None, :], axis=1)
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    closest_vertex = v[min_idx]
    if min_dist > threshold:
        return tip_pose, False
    out = tip_pose.copy()
    out[:3, 3] = closest_vertex
    return out, True


def _object_centric_targets(targets: dict, mesh: trimesh.Trimesh, threshold: float) -> tuple[dict, dict]:
    out = targets.copy()
    contact = {name: False for name in TIP_NAMES}
    for name in TIP_NAMES:
        out[name], contact[name] = _snap_tip_to_surface(out[name], mesh, threshold)
    return out, contact


def _rot_geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two rotation matrices."""
    R_rel = R_a.transpose(0, 2, 1) @ R_b
    tr = np.trace(R_rel, axis1=1, axis2=2)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos_th)


def _mask_to_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert bool mask to [start, end) spans."""
    spans = []
    if mask.size == 0:
        return spans
    in_run = False
    start = 0
    for i, v in enumerate(mask.astype(bool)):
        if v and not in_run:
            in_run = True
            start = i
        elif (not v) and in_run:
            spans.append((start, i))
            in_run = False
    if in_run:
        spans.append((start, mask.shape[0]))
    return spans


def _remove_short_true_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = mask.copy().astype(bool)
    for s, e in _mask_to_spans(out):
        if (e - s) < min_run:
            out[s:e] = False
    return out


def _compute_contact_stage_mask(
    hand_seq: list[dict],
    obj_poses: np.ndarray,
    mesh_base: trimesh.Trimesh | None = None,
    dist_thresh: float = 0.04,
    z_lift_eps: float = 1e-4,
    min_fingers: int = 1,
    min_run: int = 3,
    sim_pos_thresh: float = 0.03,
    sim_dist_thresh: float = 0.015,
) -> tuple[np.ndarray, int]:
    """Prototype-driven contact stage (single segment + single prototype).

    Priors:
    1) only one contact segment for the whole video (keep longest final segment);
    2) object above its lowest Z => candidate contact mode;
    3) fingertip farther than 4cm from object mesh => non-contact for that fingertip.
    4) use prototype similarity to select contact timesteps.
    """
    T = min(len(hand_seq), len(obj_poses))
    if T <= 1:
        return np.zeros((T,), dtype=bool), -1

    obj_z = obj_poses[:T, 2, 3]
    z_min = float(obj_z.min())
    lift_mask = obj_z > (z_min + float(z_lift_eps))

    tip_dist_map = _compute_tip_mesh_dists(hand_seq, obj_poses, mesh_base)
    near_count = np.zeros((T,), dtype=int)
    mean_dist = np.zeros((T,), dtype=float)
    for tip_name in TIP_NAMES:
        d = tip_dist_map[tip_name]
        near_count += (d <= float(dist_thresh)).astype(int)
        mean_dist += d
    mean_dist /= float(len(TIP_NAMES))

    # Candidate contact from priors.
    candidate = lift_mask & (near_count >= int(min_fingers))
    candidate = _remove_short_true_runs(candidate, min_run=max(1, int(min_run)))

    candidate_ids = np.where(candidate)[0]
    if candidate_ids.size == 0:
        return np.zeros((T,), dtype=bool), -1

    # Build object-relative fingertip position features.
    obj_pos = obj_poses[:T, :3, 3]
    obj_rot = obj_poses[:T, :3, :3]
    rel_pos_map = {}
    for tip_name in TIP_NAMES:
        tip_poses = np.stack([hand_seq[t][tip_name] for t in range(T)], axis=0)
        tip_pos = tip_poses[:, :3, 3]
        rel = np.einsum("tij,tj->ti", obj_rot.transpose(0, 2, 1), (tip_pos - obj_pos))
        rel_pos_map[tip_name] = rel

    # Pick prototype inside candidate set: prefer many near fingertips, then small mean dist.
    best_score = None
    prototype = int(candidate_ids[0])
    for t in candidate_ids.tolist():
        score = float(mean_dist[t]) - 0.005 * float(near_count[t])
        if (best_score is None) or (score < best_score):
            best_score = score
            prototype = int(t)

    # Similarity to prototype:
    # - rel_pos mean error (object-centric fingertip layout)
    # - mesh distance profile error
    pos_sim = np.zeros((T,), dtype=float)
    dist_sim = np.zeros((T,), dtype=float)
    for t in range(T):
        pos_acc = 0.0
        dist_acc = 0.0
        for tip_name in TIP_NAMES:
            pos_acc += float(np.linalg.norm(rel_pos_map[tip_name][t] - rel_pos_map[tip_name][prototype]))
            dist_acc += float(abs(tip_dist_map[tip_name][t] - tip_dist_map[tip_name][prototype]))
        pos_sim[t] = pos_acc / float(len(TIP_NAMES))
        dist_sim[t] = dist_acc / float(len(TIP_NAMES))

    similar = (pos_sim <= float(sim_pos_thresh)) & (dist_sim <= float(sim_dist_thresh))
    final_mask = candidate & similar
    final_mask = _remove_short_true_runs(final_mask, min_run=max(1, int(min_run)))

    spans = _mask_to_spans(final_mask)
    if not spans:
        # If similarity is too strict, at least keep prototype as a 1-frame stage.
        fallback = np.zeros((T,), dtype=bool)
        fallback[prototype] = True
        return fallback, prototype

    # Keep exactly one segment: longest.
    spans_sorted = sorted(spans, key=lambda se: (se[1] - se[0], -se[0]), reverse=True)
    s_best, e_best = spans_sorted[0]
    out = np.zeros((T,), dtype=bool)
    out[s_best:e_best] = True
    return out, prototype


def _compute_tip_mesh_dists(
    hand_seq: list[dict],
    obj_poses: np.ndarray,
    mesh_base: trimesh.Trimesh | None,
) -> dict[str, np.ndarray]:
    """Per-tip distance series to object surface (nearest-vertex approximation).

    If mesh is missing, fallback to object-origin distance.
    """
    T = min(len(hand_seq), len(obj_poses))
    if T <= 0:
        return {k: np.zeros((0,), dtype=float) for k in TIP_NAMES}

    obj_pos = obj_poses[:T, :3, 3]
    if mesh_base is None:
        out = {}
        for tip_name in TIP_NAMES:
            tip_poses = np.stack([hand_seq[t][tip_name] for t in range(T)], axis=0)
            tip_pos = tip_poses[:, :3, 3]
            out[tip_name] = np.linalg.norm(tip_pos - obj_pos, axis=1)
        return out

    # Transform base mesh vertices into world for each timestep.
    verts0 = np.asarray(mesh_base.vertices, dtype=float)  # (V,3)
    verts_world = []
    for t in range(T):
        Rt = obj_poses[t, :3, :3]
        pt = obj_poses[t, :3, 3]
        vt = verts0 @ Rt.T + pt[None, :]
        verts_world.append(vt)

    out = {}
    for tip_name in TIP_NAMES:
        tip_poses = np.stack([hand_seq[t][tip_name] for t in range(T)], axis=0)
        tip_pos = tip_poses[:, :3, 3]
        d = np.zeros((T,), dtype=float)
        for t in range(T):
            diff = verts_world[t] - tip_pos[t][None, :]
            d[t] = np.linalg.norm(diff, axis=1).min()
        out[tip_name] = d
    return out


def _plot_tip_to_object_6d(
    rh_seq: list[dict],
    lh_seq: list[dict],
    tool_poses: np.ndarray,
    target_poses: np.ndarray,
    tool_mesh_base: trimesh.Trimesh | None = None,
    target_mesh_base: trimesh.Trimesh | None = None,
):
    """Two colorful figures:
    - Figure 1: right hand -> tool
    - Figure 2: left hand -> target
    y value: translational distance |p_obj - p_tip|
    color: rotational error angle between tip frame and object frame
    """
    T = min(len(rh_seq), len(lh_seq), len(tool_poses), len(target_poses))
    x = np.arange(T)
    cmap = "turbo"
    vmin, vmax = 0.0, np.pi
    rh_contact_mask, rh_proto = _compute_contact_stage_mask(
        rh_seq, tool_poses, mesh_base=tool_mesh_base
    )
    lh_contact_mask, lh_proto = _compute_contact_stage_mask(
        lh_seq, target_poses, mesh_base=target_mesh_base
    )
    rh_spans = _mask_to_spans(rh_contact_mask)
    lh_spans = _mask_to_spans(lh_contact_mask)
    rh_tip_dists = _compute_tip_mesh_dists(rh_seq, tool_poses, tool_mesh_base)
    lh_tip_dists = _compute_tip_mesh_dists(lh_seq, target_poses, target_mesh_base)
    print(f"[contact_stage] right->tool spans={rh_spans} frames={int(rh_contact_mask.sum())}/{T}")
    print(f"[contact_stage] left->target spans={lh_spans} frames={int(lh_contact_mask.sum())}/{T}")
    print(f"[prototype] right->tool frame={rh_proto}")
    print(f"[prototype] left->target frame={lh_proto}")
    print("[logic] contact = lifted & near-mesh & similar-to-prototype; keep longest single segment")

    def _draw_group(
        ax,
        hand_seq: list[dict],
        obj_poses: np.ndarray,
        tip_dists: dict[str, np.ndarray],
        hand_tag: str,
        marker: str,
    ):
        sc_local = None
        for tip_name in TIP_NAMES:
            tip_poses = np.stack([hand_seq[t][tip_name] for t in range(T)], axis=0)  # (T,4,4)
            tip_rot = tip_poses[:, :3, :3]
            obj_rot = obj_poses[:T, :3, :3]

            pos_len = tip_dists[tip_name]  # value: distance to object mesh
            rot_ang = _rot_geodesic_angle(tip_rot, obj_rot)      # color

            ax.plot(x, pos_len, linewidth=0.6, alpha=0.25)
            sc_local = ax.scatter(
                x,
                pos_len,
                c=rot_ang,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=10,
                marker=marker,
                label=f"{hand_tag}:{tip_name}",
            )
        return sc_local

    # Figure 1: right hand -> tool (top) + tool world vector (bottom)
    fig_r, (ax_r_top, ax_r_bot) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    sc_r = _draw_group(
        ax_r_top, rh_seq, tool_poses, rh_tip_dists, hand_tag="R->tool", marker="o"
    )
    cbar_r = fig_r.colorbar(sc_r, ax=ax_r_top)
    cbar_r.set_label("orientation error (rad)")
    ax_r_top.set_title("Fingertip -> object 6D vectors (R->tool)")
    ax_r_top.set_ylabel("distance (m)")
    ax_r_top.grid(True, alpha=0.25)
    ax_r_top.legend(loc="upper right", fontsize=7, ncol=2)

    tool_xyz = tool_poses[:T, :3, 3]
    tool_norm = np.linalg.norm(tool_xyz, axis=1)  # value
    I_rots = np.repeat(np.eye(3)[None, :, :], T, axis=0)
    tool_rot = tool_poses[:T, :3, :3]
    tool_world_ang = _rot_geodesic_angle(I_rots, tool_rot)  # color
    ax_r_bot.plot(x, tool_norm, linewidth=0.8, alpha=0.25, color="tab:blue")
    sc_r_bot = ax_r_bot.scatter(
        x, tool_norm, c=tool_world_ang, cmap=cmap, vmin=vmin, vmax=vmax, s=12, marker="o"
    )
    cbar_r_bot = fig_r.colorbar(sc_r_bot, ax=ax_r_bot)
    cbar_r_bot.set_label("tool world rot error (rad)")
    ax_r_bot.set_xlabel("timestep")
    ax_r_bot.set_ylabel("|tool_w| (m)")
    ax_r_bot.grid(True, alpha=0.25)
    for s, e in rh_spans:
        ax_r_top.axvspan(s, e, color="red", alpha=0.28, linewidth=0)
        ax_r_bot.axvspan(s, e, color="red", alpha=0.28, linewidth=0)
    if rh_proto >= 0:
        ax_r_top.axvline(rh_proto, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
        ax_r_bot.axvline(rh_proto, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
    fig_r.tight_layout()

    # Figure 2: left hand -> target (top) + target world vector (bottom)
    fig_l, (ax_l_top, ax_l_bot) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    sc_l = _draw_group(
        ax_l_top, lh_seq, target_poses, lh_tip_dists, hand_tag="L->target", marker="x"
    )
    cbar_l = fig_l.colorbar(sc_l, ax=ax_l_top)
    cbar_l.set_label("orientation error (rad)")
    ax_l_top.set_title("Fingertip -> object 6D vectors (L->target)")
    ax_l_top.set_ylabel("distance (m)")
    ax_l_top.grid(True, alpha=0.25)
    ax_l_top.legend(loc="upper right", fontsize=7, ncol=2)

    target_xyz = target_poses[:T, :3, 3]
    target_norm = np.linalg.norm(target_xyz, axis=1)  # value
    target_rot = target_poses[:T, :3, :3]
    target_world_ang = _rot_geodesic_angle(I_rots, target_rot)  # color
    ax_l_bot.plot(x, target_norm, linewidth=0.8, alpha=0.25, color="tab:orange")
    sc_l_bot = ax_l_bot.scatter(
        x, target_norm, c=target_world_ang, cmap=cmap, vmin=vmin, vmax=vmax, s=12, marker="x"
    )
    cbar_l_bot = fig_l.colorbar(sc_l_bot, ax=ax_l_bot)
    cbar_l_bot.set_label("target world rot error (rad)")
    ax_l_bot.set_xlabel("timestep")
    ax_l_bot.set_ylabel("|target_w| (m)")
    ax_l_bot.grid(True, alpha=0.25)
    for s, e in lh_spans:
        ax_l_top.axvspan(s, e, color="red", alpha=0.28, linewidth=0)
        ax_l_bot.axvspan(s, e, color="red", alpha=0.28, linewidth=0)
    if lh_proto >= 0:
        ax_l_top.axvline(lh_proto, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
        ax_l_bot.axvline(lh_proto, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
    fig_l.tight_layout()

    plt.show()
    plt.close(fig_r)
    plt.close(fig_l)


def preview_one_frame(data_dir, mesh_dir, frame=0, device="cpu"):
    object_pose_dir = data_dir.replace(
        os.sep + "Hand_Poses" + os.sep, os.sep + "Object_Poses" + os.sep
    )
    target_path = glob(os.path.join(object_pose_dir, "target_*.npy"))[0]
    tool_path = glob(os.path.join(object_pose_dir, "tool_*.npy"))[0]

    target_id = os.path.basename(target_path).split(".")[0].split("_")[1]
    tool_id = os.path.basename(tool_path).split(".")[0].split("_")[1]

    target_mesh = trimesh.load(os.path.join(mesh_dir, f"{target_id}_cm.obj"))
    tool_mesh = trimesh.load(os.path.join(mesh_dir, f"{tool_id}_cm.obj"))
    target_mesh.vertices *= OBJ_SCALE
    tool_mesh.vertices *= OBJ_SCALE

    target_pose = np.load(target_path)[frame]
    tool_pose = np.load(tool_path)[frame]
    target_mesh.apply_transform(target_pose)
    tool_mesh.apply_transform(tool_pose)

    # hand verts + faces
    rh_vertices, _, rh_frames, faces = mano_params_to_hand_info(
        os.path.join(data_dir, "right_hand.pkl"),
        mano_beta=np.zeros(10),
        side="right",
        return_faces=True,
        return_frames=True,
        device=device,
    )
    lh_vertices, _, lh_frames = mano_params_to_hand_info(
        os.path.join(data_dir, "left_hand.pkl"),
        mano_beta=np.zeros(10),
        side="left",
        return_frames=True,
        device=device,
    )

    scene = trimesh.Scene()
    scene.add_geometry(
        [
            Trimesh(rh_vertices[frame], faces),
            Trimesh(lh_vertices[frame], faces),
            tool_mesh,
            target_mesh,
        ]
    )
    scene.show()


def preview_video(data_dir, mesh_dir, device="cpu", fps=30):
    """
    Reliable playback using Open3D (trimesh's pyglet viewer often won't update
    deforming mesh vertices once the window is running).
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            "Open3D is required for animated deforming meshes here.\n"
            "Install:  pip install open3d\n"
            "or (conda): conda install -c open3d-admin open3d\n"
            f"Import error: {e}"
        )

    obj_pose_dir = data_dir.replace(
        os.sep + "Hand_Poses" + os.sep, os.sep + "Object_Poses" + os.sep
    )
    target_path = glob(os.path.join(obj_pose_dir, "target_*.npy"))[0]
    tool_path = glob(os.path.join(obj_pose_dir, "tool_*.npy"))[0]

    target_id = os.path.basename(target_path).split(".")[0].split("_")[1]
    tool_id = os.path.basename(tool_path).split(".")[0].split("_")[1]

    target_obj = os.path.join(mesh_dir, f"{target_id}_cm.obj")
    tool_obj = os.path.join(mesh_dir, f"{tool_id}_cm.obj")

    target_poses = np.load(target_path)  # (T,4,4)
    tool_poses = np.load(tool_path)      # (T,4,4)

    # MANO verts + faces
    rh_verts, _, rh_frames, faces = mano_params_to_hand_info(
        os.path.join(data_dir, "right_hand.pkl"),
        mano_beta=np.zeros(10),
        side="right",
        return_faces=True,
        return_frames=True,
        device=device,
    )
    lh_verts, _, lh_frames = mano_params_to_hand_info(
        os.path.join(data_dir, "left_hand.pkl"),
        mano_beta=np.zeros(10),
        side="left",
        return_frames=True,
        device=device,
    )

    T = min(len(rh_frames), len(lh_frames), len(target_poses), len(tool_poses))
    interval = 1.0 / float(fps)

    def _o3d_mesh_from_vf(v, f):
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(v, dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(f, dtype=np.int32))
        m.compute_vertex_normals()
        return m

    def _read_obj_scaled(path, scale):
        m = o3d.io.read_triangle_mesh(path)
        if m.is_empty():
            raise RuntimeError(f"Open3D failed to load mesh: {path}")
        m.scale(scale, center=(0.0, 0.0, 0.0))
        m.compute_vertex_normals()
        return m

    def _apply_pose_to_vertices(V0, T44):
        V0 = np.asarray(V0, dtype=np.float64)
        ones = np.ones((V0.shape[0], 1), dtype=np.float64)
        Vh = np.concatenate([V0, ones], axis=1)                  # (N,4)
        Vw = (T44 @ Vh.T).T[:, :3]                               # (N,3)
        return Vw

    # Build meshes
    rh = _o3d_mesh_from_vf(rh_verts[0], faces)
    lh = _o3d_mesh_from_vf(lh_verts[0], faces)

    tool = _read_obj_scaled(tool_obj, OBJ_SCALE)
    target = _read_obj_scaled(target_obj, OBJ_SCALE)

    # Cache base vertices for rigid objects so we can overwrite vertices per-frame
    tool_v0 = np.asarray(tool.vertices).copy()
    target_v0 = np.asarray(target.vertices).copy()

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="preview_video", width=960, height=540)

    vis.add_geometry(rh)
    vis.add_geometry(lh)
    vis.add_geometry(tool)
    vis.add_geometry(target)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.mesh_show_back_face = True

    # Initialize camera once
    ctr = vis.get_view_control()
    vis.poll_events()
    vis.update_renderer()

    import time as _time

    for i in range(T):
        # Hands (deforming): overwrite vertices
        rh.vertices = o3d.utility.Vector3dVector(np.asarray(rh_verts[i], dtype=np.float64))
        rh.compute_vertex_normals()

        lh.vertices = o3d.utility.Vector3dVector(np.asarray(lh_verts[i], dtype=np.float64))
        lh.compute_vertex_normals()

        # Rigid objects: overwrite vertices with pose-applied base vertices
        tool.vertices = o3d.utility.Vector3dVector(_apply_pose_to_vertices(tool_v0, tool_poses[i]))
        tool.compute_vertex_normals()

        target.vertices = o3d.utility.Vector3dVector(_apply_pose_to_vertices(target_v0, target_poses[i]))
        target.compute_vertex_normals()

        # Push updates
        vis.update_geometry(rh)
        vis.update_geometry(lh)
        vis.update_geometry(tool)
        vis.update_geometry(target)

        vis.poll_events()
        vis.update_renderer()

        _time.sleep(interval)

    vis.destroy_window()



def create_obj_urdf(obj_path, scale=(1, 1, 1)):
    """
    Create a minimal URDF file for a given .obj mesh.

    Args:
        obj_path (str): Absolute or relative path to the .obj file.
        scale (tuple): Scale to apply to the mesh.

    Returns:
        urdf_path (str): Path to the generated URDF file.
    """
    obj_path = os.path.abspath(obj_path)
    mesh_dir = os.path.dirname(obj_path)
    mesh_name = os.path.basename(obj_path)

    urdf_content = f"""<?xml version="1.0"?>
        <robot name="dynamic_obj">
        <link name="base">
            <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_name}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_name}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
            </collision>
        </link>
        </robot>
        """

    urdf_name = mesh_name.replace(".obj", ".urdf")
    # Create temp directory to store URDF and copy OBJ alongside
    urdf_path = f"{mesh_dir}/{urdf_name}"

    # Write URDF
    with open(urdf_path, "w") as f:
        f.write(urdf_content)

    return urdf_path


def estimate_human_base_frame(tool_pose, target_pose, tool_obj_path, target_obj_path):

    xy_vec = tool_pose[:3, 3] - target_pose[:3, 3]

    y_axis = -xy_vec / (np.linalg.norm(xy_vec) + EPS)
    Rz_cw = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    x_axis = (Rz_cw @ y_axis.reshape((3, 1)))[:, 0]
    z_axis = np.cross(x_axis, y_axis)
    base_rot = np.array([x_axis, y_axis, z_axis]).T

    def _get_min_z(obj_path, obj_pose):
        tool_mesh = trimesh.load(
            obj_path,
        )
        # Get the AABB as a bounding box (min and max corners)
        aabb_min, aabb_max = tool_mesh.bounds
        _half_z_len = (aabb_max[2] - aabb_min[2]) * OBJ_SCALE * 0.5
        _z_min = obj_pose[2, 3] - _half_z_len
        return _z_min

    cent_pos_xy = (tool_pose[:2, 3] + target_pose[:2, 3]) * 0.5
    tool_z_min = _get_min_z(tool_obj_path, tool_pose)
    target_z_min = _get_min_z(target_obj_path, target_pose)

    cent_pos_z = min(target_z_min, tool_z_min)

    cent_pos = cent_pos_xy.tolist() + [cent_pos_z]
    base_pos = (
        cent_pos
        + (x_axis * HUMAN_BASE_X_OFFSET_TO_OBJ)
        + (z_axis * HUMAN_BASE_Z_OFFSET_TO_OBJ)
        + (y_axis * HUMAN_BASE_Y_OFFSET_TO_OBJ)
    )

    base_pose = np.eye(4)
    base_pose[:3, 3] = base_pos
    base_pose[:3, :3] = base_rot
    return base_pose


def load_data(data_dir, mesh_dir, viz=False):

    object_pose_dir = data_dir.replace(
        os.sep + "Hand_Poses" + os.sep, os.sep + "Object_Poses" + os.sep
    )
    target_path = glob(os.path.join(object_pose_dir, "target_*.npy"))[0]
    tool_path = glob(os.path.join(object_pose_dir, "tool_*.npy"))[0]

    target_id = os.path.basename(target_path).split(".")[0].split("_")[1]
    tool_id = os.path.basename(tool_path).split(".")[0].split("_")[1]
    target_obj_path = f"{mesh_dir}/{target_id}_cm.obj"
    tool_obj_path = f"{mesh_dir}/{tool_id}_cm.obj"

    target_urdf = create_obj_urdf(target_obj_path, scale=(OBJ_SCALE,) * 3)
    tool_urdf = create_obj_urdf(tool_obj_path, scale=(OBJ_SCALE,) * 3)

    left_pkl = os.path.join(data_dir, "left_hand.pkl")
    right_pkl = os.path.join(data_dir, "right_hand.pkl")

    rh_verts, _, rh_frames = mano_params_to_hand_info(
        right_pkl, mano_beta=np.zeros(10), side="right", return_frames=True, device="cpu"
    )
    lh_verts, _, lh_frames = mano_params_to_hand_info(
        left_pkl, mano_beta=np.zeros(10), side="left", return_frames=True, device="cpu"
    )

    joint_names = [
        "wrist",
        "index1",
        "index2",
        "index3",
        "middle1",
        "middle2",
        "middle3",
        "pinky1",
        "pinky2",
        "pinky3",
        "ring1",
        "ring2",
        "ring3",
        "thumb1",
        "thumb2",
        "thumb3",
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]

    # implement a contact frame and ending frame
    # during contact, calculate the finger tip pose

    rh_seq, lh_seq = [], []
    for n in range(rh_frames.shape[0]):
        rh_seq.append({name: rh_frames[n, i] for i, name in enumerate(joint_names)})
        lh_seq.append({name: lh_frames[n, i] for i, name in enumerate(joint_names)})
    # for k in range(len(lh_seq)):
    # print(lh_seq[k]['wrist'][:3, 3])
    # exit()
    target_pose = np.load(target_path)
    tool_pose = np.load(tool_path)
    # print("target pose: ", target_pose[0])
    # print("tool pose: ", tool_pose[0])

    human_base_pose = estimate_human_base_frame(
        tool_pose[0], target_pose[0], tool_obj_path, target_obj_path
    )
    # print(human_base_pose)

    if viz:
        preview_one_frame(data_dir, mesh_dir, frame=0)

    return {
        "lh_seq": lh_seq,
        "rh_seq": rh_seq,
        "target_obj_pose": target_pose,
        "tool_obj_pose": tool_pose,
        "est_human_base_pose": human_base_pose,
        "target_urdf": target_urdf,
        "tool_urdf": tool_urdf,
        "target_obj_path": target_obj_path,
        "tool_obj_path": tool_obj_path,
        "target_id": target_id,
        "tool_id": tool_id,
    }


def retarget_one_frame_hierachical(
    rby1_model: MinkSolver, init_q_dict, rh_targets, lh_targets, link_mapping
):
    hand_tasks = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    ik_targets = []
    for k in hand_tasks:
        for side_name, target_info in zip(["left", "right"], [lh_targets, rh_targets]):
            wrist_pose = target_info["wrist"]
            tip_pose = target_info[k]
            link_name = link_mapping[k].format(side_name)
            ik_abs_target = Target(
                task_name=f"{side_name}_abs_{link_name}",
                relative=False,
                link_name=link_name,
                root_name="",
                link_pose=tip_pose,
            )
            ik_targets.append(ik_abs_target)

            rel_pose = np.linalg.inv(wrist_pose) @ tip_pose
            ik_rel_target = Target(
                task_name=f"{side_name}_rel_{link_name}",
                relative=True,
                link_name=link_name,
                root_name=link_mapping["wrist"].format(side_name),
                link_pose=rel_pose,
            )
            ik_targets.append(ik_rel_target)

    for side_name, target_info in zip(["left", "right"], [lh_targets, rh_targets]):
        wrist_pose = target_info["wrist"]
        ik_target = Target(
            task_name=f"{side_name}_wrist",
            relative=False,
            link_name=link_mapping["wrist"].format(side_name),
            root_name="",
            link_pose=wrist_pose,
        )
        ik_targets.append(ik_target)

        # orientation-only wrist task (same pose, orientation weighted in Mink)
        ik_ori_target = Target(
            task_name=f"{side_name}_wrist_ori",
            relative=False,
            link_name=link_mapping["wrist"].format(side_name),
            root_name="",
            link_pose=wrist_pose,
        )
        ik_targets.append(ik_ori_target)
    fixed_links = [
        "base",
        "link_torso_0",
        "link_torso_1",
        "link_torso_2",
        "link_torso_3",
        "link_torso_4",
        "link_torso_5",
    ]

    res = rby1_model.compute_ik(
        q_dict=init_q_dict,
        ik_targets=ik_targets,
        fixed_links=fixed_links,
        solver="quadprog",
        num_iter=50,
        return_err=True,
    )
    # print("errors: ")
    # print(res["errors"])
    return res["q"]


def retarget_one_frame_wrist(
    rby1_model: MinkSolver, init_q_dict, rh_targets, lh_targets, link_mapping
):

    ik_targets = []
    for side_name, target_info in zip(["left", "right"], [lh_targets, rh_targets]):
        wrist_pose = target_info["wrist"]
        wrist_pose[2, 3] += INIT_WRIST_Z_OFFSET_IN_W
        wrist_pose[0, 3] += INIT_WRIST_X_OFFSET_IN_W
        ik_target = Target(
            relative=False,
            task_name=f"init_{side_name}_wrist",
            link_name=link_mapping["wrist"].format(side_name),
            root_name="",
            link_pose=wrist_pose,
        )
        ik_targets.append(ik_target)
    fixed_links = [
        "base",
        "link_torso_0",
        "link_torso_1",
        "link_torso_2",
        "link_torso_3",
        "link_torso_4",
        "link_torso_5",
    ]

    res = rby1_model.compute_ik(
        q_dict=init_q_dict,
        ik_targets=ik_targets,
        fixed_links=fixed_links,
        solver="quadprog",
        num_iter=50,
        return_err=True,
    )
    # print("errors: ")
    # print(res["errors"])
    return res["q"]


def transfer_poses_to_local(lh_seq, rh_seq, tool_pose, target_pose, base_pose):
    def proj_dict(pose_dict, base_pose):
        ret = {}
        for k, v in pose_dict.items():
            ret[k] = np.linalg.inv(base_pose) @ v
        return ret

    for i in range(len(lh_seq)):
        lh_seq[i] = proj_dict(lh_seq[i], base_pose)
        rh_seq[i] = proj_dict(rh_seq[i], base_pose)

    tool_pose = np.linalg.inv(base_pose)[None] @ tool_pose
    target_pose = np.linalg.inv(base_pose)[None] @ target_pose

    return lh_seq, rh_seq, tool_pose, target_pose


def transform_hand_poses(hp_seq, transform):
    def proj_dict(pose_dict, T):
        ret = {}
        for k, v in pose_dict.items():
            ret[k] = T @ v
        return ret

    for i in range(len(hp_seq)):
        hp_seq[i] = proj_dict(hp_seq[i], transform)
    return hp_seq


def lerp_pose_dict(start: dict, goal: dict, ratio: float):
    out = {}
    for k in start:
        T0, T1 = start[k], goal[k]
        Tm = T0.copy()
        Tm[:3, 3] = (1 - ratio) * T0[:3, 3] + ratio * T1[:3, 3]
        out[k] = Tm
    return out


def retarget_to_robot(
    demo_dir,
    mesh_dir,
    save_path,
    skip_viz,
    do_control=False,
    smooth_n=DEFAULT_SMOOTH_N,
    object_centric=0.0,
):

    data = load_data(data_dir=demo_dir, mesh_dir=mesh_dir, viz=False)

    if smooth_n and smooth_n > 0:
        data["tool_obj_pose"] = _kernel_avg_se3(data["tool_obj_pose"], smooth_n)
        data["target_obj_pose"] = _kernel_avg_se3(data["target_obj_pose"], smooth_n)
        data["rh_seq"] = filter_hand_seq(data["rh_seq"], smooth_n)
        data["lh_seq"] = filter_hand_seq(data["lh_seq"], smooth_n)

    # Visualize before Mink, without creating any simulation environment.
    _plot_tip_to_object_6d(
        rh_seq=data["rh_seq"],
        lh_seq=data["lh_seq"],
        tool_poses=data["tool_obj_pose"],
        target_poses=data["target_obj_pose"],
        tool_mesh_base=_mesh_from_obj(data["tool_obj_path"]),
        target_mesh_base=_mesh_from_obj(data["target_obj_path"]),
    )
    exit()

    left_hand_joints = [
        "left_hand_thumb_bend_joint",
        "left_hand_thumb_rota_joint1",
        "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint",
        "left_hand_index_joint1",
        "left_hand_index_joint2",
        "left_hand_mid_joint1",
        "left_hand_mid_joint2",
        "left_hand_ring_joint1",
        "left_hand_ring_joint2",
        "left_hand_pinky_joint1",
        "left_hand_pinky_joint2",
    ]
    right_hand_joints = [
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
    rby1_joints = [
        "wheel_fl",
        "wheel_fr",
        "wheel_rl",
        "wheel_rr",
        "torso_0",
        "torso_1",
        "torso_2",
        "torso_3",
        "torso_4",
        "torso_5",
        "head_0",
        "head_1",
        "left_arm_0",
        "left_arm_1",
        "left_arm_2",
        "left_arm_3",
        "left_arm_4",
        "left_arm_5",
        "left_arm_6",
        "right_arm_0",
        "right_arm_1",
        "right_arm_2",
        "right_arm_3",
        "right_arm_4",
        "right_arm_5",
        "right_arm_6",
    ]

    robot_urdf_path = "/home/ycl/projects/workspace/egoengine/retargeting/assets/rbyxhand_modified/model_modified.urdf"
    robot_mjcf_path = "/home/ycl/projects/workspace/egoengine/retargeting/assets/rbyxhand_modified/model_modified.xml"
    desk_urdf_path = "/home/ycl/projects/workspace/egoengine/retargeting/assets/desk/desk.urdf"

    hand2robot_link_mapping = {
        "thumb_tip": "{}_hand_thumb_rota_tip",
        "index_tip": "{}_hand_index_rota_tip",
        "middle_tip": "{}_hand_mid_tip",
        "ring_tip": "{}_hand_ring_tip",
        "pinky_tip": "{}_hand_pinky_tip",
        "wrist": "{}_hand_link",
    }
    init_q = np.array(
        [0] * (len(left_hand_joints + right_hand_joints + rby1_joints)), dtype="float32"
    )
    init_q_dict = dict(zip(left_hand_joints + right_hand_joints + rby1_joints, init_q.tolist()))
    init_q_dict["torso_1"] = 0.5
    init_q_dict["torso_2"] = -1.0
    init_q_dict["torso_3"] = 0.5

    orig_target_poses = data["target_obj_pose"]
    orig_tool_poses = data["tool_obj_pose"]

    assert len(orig_target_poses) == len(
        orig_tool_poses
    ), f"{orig_target_poses.shape} vs {orig_tool_poses.shape}"

    orig_init_target_pose = orig_target_poses[0]
    orig_init_tool_pose = orig_tool_poses[0]
    adj_init_target_pose = np.linalg.inv(data["est_human_base_pose"]) @ orig_init_target_pose
    adj_init_tool_pose = np.linalg.inv(data["est_human_base_pose"]) @ orig_init_tool_pose
    # rotate objects to align with new desk and robot

    desk_pose = np.eye(4)
    desk_pose[:3, 3] = np.array([DESK_X, DESK_Y, DESK_Z])

    rby1_model = MinkSolver(robot_mjcf_path)

    # print("original init poses: ", orig_init_tool_pose, orig_init_target_pose)
    # print("adjusted init poses: ", adj_init_tool_pose, adj_init_target_pose)

    entities = [
        Entity(name="robot", urdf=robot_urdf_path, pose=np.eye(4), color=None, fix_base=True),
        Entity(
            name="tool",
            urdf=data["tool_urdf"],
            pose=adj_init_tool_pose,
            color=[0, 1, 0, 1],
            fix_base=False,
        ),
        Entity(
            name="target",
            urdf=data["target_urdf"],
            pose=adj_init_target_pose,
            color=[1, 0, 0, 1],
            fix_base=False,
        ),
        Entity(
            name="desk",
            urdf=desk_urdf_path,
            pose=desk_pose,
            color=[0.5, 0.5, 0.5, 0.5],
            fix_base=True,
        ),
    ]
    robot_viz = PybulletVisualizer(entities)
    robot_viz.run(2)

    simulated_object_poses = robot_viz.get_object_pose(["tool", "target"])

    tool_frame_transform = simulated_object_poses["tool"] @ np.linalg.inv(orig_init_tool_pose)
    target_frame_transform = simulated_object_poses["target"] @ np.linalg.inv(orig_init_target_pose)

    # get stable object poses and use them to adjust the hand targets

    # visualize init scenario
    # robot_viz.set_joint_configuration(init_q_dict)
    # robot_viz.run()

    adj_tool_poses = tool_frame_transform[None] @ orig_tool_poses
    adj_target_poses = target_frame_transform[None] @ orig_target_poses

    rh_seq = data["rh_seq"]
    lh_seq = data["lh_seq"]
    rh_seq = transform_hand_poses(rh_seq, tool_frame_transform)  # assume right hand handle tool
    lh_seq = transform_hand_poses(lh_seq, target_frame_transform)  # assume left hand handle target

    tool_mesh_base = _mesh_from_obj(data["tool_obj_path"])
    target_mesh_base = _mesh_from_obj(data["target_obj_path"])

    # Initialize with hierarchical retargeting directly (no arbitrary offsets)
    init_res = retarget_one_frame_hierachical(
        rby1_model=rby1_model,
        init_q_dict=init_q_dict,
        rh_targets=rh_seq[0],
        lh_targets=lh_seq[0],
        link_mapping=hand2robot_link_mapping,
    )
    cam_dir = demo_dir.replace(
        os.sep + "Hand_Poses" + os.sep,
        os.sep + "Egocentric_Camera_Parameters" + os.sep,
    )

    K = np.loadtxt(os.path.join(cam_dir, "egocentric_intrinsic.txt"))
    w2c_raw = np.load(os.path.join(cam_dir, "egocentric_frame_extrinsic.npy"))  # (N,4,4)

    tgt_w = orig_target_poses  # (N,4,4) already in memory
    tgt2cam = get_target2camera(tgt_w, w2c_raw)
    # cam2w, _= get_camera_world_pose(tgt_w, tgt2cam)

    camera_traj = {
        "target2camera": [m.tolist() for m in tgt2cam],  # list of 4×4 lists
        "intrinsic": K.tolist(),  # 3×3 list
    }

    retarget_data = {
        "init_state": {
            "robot_cfg": init_res.copy(),
            "tool": adj_tool_poses[0].tolist(),
            "target": adj_target_poses[0].tolist(),
            "desk": desk_pose.tolist(),
        },
        "traj": [],
        "camera_traj": camera_traj,
    }

    # No warmup trajectory - start directly from first frame
    init_q_dict = init_res.copy()
    retarget_traj = []
    T = len(rh_seq)

    for i in tqdm.tqdm(range(T)):
        rh_targets = rh_seq[i]
        lh_targets = lh_seq[i]
        if object_centric and object_centric > 0:
            tool_mesh_w = _transform_mesh(tool_mesh_base, adj_tool_poses[i])
            target_mesh_w = _transform_mesh(target_mesh_base, adj_target_poses[i])
            rh_targets, _ = _object_centric_targets(rh_targets, tool_mesh_w, object_centric)
            lh_targets, _ = _object_centric_targets(lh_targets, target_mesh_w, object_centric)

        res = retarget_one_frame_hierachical(
            rby1_model=rby1_model,
            init_q_dict=init_q_dict,
            rh_targets=rh_targets,
            lh_targets=lh_targets,
            link_mapping=hand2robot_link_mapping,
        )

        if not skip_viz:
            viz_target = [
                "wrist",
                "thumb_tip",
                "index_tip",
                "middle_tip",
                "ring_tip",
                "pinky_tip",
            ]
            rh_poses = [rh_targets[vt] for vt in viz_target]
            lh_poses = [lh_targets[vt] for vt in viz_target]
            vz_poses = lh_poses + rh_poses
            # reached_poses = rby1_model.compute_fk(res, ["left_hand_link", "right_hand_link"])
            # reached_poses = list(reached_poses.values())
            robot_viz.visualize_poses(vz_poses, ax_len=0.05)

            if i == 0:
                robot_viz.set_joint_configuration(res)
                robot_viz.set_object_poses(
                    {"tool": adj_tool_poses[i], "target": adj_target_poses[i]}
                )
                robot_viz.run(duration=0.1)
            else:
                if not do_control:
                    robot_viz.set_joint_configuration(res)
                    robot_viz.set_object_poses(
                        {"tool": adj_tool_poses[i], "target": adj_target_poses[i]}
                    )
                    # Run the simulation for 10 seconds
                    robot_viz.run(duration=0.1)
                else:
                    robot_viz.control_robot(res, n=240 * 0.1)

        retarget_res = {
            "robot_cfg": res,
            "tool": adj_tool_poses[i].tolist(),
            "target": adj_target_poses[i].tolist(),
            "desk": desk_pose.tolist(),
            "target2camera": camera_traj["target2camera"][i],
            "intrinsic": camera_traj["intrinsic"],
        }

        init_q_dict = res.copy()
        retarget_traj.append(retarget_res)

    retarget_data["traj"] = retarget_traj
    retarget_data["warm_up_steps"] = WARM_UP_STEPS
    retarget_data["info"] = {"tool_id": data["tool_id"], "target_id": data["target_id"]}
    retarget_data["cam2tool_Ts"] = [(np.linalg.inv(w2c_raw[i]) @ adj_tool_poses[i]).tolist() for i in range(len(adj_tool_poses))]

    # save results to json
    with open(save_path, "w") as f:
        json.dump(retarget_data, f, indent=4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retarget ONE TACO demo → RB-Y1 + XHand")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("retarget", help="Retarget a single TACO demo (triplet + sequence).")
    pr.add_argument(
        "--base-dir",
        default="/home/ycl/projects/datasets/egoengine/TACO",
        help="TACO root dir (contains Hand_Poses/, Object_Models/, etc.)",
    )
    pr.add_argument(
        "--triplet",
        required=True,
        help="Triplet folder under Hand_Poses, e.g. '(hit, hammer, box)'. QUOTE IT.",
    )
    pr.add_argument(
        "--sequence", required=True, help="Sequence folder under triplet, e.g. '20231002_057'."
    )

    pr.add_argument(
        "--out-json",
        default=None,
        help="Output retarget json path (default: <demo_dir>/retarget.json)",
    )
    pr.add_argument("--visualize", action="store_true", help="Visualize in PyBullet.")
    pr.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH_N, help="Kernel half-window N (window size = 2N+1).")
    pr.add_argument("--object-centric", type=float, default=0.0,
                    help="Threshold distance (m) to snap fingertip to object surface (0 = disabled).")
    pr.add_argument(
        "--do-control",
        action="store_true",
        help="Use control_robot() instead of setJoint per step.",
    )

    return p


def main():
    args = build_parser().parse_args()

    base_dir = os.path.abspath(args.base_dir)
    mesh_dir = os.path.join(base_dir, "Object_Models", "object_models_released")

    demo_dir = os.path.join(base_dir, "Hand_Poses", args.triplet, args.sequence)


    
    '''
    preview_one_frame(
        data_dir = demo_dir,
        mesh_dir = mesh_dir,
        frame    = 0,
        device   = "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    '''
    
    '''
    preview_video(
        data_dir = demo_dir,
        mesh_dir = mesh_dir,
        device   = "cuda:0" if torch.cuda.is_available() else "cpu",
        fps      = 10
    )
    '''

    save_path = args.out_json or os.path.join(demo_dir, "retarget.json")
    retarget_to_robot(
        demo_dir=demo_dir,
        mesh_dir=mesh_dir,
        save_path=save_path,
        skip_viz=(not args.visualize),  # --visualize turns PyBullet on
        do_control=args.do_control,
        smooth_n=args.smooth,
        object_centric=args.object_centric,
    )

    print(f"Retargeting finished! saved to {save_path}")


if __name__ == "__main__":
    main()

'''
python -m retargeting.show_human_new retarget --base-dir /home/ycl/projects/datasets/egoengine/TACO --triplet "(hit, hammer, box)" --sequence 20231002_057 --visualize

python -m retargeting.show_human_new retarget --base-dir /home/ycl/projects/datasets/egoengine/TACO --triplet "(brush, eraser, bowl)" --sequence 20231027_037 --visualize


--filter N: (2N+1 window size) to smooth the hand and object pose sequences
--object-centric T: T is the threshold distance (in meters) to consider the hand is holding the object, and switch to object-centric IK targets during that period.
'''
