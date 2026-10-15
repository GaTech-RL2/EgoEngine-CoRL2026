#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug visualization only (no file save):
- Visualize world / AprilTag / camera trajectory
- Compute hand world points from camera frame
- Print camera, wrist, index positions
"""

import argparse
import json
from pathlib import Path

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def _assert_T(T: np.ndarray, name: str):
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} last row must be [0 0 0 1], got {T[3]}")


def _draw_triad(ax, T: np.ndarray, axis_len: float, label: str):
    o = T[:3, 3]
    R = T[:3, :3]
    X = o + R[:, 0] * axis_len
    Y = o + R[:, 1] * axis_len
    Z = o + R[:, 2] * axis_len
    ax.plot([o[0], X[0]], [o[1], X[1]], [o[2], X[2]], c="r", lw=2)
    ax.plot([o[0], Y[0]], [o[1], Y[1]], [o[2], Y[2]], c="g", lw=2)
    ax.plot([o[0], Z[0]], [o[1], Z[1]], [o[2], Z[2]], c="b", lw=2)
    ax.text(o[0], o[1], o[2], label, color="k")


def _set_equal_3d(ax, P: np.ndarray):
    mn = np.nanmin(P, axis=0)
    mx = np.nanmax(P, axis=0)
    ctr = (mn + mx) / 2.0
    rng = mx - mn
    r = float(np.max(rng) * 1.2) if np.max(rng) > 0 else 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _load_hand_points(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", []) if isinstance(data, dict) else data
    T = len(frames)
    pts = np.full((T, 21, 3), np.nan, dtype=np.float32)
    for t, fr in enumerate(frames):
        if not isinstance(fr, dict):
            continue
        arr = fr.get("right_camera") or fr.get("right_device")
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape == (21, 3):
            pts[t] = arr
    return pts


def _fill_invalid_frames(pts: np.ndarray) -> np.ndarray:
    out = pts.copy()
    T = out.shape[0]
    last_valid = None
    for t in range(T):
        if np.isfinite(out[t]).all():
            last_valid = out[t].copy()
            continue
        if last_valid is not None:
            out[t] = last_valid
            continue
        found = None
        for j in range(t + 1, T):
            if np.isfinite(out[j]).all():
                found = out[j].copy()
                break
        if found is not None:
            out[t] = found
            last_valid = found.copy()
        else:
            out[t] = np.nan_to_num(out[t])
    return out


def _transform_points(cam2w: np.ndarray, P_cam: np.ndarray) -> np.ndarray:
    T, N, _ = P_cam.shape
    ones = np.ones((T, N, 1), dtype=np.float32)
    Ph = np.concatenate([P_cam, ones], axis=-1)
    Pw = np.einsum("tij,tnj->tni", cam2w, Ph)
    return Pw[..., :3]


def _load_object_poses(path: Path) -> np.ndarray:
    poses = np.load(path).astype(np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Bad object pose shape: {poses.shape} from {path}")
    return poses


def _transform_pose_seq_cam2world(seq_cam: np.ndarray, cam2w: np.ndarray) -> np.ndarray:
    Tn = min(len(seq_cam), len(cam2w))
    out = np.empty((Tn, 4, 4), dtype=np.float32)
    for t in range(Tn):
        out[t] = cam2w[t] @ seq_cam[t]
    return out


PALM_IDX = 0
TIP_IDXS = [4, 8, 12, 16, 20]
HAND_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]

# This dataset's camera extrinsics already align with Aria convention.
# Keep identity to avoid double-rotating camera axes.
R_CAM_TO_ARIA = np.eye(3, dtype=np.float32)


def main():
    data_root = Path("/home/ycl/projects/datasets/mustard/data")
    ep = "mustard_1"
    sess = "1"

    ap = argparse.ArgumentParser(description="Visualize camera/tag/hand (no save)")
    ap.add_argument("--tag_ref", default=str(Path(__file__).resolve().parent / "tag_in_world.txt"))
    ap.add_argument(
        "--w2cam",
        default=str(data_root / "Egocentric_Camera_Parameters" / ep / sess / "upd_egocentric_frame_extrinsic.npy"),
    )
    ap.add_argument(
        "--hand_json",
        default=str(data_root / "Hand_Poses" / ep / sess / "all_pose.json"),
    )
    ap.add_argument(
        "--object_npy",
        default=str(data_root / "Object_Poses" / ep / sess / "object.npy"),
    )
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    T_world_tag = np.loadtxt(args.tag_ref).astype(np.float32)
    _assert_T(T_world_tag, "T_world_tag")

    w2cam = np.load(args.w2cam).astype(np.float32)
    if w2cam.ndim != 3 or w2cam.shape[1:] != (4, 4):
        raise ValueError(f"Bad w2cam shape: {w2cam.shape}")
    cam2w = np.linalg.inv(w2cam).astype(np.float32)
    # Optional camera-axis correction (identity by default).
    # Apply on camera rotation only: R_w_cam <- R_w_cam * R_cam_to_aria
    cam2w[:, :3, :3] = np.einsum("tij,jk->tik", cam2w[:, :3, :3], R_CAM_TO_ARIA)

    hand_cam = _fill_invalid_frames(_load_hand_points(Path(args.hand_json)))
    obj_cam = _load_object_poses(Path(args.object_npy))

    Tn = min(len(cam2w), len(hand_cam), len(obj_cam))
    cam2w = cam2w[:Tn]
    hand_cam = hand_cam[:Tn]
    obj_cam = obj_cam[:Tn]
    obj_w = _transform_pose_seq_cam2world(obj_cam, cam2w)

    # Camera->hand transform in Aria camera frame:
    # use identity (no extra rotation).
    CAM_TO_HAND_R = np.eye(3, dtype=np.float32)

    hand_cam_adj = hand_cam @ CAM_TO_HAND_R.T
    hand_w = _transform_points(cam2w, hand_cam_adj)
    hand_plot = hand_w[:, [PALM_IDX, *TIP_IDXS], :]

    cam_pos = cam2w[:, :3, 3]
    obj_pos = obj_w[:, :3, 3]
    wrist = hand_w[:, PALM_IDX, :]
    index_tip = hand_w[:, 8, :]

    print(f"[POSE] tag_world_xyz = {T_world_tag[:3, 3].tolist()}")
    print(f"[DEBUG] R_CAM_TO_ARIA =\n{R_CAM_TO_ARIA}")
    print(f"[DEBUG] CAM_TO_HAND_R =\n{CAM_TO_HAND_R}")
    print(f"[POSE] camera_world_xyz_frame0 = {cam_pos[0].tolist()}")
    print(f"[POSE] camera_world_xyz_last = {cam_pos[-1].tolist()}")
    print(f"[POSE] object_world_xyz_frame0 = {obj_pos[0].tolist()}")
    print(f"[POSE] object_world_xyz_last = {obj_pos[-1].tolist()}")
    print(f"[POSE] wrist_world_xyz_frame0 = {wrist[0].tolist()}")
    print(f"[POSE] wrist_world_xyz_last = {wrist[-1].tolist()}")
    print(f"[POSE] index_world_xyz_frame0 = {index_tip[0].tolist()}")
    print(f"[POSE] index_world_xyz_last = {index_tip[-1].tolist()}")

    all_pts = np.vstack([
        cam_pos,
        obj_pos,
        hand_plot.reshape(-1, 3),
        wrist,
        index_tip,
        T_world_tag[:3, 3][None, :],
        np.zeros((1, 3), dtype=np.float32),
    ])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Debug: World / AprilTag / Camera / Hand (loop)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=args.elev, azim=args.azim)

    _set_equal_3d(ax, all_pts)
    _draw_triad(ax, np.eye(4, dtype=np.float32), axis_len=0.08, label="world")
    _draw_triad(ax, T_world_tag, axis_len=0.08, label="tag")
    _draw_triad(ax, cam2w[0], axis_len=0.06, label="cam@0")
    _draw_triad(ax, obj_w[0], axis_len=0.06, label="obj@0")

    ax.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], c="k", lw=1.2, alpha=0.6, label="camera traj")
    ax.plot(obj_pos[:, 0], obj_pos[:, 1], obj_pos[:, 2], c="tab:red", lw=1.6, alpha=0.8, label="object traj")
    ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2], c="tab:orange", lw=1.8, alpha=0.8, label="wrist traj")
    cam_dot = ax.scatter([], [], [], c="k", s=55, label="camera")
    obj_dot = ax.scatter([], [], [], c="tab:red", s=60, label="object")
    hand_pts = ax.scatter([], [], [], c="tab:blue", s=32, label="hand joints")
    wrist_dot = ax.scatter([], [], [], c="tab:orange", s=65, label="wrist")
    index_dot = ax.scatter([], [], [], c="tab:green", s=65, label="index")
    hand_lines = [ax.plot([], [], [], c="tab:blue", lw=2.2)[0] for _ in HAND_EDGES]
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def _set_scatter_xyz(sc, p):
        sc._offsets3d = ([p[0]], [p[1]], [p[2]])

    def _set_scatter_xyz_many(sc, P):
        sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

    def _update(i: int):
        pcam = cam_pos[i]
        pobj = obj_pos[i]
        phand = hand_plot[i]
        pw = phand[0]
        pi = hand_w[i, 8]

        _set_scatter_xyz(cam_dot, pcam)
        _set_scatter_xyz(obj_dot, pobj)
        _set_scatter_xyz_many(hand_pts, phand)
        _set_scatter_xyz(wrist_dot, pw)
        _set_scatter_xyz(index_dot, pi)

        for k, (a, b) in enumerate(HAND_EDGES):
            seg = phand[[a, b]]
            hand_lines[k].set_data(seg[:, 0], seg[:, 1])
            hand_lines[k].set_3d_properties(seg[:, 2])

        frame_text.set_text(f"frame {i+1}/{Tn}")
        return [cam_dot, obj_dot, hand_pts, wrist_dot, index_dot, frame_text, *hand_lines]

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=Tn,
        interval=max(1, int(1000 / max(1, args.fps))),
        blit=False,
        repeat=True,
    )

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
