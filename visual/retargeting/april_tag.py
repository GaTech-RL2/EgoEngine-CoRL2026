#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple
 

# ---------- drawing helper ----------
def draw_pose_axes(overlay, camera_params, tag_size, pose_R, pose_t, center):
    """Visualize tag’s axes on the image."""
    fx, fy, cx, cy = camera_params
    opoints = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=np.float64) * float(tag_size)

    # Transform axes endpoints from tag frame to camera frame
    R = pose_R.astype(np.float64)
    t = pose_t.reshape(3).astype(np.float64)
    Xc = (R @ opoints.T).T + t  # shape (3,3)

    # Project to pixels using pinhole model
    zs = Xc[:, 2]
    # Avoid division by zero
    zs = np.where(zs == 0, 1e-9, zs)
    us = fx * (Xc[:, 0] / zs) + cx
    vs = fy * (Xc[:, 1] / zs) + cy
    ipoints = np.round(np.stack([us, vs], axis=1)).astype(int)
    center = tuple(np.round(center).astype(int).ravel())

    cv2.line(overlay, center, tuple(ipoints[0].ravel()), (0, 0, 255), 2)   # X (red)
    cv2.line(overlay, center, tuple(ipoints[1].ravel()), (0, 255, 0), 2)   # Y (green)
    cv2.line(overlay, center, tuple(ipoints[2].ravel()), (255, 0, 0), 2)   # Z (blue)

# ---------- intrinsics loader ----------
def load_intrinsics_k(path: str | Path):
    """
    Load a 3x3 camera matrix K from a text file (whitespace-separated).
    Returns (fx, fy, cx, cy) and K.
    """
    K = np.loadtxt(str(path)).astype(float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix in {path}, got shape {K.shape}")
    if not np.isclose(K[2, 2], 1.0):
        raise ValueError(f"K[2,2] should be 1.0, got {K[2,2]}")
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    return fx, fy, cx, cy, K

# ---------- SE(3) helpers ----------
def make_T(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rm
    T[:3, 3]  = t.reshape(3)
    return T

def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def average_quaternions(quats: np.ndarray) -> np.ndarray:
    """
    Average unit quaternions (Nx4) robustly (hemisphere correction then mean+normalize).
    """
    if quats.shape[0] == 1:
        q = quats[0]
        return q / np.linalg.norm(q)
    q_ref = quats[0] / np.linalg.norm(quats[0])
    aligned = []
    for q in quats:
        q = q / np.linalg.norm(q)
        if np.dot(q, q_ref) < 0:  # keep on same hemisphere
            q = -q
        aligned.append(q)
    q_mean = np.mean(np.vstack(aligned), axis=0)
    q_mean /= np.linalg.norm(q_mean)
    return q_mean

# ---------- core: first-N-frame tag pose from video with Z-rotation correction & averaging ----------
def apriltag_from_video_first_n_frames(
    video_path: str,
    camera_params: tuple[float, float, float, float],
    tag_family: str = "tagStandard41h12",
    tag_size_m: float = 0.06 * 5 / 9,
    number_frames: int = 1,
    rotate_cw_deg: float = 0,
    visualize: bool = True,
    outdir: str = "outputs",
):
    """
    Detect AprilTags across the first `number_frames` frames of a video, apply a clockwise
    rotation around the tag's +Z axis by `rotate_cw_deg`, average poses per tag ID, and
    save averaged transforms and visualization.

    Saved per tag (using AVERAGED pose):
      - tag<ID>_T_tag_to_cam.txt            : 4x4 tag→camera (after Z-rotation correction)
      - tag<ID>_T_cam_to_tag.txt            : 4x4 camera→tag (inverse of above)
      - tag<ID>_T_cam_in_tag.txt            : 4x4 camera pose in tag/world (alias of camera→tag)
      - tag<ID>_[R|t]_cam_in_tag_3x4.txt    : 3×4 [R|t] (camera in tag/world)
      - annotated_first_frame.png           : axes drawn using averaged pose(s)
    """
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fx, fy, cx, cy = camera_params
    at_detector = Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # Rotation to apply to the TAG FRAME (clockwise about +Z by rotate_cw_deg).
    # Clockwise in the image corresponds to a NEGATIVE right-hand rotation about +Z.
    Rz_correction = Rotation.from_euler('z', -rotate_cw_deg, degrees=True).as_matrix()
    # Optional reflection of the TAG FRAME's Z axis (mirror across XY plane of the tag)
    Sz_mirror = np.diag([1.0, 1.0, -1.0])

    # Accumulators per tag id
    quat_accum: Dict[int, List[np.ndarray]] = {}
    t_accum: Dict[int, List[np.ndarray]] = {}
    center_last: Dict[int, np.ndarray] = {}
    first_frame = None
    grabbed_frames = 0

    while grabbed_frames < number_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break  # reached end of video
        if first_frame is None:
            first_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size_m
        )

        for det in dets:
            # Native dt_apriltags output: tag -> camera
            R_tag_to_cam = det.pose_R.astype(np.float64)
            t_tag_to_cam = det.pose_t.flatten().astype(np.float64)

            # Apply tag-frame correction:
            #  - rotate tag’s local axes CW about +Z by rotate_cw_deg
            #  - optionally mirror the tag's Z axis (flip normal)
            # Compose on the RIGHT to change the basis of the tag frame.
            R_tag_to_cam_corr = R_tag_to_cam @ Rz_correction
            # (origin unchanged) -> translation remains the same
            t_tag_to_cam_corr = t_tag_to_cam

            # Accumulate (as quaternions) + translations
            q = Rotation.from_matrix(R_tag_to_cam_corr).as_quat()  # [x,y,z,w]
            tid = int(det.tag_id)
            quat_accum.setdefault(tid, []).append(q)
            t_accum.setdefault(tid, []).append(t_tag_to_cam_corr)
            center_last[tid] = det.center.copy()

            # Note: per-frame camera-in-tag poses were previously collected for GIFs.
            # We omit that to keep this script minimal.

        grabbed_frames += 1

    cap.release()

    if first_frame is None:
        raise RuntimeError("Failed to read any frame from the video.")

    # Build overlay on the FIRST frame, using averaged poses
    overlay = first_frame.copy()
    results = []

    for tid in sorted(quat_accum.keys()):
        quats = np.vstack(quat_accum[tid])               # Nx4
        ts = np.vstack(t_accum[tid])                     # Nx3
        q_avg = average_quaternions(quats)               # 4,
        R_avg = Rotation.from_quat(q_avg).as_matrix()    # 3x3
        t_avg = np.mean(ts, axis=0)                      # 3,

        # Build & save transforms
        T_tag_to_cam = make_T(R_avg, t_avg)              # corrected & averaged
        T_cam_in_tag = invert_T(T_tag_to_cam)
        R_cam_in_tag = T_cam_in_tag[:3, :3]
        t_cam_in_tag = T_cam_in_tag[:3, 3]
        Rt_3x4 = np.hstack([R_cam_in_tag, t_cam_in_tag.reshape(3, 1)])

        np.savetxt(os.path.join(outdir, f"tag{tid}_T_tag_to_cam.txt"),
                   T_tag_to_cam, fmt="%.9f")
        # Save camera→tag as well (both a clear name and the existing alias)
        np.savetxt(os.path.join(outdir, f"tag{tid}_T_cam_to_tag.txt"),
                   T_cam_in_tag, fmt="%.9f")
        np.savetxt(os.path.join(outdir, f"tag{tid}_T_cam_in_tag.txt"),
                   T_cam_in_tag, fmt="%.9f")
        np.savetxt(os.path.join(outdir, f"tag{tid}_[R|t]_cam_in_tag_3x4.txt"),
                   Rt_3x4, fmt="%.9f")

        # Draw averaged pose axes on overlay
        draw_pose_axes(overlay, (fx, fy, cx, cy), tag_size_m, R_avg, t_avg, center_last[tid])

        # Outline & label (use the last-seen center as anchor)
        cv2.putText(overlay, f"id:{tid} (avg {len(ts)} frames)",
                    tuple(np.round(center_last[tid]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        results.append({
            "id": tid,
            "R_tag_to_cam_avg": R_avg,
            "t_tag_to_cam_avg": t_avg,
            "T_tag_to_cam": T_tag_to_cam,
            "T_cam_in_tag": T_cam_in_tag
        })

    # Save visualization
    if visualize and len(results) > 0:
        img_path = os.path.join(outdir, "annotated_first_frame.png")
        cv2.imwrite(img_path, overlay)
        print(f"Saved annotated image -> {img_path}")

        # Also save a 3D pose figure for camera and AprilTag(s)
        try:
            save_pose_3d_fig(results, outdir, axis_len=max(0.5*tag_size_m, 0.2))
        except Exception as e:
            print(f"[WARN] Failed to save 3D pose figure: {e}")

        # Animated GIF export removed for minimal script.

    if len(results) == 0:
        print("No tags detected in the first N frames.")

    return results, overlay if visualize else first_frame


# ---------- simple 3D pose figure (camera & tag frames) ----------
def _set_equal_3d(ax, P: np.ndarray, pad=0.1):
    mn = np.nanmin(P, axis=0)
    mx = np.nanmax(P, axis=0)
    ctr = (mn + mx) / 2.0
    rng = (mx - mn)
    r = float(np.max(rng) * (1.0 + pad)) if np.isfinite(rng).all() and np.max(rng) > 0 else 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass


def _draw_triad(ax, origin: np.ndarray, R: np.ndarray, axis_len: float = 0.2, label: str | None = None):
    o = origin.reshape(3)
    X = o + R[:, 0] * axis_len
    Y = o + R[:, 1] * axis_len
    Z = o + R[:, 2] * axis_len
    ax.plot([o[0], X[0]],[o[1], X[1]],[o[2], X[2]], c='red',   lw=2)
    ax.plot([o[0], Y[0]],[o[1], Y[1]],[o[2], Y[2]], c='green', lw=2)
    ax.plot([o[0], Z[0]],[o[1], Z[1]],[o[2], Z[2]], c='blue',  lw=2)
    if label:
        ax.text(o[0], o[1], o[2], label, color='k')


def save_pose_3d_fig(results: List[Dict], outdir: str, axis_len: float = 0.2):
    """
    Save two static 3D figures visualizing poses from the averaged result(s):
      1) In camera coordinates: camera at origin, tag(s) at their measured pose.
      2) In tag coordinates (per tag): tag at origin, camera at its pose in tag.
    """
    if not results:
        return

    # 1) Camera coords: camera at origin with canonical axes; draw each tag triad
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Poses in Camera Coordinates')
    ax.set_xlabel('Xc'); ax.set_ylabel('Yc'); ax.set_zlabel('Zc')

    # camera triad (identity)
    _draw_triad(ax, np.zeros(3), np.eye(3), axis_len, label='cam')

    pts = [np.zeros(3)]
    for d in results:
        R_tc = d['R_tag_to_cam_avg'] if 'R_tag_to_cam_avg' in d else d['T_tag_to_cam'][:3,:3]
        t_tc = d['t_tag_to_cam_avg'] if 't_tag_to_cam_avg' in d else d['T_tag_to_cam'][:3,3]
        _draw_triad(ax, t_tc, R_tc, axis_len, label=f"tag{d['id']}")
        pts.append(t_tc)
    P = np.vstack(pts)
    _set_equal_3d(ax, P, pad=0.2)
    cam_png = os.path.join(outdir, 'pose_3d_in_camera.png')
    fig.tight_layout()
    fig.savefig(cam_png, dpi=160)
    plt.close(fig)
    print(f"Saved 3D pose (camera coords) -> {cam_png}")

    # 2) Per tag: tag coords, camera pose in tag frame
    for d in results:
        T_cam_in_tag = d['T_cam_in_tag']
        R_ct = T_cam_in_tag[:3, :3]
        t_ct = T_cam_in_tag[:3, 3]
        fig2 = plt.figure(figsize=(7,7))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_title(f"Pose in Tag{d['id']} Coordinates")
        ax2.set_xlabel('Xt'); ax2.set_ylabel('Yt'); ax2.set_zlabel('Zt')
        _draw_triad(ax2, np.zeros(3), np.eye(3), axis_len, label=f"tag{d['id']}")
        _draw_triad(ax2, t_ct, R_ct, axis_len, label='cam')
        P2 = np.vstack([np.zeros(3), t_ct])
        _set_equal_3d(ax2, P2, pad=0.2)
        tag_png = os.path.join(outdir, f"pose_3d_in_tag{d['id']}.png")
        fig2.tight_layout()
        fig2.savefig(tag_png, dpi=160)
        plt.close(fig2)
        print(f"Saved 3D pose (tag{d['id']} coords) -> {tag_png}")


# ---------- Animated GIF: camera poses in tag frame ----------
# (Animated GIF utilities removed for minimal script.)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Average AprilTag poses over first N frames with Z-rotation correction.")
    p.add_argument("video", help="Path to video file (e.g., video.mp4)")
    p.add_argument("--intrinsics_k", required=True,
                   help="Path to 3x3 intrinsics text file (K matrix). Required.")
    p.add_argument("--tag_family", default="tagStandard41h12")
    p.add_argument("--tag_size_m", type=float, default=0.06 * 5 / 9)
    p.add_argument("--number_frames", type=int, default=5,
                   help="Use first N frames for averaging (>=1)")
    p.add_argument("--no-viz", action="store_true", help="Disable saving annotated image")
    p.add_argument("--outdir", default="outputs",
                   help="Directory to save annotated image and transforms")
    args = p.parse_args()

    # Intrinsics (required)
    fx, fy, cx, cy, K = load_intrinsics_k(args.intrinsics_k)
    print("Loaded intrinsics K:")
    print(K)

    results, _ = apriltag_from_video_first_n_frames(
        args.video,
        camera_params=(fx, fy, cx, cy),
        tag_family=args.tag_family,
        tag_size_m=args.tag_size_m,
        number_frames=max(1, args.number_frames),
        visualize=not args.no_viz,
        outdir=args.outdir,
    )

    if results:
        # Show a quick text summary for the first averaged tag
        d = results[0]
        print(f"[tag {d['id']}] Averaged over poses")
        print("T_tag_to_cam (4x4):\n", d["T_tag_to_cam"])
        print("T_cam_in_tag (4x4, camera extrinsics in tag/world):\n", d["T_cam_in_tag"])

if __name__ == "__main__":
    main()
