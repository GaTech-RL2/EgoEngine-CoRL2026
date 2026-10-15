#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch preprocess for Aria mustard/human-style layouts (episode/session):
  1) Runs in-code AprilTag detection on each {episode}/{session} to get tag*_T_cam_in_tag.txt
  2) Aligns the whole camera trajectory to world using tag@t0
  3) Saves the 201-style trio:
       - upd_egocentric_frame_extrinsic.npy        (aligned w2c)
       - upd_egocentric_frame_extrinsic_w2c.npy    (same as above, explicit)
       - upd_egocentric_frame_extrinsic_cam2w.npy  (inverse)
  4) (optional) If --viz, renders pose_3d_in_world_first_frame.png and camera_traj_world.gif

Notes for detection:
  - AprilTag ID is hard-coded via HARDCODED_TAG_ID.
  - Frame is always upsampled by UPSAMPLE_FACTOR before detection.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import subprocess
from dt_apriltags import Detector

# ---------- Defaults ----------
_DEF_DATASET_ROOT = os.environ.get("EGOENGINE_DATASET_ROOT", "/home/ycl/projects/datasets/egoengine")
_DEF_CODE_ROOT = os.environ.get("EGOENGINE_CODE_ROOT", "/home/ycl/projects/workspace/egoengine")
DEF_DATA_ROOT = os.path.join(_DEF_DATASET_ROOT, "Aria/mustard/data")
DEF_TAG_REF = os.path.join(_DEF_CODE_ROOT, "retargeting/tag_in_world.txt")

VIDEO_ROOT_NAME = "Egocentric_RGB_Videos"
PARAM_ROOT_NAME = "Egocentric_Camera_Parameters/Egocentric_Camera_Parameters"
PARAM_ROOT_NAMES = (
    "Egocentric_Camera_Parameters/Egocentric_Camera_Parameters",
    "Egocentric_Camera_Parameters",
)

VIDEO_NAME = "cropped_video.mp4"
INTRINSICS_NAME = "egocentric_intrinsic.txt"
EXTRINSICS_NAME = "egocentric_frame_extrinsic.npy"  # raw (usually w2c)

# Hard requirements from user request
HARDCODED_TAG_ID = 13
UPSAMPLE_FACTOR = 2.0
HARDCODED_TAG_FAMILY = "tag36h11"
HARDCODED_TAG_SIZE_M = 0.0635
HARDCODED_NUM_FRAMES = 0  # <=0 means scan all frames

_DETECTOR: Optional[Detector] = None


# ---------- Small utils ----------
def _natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _iter_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    for p in sorted(root.iterdir(), key=lambda x: _natural_key(x.name)):
        if p.is_dir():
            yield p


def _iter_sessions(ep_dir: Path) -> Iterable[Path]:
    if not ep_dir.is_dir():
        return []
    for s in sorted(ep_dir.iterdir(), key=lambda x: _natural_key(x.name)):
        if s.is_dir():
            yield s


def _script_path(rel_name: str) -> str:
    here = Path(__file__).resolve().parent
    return str((here / rel_name).resolve())


def _resolve_param_root(data_root: Path) -> Path:
    for root_name in PARAM_ROOT_NAMES:
        candidate = data_root / root_name
        if candidate.is_dir():
            return candidate
    return data_root / PARAM_ROOT_NAMES[0]


def _assert_T(T: np.ndarray, name: str):
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} last row must be [0 0 0 1], got {T[3]}")


def _pick_cam_in_tag_file(outdir: Path) -> Optional[Path]:
    p1 = outdir / "tag1_T_cam_in_tag.txt"
    if p1.exists():
        return p1
    cands = sorted(outdir.glob("tag*_T_cam_in_tag.txt"))
    return cands[0] if cands else None


def _load_T_txt(path: Path) -> np.ndarray:
    T = np.loadtxt(str(path)).astype(float)
    if T.shape == (3, 4):
        T = np.vstack([T, np.array([0, 0, 0, 1.0])])
    _assert_T(T, str(path))
    return T


def _save_trio(params_dir: Path, cam2w_aligned: np.ndarray):
    if cam2w_aligned.ndim != 3 or cam2w_aligned.shape[1:] != (4, 4):
        raise ValueError(f"cam2w_aligned must be (T,4,4), got {cam2w_aligned.shape}")
    w2c_aligned = np.linalg.inv(cam2w_aligned)
    w2c32 = w2c_aligned.astype("float32")
    cam2w32 = cam2w_aligned.astype("float32")

    main = params_dir / "upd_egocentric_frame_extrinsic.npy"
    np.save(str(main), w2c32)
    np.save(str(params_dir / "upd_egocentric_frame_extrinsic_w2c.npy"), w2c32)
    np.save(str(params_dir / "upd_egocentric_frame_extrinsic_cam2w.npy"), cam2w32)


def _load_intrinsics_k(path: Path) -> Tuple[float, float, float, float]:
    K = np.loadtxt(str(path)).astype(float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K in {path}, got {K.shape}")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _poly_area(corners: np.ndarray) -> float:
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _apply_cam_in_tag_one_transform(T_cam_in_tag: np.ndarray) -> np.ndarray:
    """
    Try one transform only:
      x' = y, y' = -x, z' = z
    """
    R = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    return H @ T_cam_in_tag


def _get_detector(tag_family: str) -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
    return _DETECTOR


def _disarm_detector_for_shutdown() -> None:
    global _DETECTOR
    if _DETECTOR is None:
        return
    # The native dt_apriltags destructor is unstable in this environment.
    # Drop the native pointers and let process exit reclaim memory.
    _DETECTOR.tag_detector_ptr = None
    _DETECTOR.tag_families = {}
    _DETECTOR = None


# ---------- AprilTag step (changed logic only) ----------
def run_apriltag(
    video: Path,
    intrinsics_k: Path,
    outdir: Path,
    number_frames: int,
    tag_family: str,
    tag_size_m: float,
    quiet: bool = False,
) -> bool:
    """
    Detect ONLY HARDCODED_TAG_ID in video, with mandatory 2x upsampling.
    Save camera-in-tag pose as tag{ID}_T_cam_in_tag.txt and alias tag1_T_cam_in_tag.txt.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if not video.exists() or not intrinsics_k.exists():
        if not quiet:
            print(f"[APRILTAG][MISS] video or intrinsics missing: {video} | {intrinsics_k}")
        return False

    fx, fy, cx, cy = _load_intrinsics_k(intrinsics_k)
    detector = _get_detector(tag_family)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        if not quiet:
            print(f"[APRILTAG][MISS] cannot open video: {video}")
        return False

    if not quiet:
        frame_msg = "all" if number_frames <= 0 else str(max(1, number_frames))
        intr_up = (
            fx * UPSAMPLE_FACTOR,
            fy * UPSAMPLE_FACTOR,
            cx * UPSAMPLE_FACTOR,
            cy * UPSAMPLE_FACTOR,
        )
        print(
            f"[APRILTAG] video={video} frames={frame_msg} family={tag_family} "
            f"tag_id={HARDCODED_TAG_ID} tag_size_m={tag_size_m} upsample={UPSAMPLE_FACTOR}"
        )
        print(f"[APRILTAG] intrinsics_k=(fx={fx}, fy={fy}, cx={cx}, cy={cy})")
        print(
            f"[APRILTAG] intrinsics_k_up=(fx={intr_up[0]}, fy={intr_up[1]}, "
            f"cx={intr_up[2]}, cy={intr_up[3]})"
        )

    best_area = -1.0
    best_pose = None
    best_frame = -1

    checked = 0
    max_frames = number_frames if number_frames > 0 else 10**12

    try:
        while checked < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            checked += 1

            h, w = frame.shape[:2]
            up_w = int(round(w * UPSAMPLE_FACTOR))
            up_h = int(round(h * UPSAMPLE_FACTOR))
            frame_up = cv2.resize(frame, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame_up, cv2.COLOR_BGR2GRAY)

            intr_up = (
                fx * UPSAMPLE_FACTOR,
                fy * UPSAMPLE_FACTOR,
                cx * UPSAMPLE_FACTOR,
                cy * UPSAMPLE_FACTOR,
            )

            try:
                dets = detector.detect(gray, intrinsics=intr_up, tag_size=tag_size_m)
            except TypeError:
                dets = detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=intr_up,
                    tag_size=tag_size_m,
                )

            if not dets:
                continue

            # Hard-coded ID filter
            wanted = [d for d in dets if int(getattr(d, "tag_id", -1)) == HARDCODED_TAG_ID]
            if not wanted:
                continue

            # Pick strongest observation of the wanted ID
            for d in wanted:
                c = np.asarray(getattr(d, "corners", []), dtype=float)
                if c.shape != (4, 2):
                    continue
                area = _poly_area(c)
                if area > best_area:
                    pose = np.eye(4, dtype=float)
                    pose[:3, :3] = d.pose_R
                    pose[:3, 3:4] = d.pose_t
                    best_area = area
                    best_pose = pose
                    best_frame = checked
    finally:
        cap.release()

    if best_pose is None:
        if not quiet:
            print(f"[APRILTAG][MISS] no detection for tag id {HARDCODED_TAG_ID}")
        return False

    cam_in_tag = np.linalg.inv(best_pose)
    cam_in_tag = _apply_cam_in_tag_one_transform(cam_in_tag)
    out_id = outdir / f"tag{HARDCODED_TAG_ID}_T_cam_in_tag.txt"
    np.savetxt(str(out_id), cam_in_tag, fmt="%.9f")

    # Keep compatibility with scripts that prioritize tag1 file.
    np.savetxt(str(outdir / "tag1_T_cam_in_tag.txt"), cam_in_tag, fmt="%.9f")

    if not quiet:
        print(
            f"[APRILTAG] OK tag_id={HARDCODED_TAG_ID} best_frame={best_frame} "
            f"best_area={best_area:.2f} -> {out_id}"
        )
    return True


# ---------- Alignment (same as batch) ----------
def align_and_save_trio(
    tag_ref_path: Path,
    cam_in_tag_path: Path,
    extrinsics_raw_path: Path,
    extrinsics_is_w2c: bool,
    params_dir: Path,
    quiet: bool = False,
) -> np.ndarray | None:
    T_world_tag = _load_T_txt(tag_ref_path)
    T_tag_cam0 = _load_T_txt(cam_in_tag_path)

    if not extrinsics_raw_path.exists():
        if not quiet:
            print(f"[ALIGN][MISS] extrinsics not found: {extrinsics_raw_path}")
        return None
    seq = np.load(str(extrinsics_raw_path)).astype(float)
    if seq.ndim != 3 or seq.shape[1:] != (4, 4):
        raise ValueError(f"Bad extrinsics shape {seq.shape} in {extrinsics_raw_path}")

    cam2w_raw = np.linalg.inv(seq) if extrinsics_is_w2c else seq
    T_world_cam0 = T_world_tag @ T_tag_cam0
    H = T_world_cam0 @ np.linalg.inv(cam2w_raw[0])
    cam2w_aligned = np.einsum("ij,tjk->tik", H, cam2w_raw)

    _save_trio(params_dir, cam2w_aligned)
    if not quiet:
        print(f"[ALIGN] Saved trio -> {params_dir}")
    return cam2w_aligned


# ---------- Visualization (same as batch) ----------
def run_visualize(
    tag_ref: Path,
    cam_in_tag: Path,
    extrinsics_path: Path,
    out_png: Path,
    gif_out: Path,
    fps: int,
    elev: float,
    azim: float,
    axis_len: float,
    extrinsics_are_w2c: bool,
) -> bool:
    script = _script_path("visualize_cam_and_tag_in_ref.py")
    cmd = [
        sys.executable,
        script,
        "--tag_ref",
        str(tag_ref),
        "--cam_in_tag",
        str(cam_in_tag),
        "--out",
        str(out_png),
        "--gif_out",
        str(gif_out),
        "--fps",
        str(fps),
        "--elev",
        str(elev),
        "--azim",
        str(azim),
        "--axis_len",
        str(axis_len),
        "--extrinsics",
        str(extrinsics_path),
    ]
    if extrinsics_are_w2c:
        cmd.append("--extrinsics_is_w2c")
    print("[VIZ]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[VIZ][ERROR {res.returncode}] {extrinsics_path}")
        if res.stdout.strip():
            print("STDOUT:\n" + res.stdout)
        if res.stderr.strip():
            print("STDERR:\n" + res.stderr)
        return False
    print(f"[VIZ] Wrote {out_png.name} and {gif_out.name} in {out_png.parent}")
    return True


# ---------- Discovery & Driver ----------
def discover_pairs(
    data_root: Path,
    episodes_filter: Optional[List[str]] = None,
    sess_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    video_root = data_root / VIDEO_ROOT_NAME
    param_root = _resolve_param_root(data_root)
    if not video_root.is_dir() or not param_root.is_dir():
        return []

    eps_v = {p.name for p in _iter_dirs(video_root)}
    eps_p = {p.name for p in _iter_dirs(param_root)}
    eps_common = sorted(eps_v & eps_p, key=_natural_key)
    if episodes_filter:
        allow = set(episodes_filter)
        eps_common = [e for e in eps_common if e in allow]

    pairs: List[Tuple[str, str]] = []
    for ep in eps_common:
        sess_v = {s.name for s in _iter_sessions(video_root / ep)}
        sess_p = {s.name for s in _iter_sessions(param_root / ep)}
        sess_common = sorted(sess_v & sess_p, key=_natural_key)
        if sess_filter:
            allow_s = set(sess_filter)
            sess_common = [s for s in sess_common if s in allow_s]
        for sess in sess_common:
            pairs.append((ep, sess))
    return pairs


def process_one(
    data_root: Path,
    tag_ref: Path,
    ep: str,
    sess: str,
    extrinsics_is_w2c: bool,
    skip_existing: bool,
    do_viz: bool,
    viz_fps: int,
    viz_elev: float,
    viz_azim: float,
    viz_axis_len: float,
) -> bool:
    params_dir = _resolve_param_root(data_root) / ep / sess
    video = data_root / VIDEO_ROOT_NAME / ep / sess / VIDEO_NAME
    intr_k = params_dir / INTRINSICS_NAME
    extr_raw = params_dir / EXTRINSICS_NAME

    upd_main = params_dir / "upd_egocentric_frame_extrinsic.npy"
    upd_w2c = params_dir / "upd_egocentric_frame_extrinsic_w2c.npy"
    upd_cam2w = params_dir / "upd_egocentric_frame_extrinsic_cam2w.npy"

    if skip_existing and upd_main.exists() and upd_w2c.exists() and upd_cam2w.exists():
        print(f"[SKIP] {ep}/{sess} (trio already exists)")
        if do_viz:
            cam_in_tag = _pick_cam_in_tag_file(params_dir)
            if cam_in_tag is None:
                print(f"[VIZ][SKIP] no tag*_T_cam_in_tag.txt in {params_dir}")
                return True
            out_png = params_dir / "pose_3d_in_world_first_frame.png"
            gif_out = params_dir / "camera_traj_world.gif"
            run_visualize(
                tag_ref,
                cam_in_tag,
                upd_cam2w,
                out_png,
                gif_out,
                viz_fps,
                viz_elev,
                viz_azim,
                viz_axis_len,
                extrinsics_are_w2c=False,
            )
        return True

    missing = []
    if not video.exists():
        missing.append(str(video))
    if not intr_k.exists():
        missing.append(str(intr_k))
    if not extr_raw.exists():
        missing.append(str(extr_raw))
    if not tag_ref.exists():
        missing.append(str(tag_ref))
    if missing:
        print(f"[MISS] {ep}/{sess} missing:\n  - " + "\n  - ".join(missing))
        return False

    # Step 1: AprilTag detection (changed logic)
    ok1 = run_apriltag(
        video=video,
        intrinsics_k=intr_k,
        outdir=params_dir,
        number_frames=HARDCODED_NUM_FRAMES,
        tag_family=HARDCODED_TAG_FAMILY,
        tag_size_m=HARDCODED_TAG_SIZE_M,
        quiet=False,
    )
    if not ok1:
        return False

    cam_in_tag = _pick_cam_in_tag_file(params_dir)
    if cam_in_tag is None:
        print(f"[WARN] {ep}/{sess}: no tag*_T_cam_in_tag.txt found in {params_dir}")
        return False

    # Step 2: align & save trio
    cam2w_aligned = align_and_save_trio(
        tag_ref_path=tag_ref,
        cam_in_tag_path=cam_in_tag,
        extrinsics_raw_path=extr_raw,
        extrinsics_is_w2c=extrinsics_is_w2c,
        params_dir=params_dir,
        quiet=False,
    )
    if cam2w_aligned is None:
        return False

    # Optional Step 3: visualization
    if do_viz:
        out_png = params_dir / "pose_3d_in_world_first_frame.png"
        gif_out = params_dir / "camera_traj_world.gif"
        ok_v = run_visualize(
            tag_ref,
            cam_in_tag,
            params_dir / "upd_egocentric_frame_extrinsic_cam2w.npy",
            out_png,
            gif_out,
            viz_fps,
            viz_elev,
            viz_azim,
            viz_axis_len,
            extrinsics_are_w2c=False,
        )
        if not ok_v:
            return False

    return True


# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch single-tag april_tag + alignment with optional visualization.")
    ap.add_argument(
        "--data-root",
        default=DEF_DATA_ROOT,
        help="Root containing Egocentric_RGB_Videos/ and Egocentric_Camera_Parameters/",
    )
    ap.add_argument(
        "--tag-ref",
        default=DEF_TAG_REF,
        help="Path to tag_in_world.txt (4x4 T_world_tag)",
    )
    ap.add_argument("--episodes", nargs="*", type=str, help="Only these episode names. Default: auto-discover.")
    ap.add_argument("--sessions", nargs="*", type=str, help="Only these sessions (e.g., 1). Default: auto-discover.")
    ap.add_argument(
        "--extrinsics-is-w2c",
        action="store_true",
        default=True,
        help="Raw extrinsics are world->camera (invert internally). Default True.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip if all three updated extrinsics already exist.")

    ap.add_argument("--viz", action="store_true", help="Also render PNG + GIF per sequence.")
    ap.add_argument("--viz-fps", type=int, default=20)
    ap.add_argument("--viz-elev", type=float, default=25.0)
    ap.add_argument("--viz-azim", type=float, default=-60.0)
    ap.add_argument("--viz-axis-len", type=float, default=0.15)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    tag_ref = Path(args.tag_ref).resolve()

    pairs = discover_pairs(
        data_root=data_root,
        episodes_filter=args.episodes,
        sess_filter=args.sessions,
    )
    if not pairs:
        print("No valid {episode}/{session} pairs under:", data_root)
        print(
            f"Expected:\n  {VIDEO_ROOT_NAME}/<episode>/<session>/{VIDEO_NAME}\n"
            "  one of:\n"
            + "\n".join(
                f"    {root_name}/<episode>/<session>/{INTRINSICS_NAME} & {EXTRINSICS_NAME}"
                for root_name in PARAM_ROOT_NAMES
            )
        )
        return 1

    print(f"Found {len(pairs)} pairs -> first {pairs[0]} last {pairs[-1]}")
    print(
        f"Using hardcoded tag id={HARDCODED_TAG_ID}, family={HARDCODED_TAG_FAMILY}, "
        f"tag_size_m={HARDCODED_TAG_SIZE_M}, frames={HARDCODED_NUM_FRAMES}, "
        f"upsample={UPSAMPLE_FACTOR}x"
    )
    _get_detector(HARDCODED_TAG_FAMILY)

    ok = fail = 0
    for ep, sess in pairs:
        print(f"\n=== {ep}/{sess} ===")
        if process_one(
            data_root=data_root,
            tag_ref=tag_ref,
            ep=ep,
            sess=sess,
            extrinsics_is_w2c=args.extrinsics_is_w2c,
            skip_existing=args.skip_existing,
            do_viz=args.viz,
            viz_fps=args.viz_fps,
            viz_elev=args.viz_elev,
            viz_azim=args.viz_azim,
            viz_axis_len=args.viz_axis_len,
        ):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. ok={ok}, fail={fail}, total={len(pairs)}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        _disarm_detector_for_shutdown()
