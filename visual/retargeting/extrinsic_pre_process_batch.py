#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch preprocess for Aria:
  1) Runs retargeting/april_tag.py on each {ID}/{session} to get tag*_T_cam_in_tag.txt
  2) Aligns the whole camera trajectory to world using tag@t0
  3) Saves the 201-style trio:
       - upd_egocentric_frame_extrinsic.npy        (aligned w2c)
       - upd_egocentric_frame_extrinsic_w2c.npy    (same as above, explicit)
       - upd_egocentric_frame_extrinsic_cam2w.npy  (inverse)
  4) (optional) If --viz, renders pose_3d_in_world_first_frame.png and camera_traj_world.gif

Usage (from repo root):
  # no viz
  python -m retargeting.extrinsic_pre_process_batch \
    --tag-ref /home/ycl/projects/workspace/egoengine/retargeting/tag_in_world.txt \
    --number-frames 1 --skip-existing

  # with viz
  python -m retargeting.extrinsic_pre_process_batch \
    --tag-ref /home/ycl/projects/workspace/egoengine/retargeting/tag_in_world.txt \
    --number-frames 1 --viz
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import subprocess

# ---------- Defaults ----------
_DEF_DATASET_ROOT = os.environ.get("EGOENGINE_DATASET_ROOT", "/home/ycl/projects/datasets/egoengine")
_DEF_CODE_ROOT = os.environ.get("EGOENGINE_CODE_ROOT", "/home/ycl/projects/workspace/egoengine")
DEF_DATA_ROOT = os.path.join(_DEF_DATASET_ROOT, "Aria/mustard/data")
DEF_TAG_REF = os.path.join(_DEF_CODE_ROOT, "retargeting/tag_in_world.txt")

VIDEO_ROOT_NAME   = "Egocentric_RGB_Videos"
PARAM_ROOT_NAME   = "Egocentric_Camera_Parameters"

VIDEO_NAME        = "cropped_video.mp4"
INTRINSICS_NAME   = "egocentric_intrinsic.txt"
EXTRINSICS_NAME   = "egocentric_frame_extrinsic.npy"   # raw (usually w2c)

# ---------- Small utils ----------
def _iter_numeric_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    for p in sorted(root.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 10**9):
        if p.is_dir() and p.name.isdigit():
            yield p

def _iter_sessions(id_dir: Path) -> Iterable[Path]:
    if not id_dir.is_dir():
        return []
    for s in sorted(id_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 10**9):
        if s.is_dir() and s.name.isdigit():
            yield s

def _script_path(rel_name: str) -> str:
    """Resolve 'retargeting/april_tag.py' or 'retargeting/visualize_cam_and_tag_in_ref.py' relative to THIS file."""
    here = Path(__file__).resolve().parent
    return str((here / rel_name).resolve())

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
    if T.shape == (3, 4):  # support 3x4 Rt if it sneaks in; upgrade to 4x4
        T = np.vstack([T, np.array([0, 0, 0, 1.0])])
    _assert_T(T, str(path))
    return T

def _save_trio(params_dir: Path, cam2w_aligned: np.ndarray):
    """Save aligned extrinsics in three flavors, float32."""
    if cam2w_aligned.ndim != 3 or cam2w_aligned.shape[1:] != (4, 4):
        raise ValueError(f"cam2w_aligned must be (T,4,4), got {cam2w_aligned.shape}")
    w2c_aligned = np.linalg.inv(cam2w_aligned)
    w2c32 = w2c_aligned.astype("float32")
    cam2w32 = cam2w_aligned.astype("float32")

    main = params_dir / "upd_egocentric_frame_extrinsic.npy"        # w2c (main)
    np.save(str(main), w2c32)
    np.save(str(params_dir / "upd_egocentric_frame_extrinsic_w2c.npy"), w2c32)
    np.save(str(params_dir / "upd_egocentric_frame_extrinsic_cam2w.npy"), cam2w32)

# ---------- AprilTag step (NO viz) ----------
def run_apriltag(video: Path, intrinsics_k: Path, outdir: Path, number_frames: int,
                 tag_family: str = "tagStandard41h12", tag_size_m: float = 0.06 * 5 / 9,
                 rotate_cw_deg: float = 0.0, quiet: bool = False) -> bool:
    outdir.mkdir(parents=True, exist_ok=True)
    script = _script_path("april_tag.py")
    cmd = [
        sys.executable, script, str(video),
        "--intrinsics_k", str(intrinsics_k),
        "--number_frames", str(max(1, number_frames)),
        "--outdir", str(outdir),
        "--tag_family", tag_family,
        "--tag_size_m", str(tag_size_m),
        "--no-viz",               # <- no PNGs from april_tag.py
    ]
    if rotate_cw_deg != 0:
        cmd += ["--rotate_cw_deg", str(rotate_cw_deg)]
    if not quiet:
        print("[APRILTAG]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        if not quiet:
            print(f"[APRILTAG][ERROR {res.returncode}] {video}")
            if res.stdout.strip(): print("STDOUT:\n" + res.stdout)
            if res.stderr.strip(): print("STDERR:\n" + res.stderr)
        return False
    if not quiet:
        print(f"[APRILTAG] OK -> {outdir}")
    return True

# ---------- Alignment (in-code; NO images/GIFs) ----------
def align_and_save_trio(
    tag_ref_path: Path, cam_in_tag_path: Path, extrinsics_raw_path: Path,
    extrinsics_is_w2c: bool, params_dir: Path, quiet: bool = False
) -> np.ndarray | None:
    # Load transforms
    T_world_tag = _load_T_txt(tag_ref_path)
    T_tag_cam0  = _load_T_txt(cam_in_tag_path)

    # Load raw extrinsics sequence
    if not extrinsics_raw_path.exists():
        if not quiet:
            print(f"[ALIGN][MISS] extrinsics not found: {extrinsics_raw_path}")
        return None
    seq = np.load(str(extrinsics_raw_path)).astype(float)
    if seq.ndim != 3 or seq.shape[1:] != (4, 4):
        raise ValueError(f"Bad extrinsics shape {seq.shape} in {extrinsics_raw_path}")

    cam2w_raw = np.linalg.inv(seq) if extrinsics_is_w2c else seq
    # First-frame camera pose in world from tag reference
    T_world_cam0 = T_world_tag @ T_tag_cam0
    # Align trajectory so that frame0 matches the tag-anchored world
    H = T_world_cam0 @ np.linalg.inv(cam2w_raw[0])  # (4,4)
    cam2w_aligned = np.einsum("ij,tjk->tik", H, cam2w_raw)  # (T,4,4)

    _save_trio(params_dir, cam2w_aligned)
    if not quiet:
        print(f"[ALIGN] Saved trio -> {params_dir}")
    return cam2w_aligned

# ---------- Visualization (optional) ----------
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
        sys.executable, script,
        "--tag_ref", str(tag_ref),
        "--cam_in_tag", str(cam_in_tag),
        "--out", str(out_png),
        "--gif_out", str(gif_out),
        "--fps", str(fps),
        "--elev", str(elev),
        "--azim", str(azim),
        "--axis_len", str(axis_len),
        "--extrinsics", str(extrinsics_path),
    ]
    if extrinsics_are_w2c:
        cmd.append("--extrinsics_is_w2c")
    print("[VIZ]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[VIZ][ERROR {res.returncode}] {extrinsics_path}")
        if res.stdout.strip(): print("STDOUT:\n" + res.stdout)
        if res.stderr.strip(): print("STDERR:\n" + res.stderr)
        return False
    print(f"[VIZ] Wrote {out_png.name} and {gif_out.name} in {out_png.parent}")
    return True

# ---------- Discovery & Driver ----------
def discover_pairs(
    data_root: Path, ids_filter: Optional[List[int]] = None, sess_filter: Optional[List[int]] = None
) -> List[Tuple[str, str]]:
    video_root = data_root / VIDEO_ROOT_NAME
    param_root = data_root / PARAM_ROOT_NAME
    if not video_root.is_dir() or not param_root.is_dir():
        return []
    ids_v = {p.name for p in _iter_numeric_dirs(video_root)}
    ids_p = {p.name for p in _iter_numeric_dirs(param_root)}
    ids_common = sorted(ids_v & ids_p, key=lambda x: int(x))
    if ids_filter:
        allow = {str(i) for i in ids_filter}
        ids_common = [i for i in ids_common if i in allow]

    pairs: List[Tuple[str, str]] = []
    for id_str in ids_common:
        sess_v = {s.name for s in _iter_sessions(video_root / id_str)}
        sess_p = {s.name for s in _iter_sessions(param_root / id_str)}
        sess_common = sorted(sess_v & sess_p, key=lambda x: int(x))
        if sess_filter:
            allow_s = {str(s) for s in sess_filter}
            sess_common = [s for s in sess_common if s in allow_s]
        for sess in sess_common:
            pairs.append((id_str, sess))
    return pairs

def process_one(
    data_root: Path,
    tag_ref: Path,
    id_str: str,
    sess_str: str,
    number_frames: int,
    rotate_cw_deg: float,
    extrinsics_is_w2c: bool,
    skip_existing: bool,
    do_viz: bool,
    viz_fps: int,
    viz_elev: float,
    viz_azim: float,
    viz_axis_len: float,
) -> bool:
    params_dir = data_root / PARAM_ROOT_NAME / id_str / sess_str
    video      = data_root / VIDEO_ROOT_NAME / id_str / sess_str / VIDEO_NAME
    intr_k     = params_dir / INTRINSICS_NAME
    extr_raw   = params_dir / EXTRINSICS_NAME

    # Trio targets
    upd_main   = params_dir / "upd_egocentric_frame_extrinsic.npy"
    upd_w2c    = params_dir / "upd_egocentric_frame_extrinsic_w2c.npy"
    upd_cam2w  = params_dir / "upd_egocentric_frame_extrinsic_cam2w.npy"

    if skip_existing and upd_main.exists() and upd_w2c.exists() and upd_cam2w.exists():
        print(f"[SKIP] {id_str}/{sess_str} (trio already exists)")
        # still optionally run viz from existing files
        if do_viz:
            cam_in_tag = _pick_cam_in_tag_file(params_dir)
            if cam_in_tag is None:
                print(f"[VIZ][SKIP] no tag*_T_cam_in_tag.txt in {params_dir}")
                return True
            out_png = params_dir / "pose_3d_in_world_first_frame.png"
            gif_out = params_dir / "camera_traj_world.gif"
            # use the already-aligned cam2w for clean viz (no re-alignment drift)
            run_visualize(tag_ref, cam_in_tag, upd_cam2w, out_png, gif_out,
                          viz_fps, viz_elev, viz_azim, viz_axis_len,
                          extrinsics_are_w2c=False)
        return True

    missing = []
    if not video.exists():    missing.append(str(video))
    if not intr_k.exists():   missing.append(str(intr_k))
    if not extr_raw.exists(): missing.append(str(extr_raw))
    if not tag_ref.exists():  missing.append(str(tag_ref))
    if missing:
        print(f"[MISS] {id_str}/{sess_str} missing:\n  - " + "\n  - ".join(missing))
        return False

    # Step 1: april_tag (no viz)
    ok1 = run_apriltag(
        video=video,
        intrinsics_k=intr_k,
        outdir=params_dir,             # write outputs in targeted folder
        number_frames=number_frames,
        rotate_cw_deg=rotate_cw_deg,
        quiet=False
    )
    if not ok1:
        return False

    cam_in_tag = _pick_cam_in_tag_file(params_dir)
    if cam_in_tag is None:
        print(f"[WARN] {id_str}/{sess_str}: no tag*_T_cam_in_tag.txt found in {params_dir}")
        return False

    # Step 2: align & save trio
    cam2w_aligned = align_and_save_trio(
        tag_ref_path=tag_ref,
        cam_in_tag_path=cam_in_tag,
        extrinsics_raw_path=extr_raw,
        extrinsics_is_w2c=extrinsics_is_w2c,
        params_dir=params_dir,
        quiet=False
    )
    if cam2w_aligned is None:
        return False

    # Optional Step 3: visualization (PNG + GIF)
    if do_viz:
        out_png = params_dir / "pose_3d_in_world_first_frame.png"
        gif_out = params_dir / "camera_traj_world.gif"
        # visualize using the aligned cam2w we just saved (no extra alignment surprise)
        ok_v = run_visualize(tag_ref, cam_in_tag, params_dir / "upd_egocentric_frame_extrinsic_cam2w.npy",
                             out_png, gif_out, viz_fps, viz_elev, viz_azim, viz_axis_len,
                             extrinsics_are_w2c=False)
        if not ok_v:
            return False

    return True

# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch april_tag + alignment with optional visualization.")
    ap.add_argument("--data-root", default=DEF_DATA_ROOT,
                    help="Root containing Egocentric_RGB_Videos/ and Egocentric_Camera_Parameters/")
    ap.add_argument("--tag-ref", default=DEF_TAG_REF,
                    help="Path to tag_in_world.txt (4x4 T_world_tag)")
    ap.add_argument("--ids", nargs="*", type=int,
                    help="Only these IDs (e.g., 201 202 ...). Default: auto-discover.")
    ap.add_argument("--sessions", nargs="*", type=int,
                    help="Only these sessions (e.g., 1 2). Default: auto-discover.")
    ap.add_argument("--number-frames", type=int, default=1,
                    help="Use first N frames for AprilTag averaging.")
    ap.add_argument("--rotate-cw-deg", type=float, default=0.0,
                    help="Rotate tag axes clockwise in image by this many degrees (about +Z).")
    ap.add_argument("--extrinsics-is-w2c", action="store_true", default=True,
                    help="Raw extrinsics are world->camera (invert internally). Default True.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if all three updated extrinsics already exist.")

    # Visualization toggles
    ap.add_argument("--viz", action="store_true", help="Also render PNG + GIF per sequence.")
    ap.add_argument("--viz-fps", type=int, default=20)
    ap.add_argument("--viz-elev", type=float, default=25.0)
    ap.add_argument("--viz-azim", type=float, default=-60.0)
    ap.add_argument("--viz-axis-len", type=float, default=0.15)
    return ap

def main():
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    tag_ref   = Path(args.tag_ref).resolve()

    pairs = discover_pairs(
        data_root=data_root,
        ids_filter=args.ids,
        sess_filter=args.sessions
    )
    if not pairs:
        print("No valid {ID}/{session} pairs under:", data_root)
        print(f"Expected:\n  {VIDEO_ROOT_NAME}/<ID>/<session>/{VIDEO_NAME}\n"
              f"  {PARAM_ROOT_NAME}/<ID>/<session>/{INTRINSICS_NAME} & {EXTRINSICS_NAME}")
        sys.exit(1)

    print(f"Found {len(pairs)} pairs → first {pairs[0]}  last {pairs[-1]}")
    ok = fail = 0
    for id_str, sess_str in pairs:
        print(f"\n=== {id_str}/{sess_str} ===")
        if process_one(
            data_root=data_root,
            tag_ref=tag_ref,
            id_str=id_str,
            sess_str=sess_str,
            number_frames=args.number_frames,
            rotate_cw_deg=args.rotate_cw_deg,
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
    sys.exit(0 if fail == 0 else 2)

if __name__ == "__main__":
    main()
