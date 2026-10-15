#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DATA_ROOT = Path("/home/ycl/projects/datasets/mustard/data")


@dataclass
class DemoPaths:
    demo_name: str
    session_name: str
    video_dir: Path
    camera_dir: Path
    hand_dir: Path
    object_dir: Path
    uv_dir: Path
    removed_dir: Path

    @property
    def display_name(self) -> str:
        return f"{self.demo_name}/{self.session_name}"


def _iter_demos(data_root: Path) -> list[DemoPaths]:
    task_name = _infer_task_name(data_root)
    return _iter_demos_for_task(data_root, task_name)


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _infer_task_name(data_root: Path) -> str | None:
    if data_root.name == "data" and data_root.parent.name:
        return data_root.parent.name
    return data_root.name or None


def _normalize_data_root(path: Path) -> Path:
    if (path / "Egocentric_RGB_Videos").exists():
        return path
    nested_data = path / "data"
    if (nested_data / "Egocentric_RGB_Videos").exists():
        return nested_data
    return path


def _resolve_camera_dir(data_root: Path, demo_name: str, session_name: str) -> Path:
    candidates = [
        data_root / "Egocentric_Camera_Parameters" / demo_name / session_name,
        data_root
        / "Egocentric_Camera_Parameters"
        / "Egocentric_Camera_Parameters"
        / demo_name
        / session_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _iter_demos_for_task(data_root: Path, task_name: str | None) -> list[DemoPaths]:
    demos: list[DemoPaths] = []
    egocentric_root = data_root / "Egocentric_RGB_Videos"
    demo_glob = f"{task_name}_*/1/cropped_video.mp4" if task_name else "*/1/cropped_video.mp4"
    for video_path in sorted(egocentric_root.glob(demo_glob), key=_natural_key):
        session_dir = video_path.parent
        demo_name = session_dir.parent.name
        session_name = session_dir.name
        demos.append(
            DemoPaths(
                demo_name=demo_name,
                session_name=session_name,
                video_dir=session_dir,
                camera_dir=_resolve_camera_dir(data_root, demo_name, session_name),
                hand_dir=data_root / "Hand_Poses" / demo_name / session_name,
                object_dir=data_root / "Object_Poses" / demo_name / session_name,
                uv_dir=data_root / "UV" / demo_name / session_name,
                removed_dir=data_root / "Removed" / demo_name / session_name,
            )
        )
    return demos


def _video_info(video_path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return frame_count, width, height, fps


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def _draw_overlay(frame: np.ndarray, title: str, frame_idx: int, frame_count: int) -> np.ndarray:
    canvas = frame.copy()
    lines = [
        title,
        f"frame: {frame_idx}/{max(frame_count - 1, 0)}",
        "a: back  d: forward  s: cut before this frame  n: next video  q: quit",
    ]
    y = 36
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return canvas


def _rewrite_video(path: Path, start_idx: int) -> None:
    if not path.exists():
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if start_idx < 0 or start_idx >= frame_count:
        cap.release()
        raise ValueError(f"Invalid cut index {start_idx} for {path} with {frame_count} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=str(path.parent)) as tmp:
        tmp_path = Path(tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=str(path.parent)) as tmp_final:
        final_tmp_path = Path(tmp_final.name)

    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    _transcode_mp4_compatible(tmp_path, final_tmp_path)
    tmp_path.unlink(missing_ok=True)
    shutil.move(str(final_tmp_path), str(path))


def _transcode_mp4_compatible(src_path: Path, dst_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_path} -> {dst_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _rewrite_png_sequence(seq_dir: Path, start_idx: int, target_count: int | None = None) -> None:
    if not seq_dir.exists() or not seq_dir.is_dir():
        return
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if not frames:
        return
    if start_idx < 0 or start_idx >= len(frames):
        raise ValueError(f"Invalid cut index {start_idx} for {seq_dir} with {len(frames)} frames")

    kept = frames[start_idx:]
    if target_count is not None:
        kept = kept[:target_count]
    tmp_dir = seq_dir.parent / f"{seq_dir.name}__tmp_cut"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    for new_idx, src in enumerate(kept):
        dst = tmp_dir / f"{new_idx:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    shutil.rmtree(seq_dir)
    tmp_dir.rename(seq_dir)


def _rewrite_json_frames(path: Path, start_idx: int) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if start_idx < 0 or start_idx >= len(frames):
        raise ValueError(f"Invalid cut index {start_idx} for {path} with {len(frames)} frames")
    trimmed = frames[start_idx:]
    for i, frame in enumerate(trimmed, start=1):
        if isinstance(frame, dict) and "frame_idx" in frame:
            frame["frame_idx"] = i
    data["frames"] = trimmed
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _rewrite_npy(path: Path, start_idx: int) -> None:
    if not path.exists():
        return
    arr = np.load(path)
    if arr.ndim == 0:
        return
    if start_idx < 0 or start_idx >= arr.shape[0]:
        raise ValueError(f"Invalid cut index {start_idx} for {path} with shape {arr.shape}")
    np.save(path, arr[start_idx:])


def _rewrite_frame_zero_png(video_path: Path, png_path: Path) -> None:
    if not video_path.exists():
        return
    frame = _read_frame(video_path, 0)
    cv2.imwrite(str(png_path), frame)


def _cut_demo(paths: DemoPaths, start_idx: int) -> None:
    target_frame_count = None
    video_path = paths.video_dir / "cropped_video.mp4"
    if video_path.exists():
        frame_count, _, _, _ = _video_info(video_path)
        target_frame_count = frame_count - start_idx

    # Egocentric videos
    for name in ("cropped_video.mp4", "mask.mp4", "mask_bin.mp4"):
        _rewrite_video(paths.video_dir / name, start_idx)

    # Egocentric camera parameters
    for name in (
        "egocentric_frame_extrinsic.npy",
        "upd_egocentric_frame_extrinsic.npy",
        "upd_egocentric_frame_extrinsic_cam2w.npy",
        "upd_egocentric_frame_extrinsic_w2c.npy",
    ):
        _rewrite_npy(paths.camera_dir / name, start_idx)

    # Object poses and tracking
    for name in ("object.npy", "rotated.npy"):
        _rewrite_npy(paths.object_dir / name, start_idx)
    _rewrite_video(paths.object_dir / "tracking.mp4", start_idx)

    # Hand / UV
    _rewrite_json_frames(paths.hand_dir / "all_pose.json", start_idx)
    _rewrite_json_frames(paths.uv_dir / "uv.json", start_idx)

    # Removed videos / sequences
    for name in ("inpainted_video.mp4", "mask_5.mp4", "removed_w_mask_5.mp4", "w_mask_5.mp4"):
        _rewrite_video(paths.removed_dir / name, start_idx)
    _rewrite_png_sequence(paths.removed_dir / "mask_5", start_idx, target_frame_count)

    # Representative first frame
    _rewrite_frame_zero_png(paths.video_dir / "cropped_video.mp4", paths.video_dir / "frame_000.png")


def _interactive_cut(demos: Iterable[DemoPaths]) -> None:
    demo_list = list(demos)
    if not demo_list:
        print("No demos found.")
        return

    win = "cut_frames"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 900)

    demo_idx = 0
    frame_idx = 0
    step = 1

    while 0 <= demo_idx < len(demo_list):
        paths = demo_list[demo_idx]
        video_path = paths.video_dir / "cropped_video.mp4"
        frame_count, _, _, _ = _video_info(video_path)
        frame_idx = max(0, min(frame_idx, frame_count - 1))
        frame = _read_frame(video_path, frame_idx)
        title = f"{demo_idx + 1}/{len(demo_list)}  {paths.display_name}"
        canvas = _draw_overlay(frame, title, frame_idx, frame_count)
        cv2.imshow(win, canvas)

        key = cv2.waitKeyEx(0)
        key_chr = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 256 else ""

        if key_chr == "q" or key == 27:
            break
        if key_chr == "a" or key in (81, 2424832):
            frame_idx = max(0, frame_idx - step)
            continue
        if key_chr == "d" or key in (83, 2555904):
            frame_idx = min(frame_count - 1, frame_idx + step)
            continue
        if key_chr == "n":
            demo_idx += 1
            frame_idx = 0
            continue
        if key_chr == "s":
            if frame_idx == 0:
                print(f"[SKIP-CUT] {paths.display_name}: already starts at frame 0")
            else:
                print(f"[CUT] {paths.display_name}: dropping frames [0, {frame_idx - 1}]")
                _cut_demo(paths, frame_idx)
            demo_idx += 1
            frame_idx = 0
            continue

    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactively trim leading frames from task demos.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Dataset root. Can point either at the task directory or its data/ directory.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help=(
            "Demo prefix to process, e.g. drawer or flower. "
            "Defaults to the dataset directory name when data-root ends with /data."
        ),
    )
    parser.add_argument("--list-demos", action="store_true", help="List discovered demos and exit.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    data_root = _normalize_data_root(args.data_root.resolve())
    if not data_root.exists():
        print(f"Data root not found: {data_root}", file=sys.stderr)
        return 1

    task_name = args.task_name or _infer_task_name(data_root)
    demos = _iter_demos_for_task(data_root, task_name)
    print(f"Task: {task_name or '(any top-level demo)'}")
    print(f"Found {len(demos)} demos under {data_root}")
    if args.list_demos:
        for paths in demos:
            camera_status = "ok" if paths.camera_dir.exists() else "missing"
            print(f"  {paths.display_name}  camera={camera_status}")
        return 0
    print("Keys: a=back, d=forward, s=cut before current frame and move next, n=skip to next, q=quit")
    _interactive_cut(demos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
