#!/usr/bin/env python3
"""Create a compact HDF5 dataset from Aria replay outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_DATA_ROOT = Path("/home/ycl/projects/datasets/mustard/data")
DEFAULT_CROP = (150, 510, 250, 610)
DEFAULT_RESIZE = (640, 640)
ENV_ARGS = {"env_name": "RBY1Teleop", "env_type": "robosuite", "env_kwargs": {}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact robomimic-style HDF5 from replay.json.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--trajectory-source",
        choices=("replay", "best", "custom"),
        default="replay",
        help="Which trajectory JSONs to read.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=None,
        help=(
            "Root for non-replay trajectory JSONs. For best trajectories, expected layout is "
            "<trajectory-root>/<episode>_rl/<trajectory-name>."
        ),
    )
    parser.add_argument("--trajectory-name", type=str, default=None)
    parser.add_argument(
        "--video-source",
        choices=("inpainted", "cropped", "custom"),
        default="inpainted",
        help="Which video tree to read. Use custom with --video-root and --video-name.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Root for custom videos, organized as <video-root>/<episode>/1/<video-name>.",
    )
    parser.add_argument("--video-name", type=str, default=None)
    parser.add_argument("--episodes", nargs="*", default=None, help="Optional episode names, e.g. drawer_1.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered episodes.")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=DEFAULT_CROP,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Crop as frame[y0:y1, x0:x1].",
    )
    parser.add_argument("--compression", default="gzip", help="HDF5 compression for images, or none.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]


def normalize_data_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / "Hand_Poses").is_dir() and (path / "data" / "Hand_Poses").is_dir():
        return path / "data"
    return path


def infer_task_name(data_root: Path) -> str:
    return data_root.parent.name if data_root.name == "data" else data_root.name


def discover_episodes(data_root: Path, episodes: list[str] | None) -> list[str]:
    hand_root = data_root / "Hand_Poses"
    if episodes:
        return episodes
    return [p.parts[-3] for p in sorted(hand_root.glob("*/1/replay.json"), key=lambda p: natural_key(p.parts[-3]))]


def resolve_trajectory_path(args: argparse.Namespace, episode: str) -> Path:
    if args.trajectory_source == "replay":
        return args.data_root / "Hand_Poses" / episode / "1" / (args.trajectory_name or "replay.json")
    if args.trajectory_source == "best":
        if args.trajectory_root is None:
            raise ValueError("--trajectory-root is required for --trajectory-source best")
        return args.trajectory_root / f"{episode}_rl" / (args.trajectory_name or "best_trajectory.json")
    if args.trajectory_root is None or args.trajectory_name is None:
        raise ValueError("--trajectory-root and --trajectory-name are required for --trajectory-source custom")
    return args.trajectory_root / episode / "1" / args.trajectory_name


def resolve_video_path(args: argparse.Namespace, episode: str) -> Path:
    if args.video_source == "inpainted":
        return args.data_root / "Removed" / episode / "1" / (args.video_name or "inpainted_video.mp4")
    if args.video_source == "cropped":
        return args.data_root / "Egocentric_RGB_Videos" / episode / "1" / (
            args.video_name or "cropped_video.mp4"
        )
    if args.video_root is None or args.video_name is None:
        raise ValueError("--video-root and --video-name are required for --video-source custom")
    return args.video_root / episode / "1" / args.video_name


def load_replay_arrays(replay_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with replay_path.open("r") as f:
        replay = json.load(f)

    traj = replay.get("traj", [])
    if not traj:
        raise ValueError(f"No replay samples found in {replay_path}")

    eef_proprio = []
    eef = []
    hand_right_qpos = []
    hand_right_cmd_qpos = []
    for idx, step in enumerate(traj):
        try:
            arm_prop = np.asarray(step["right_arm6_proprio"], dtype=np.float32).reshape(6)
            xhand_prop = np.asarray(step["xhand_proprio"], dtype=np.float32).reshape(-1)
            arm_cmd = np.asarray(step["right_arm6"], dtype=np.float32).reshape(6)
            xhand_cmd = np.asarray(step["xhand"], dtype=np.float32).reshape(-1)
        except KeyError as exc:
            raise KeyError(f"Missing key {exc} in {replay_path}, sample {idx}") from exc

        if xhand_prop.shape != xhand_cmd.shape:
            raise ValueError(
                f"XHand proprio/action shape mismatch in {replay_path}, sample {idx}: "
                f"{xhand_prop.shape} vs {xhand_cmd.shape}"
            )
        eef_proprio.append(arm_prop.astype(np.float32))
        eef.append(arm_cmd.astype(np.float32))
        hand_right_qpos.append(xhand_prop.astype(np.float32))
        hand_right_cmd_qpos.append(xhand_cmd.astype(np.float32))

    return (
        np.asarray(eef_proprio, dtype=np.float32),
        np.asarray(eef, dtype=np.float32),
        np.asarray(hand_right_qpos, dtype=np.float32),
        np.asarray(hand_right_cmd_qpos, dtype=np.float32),
    )


def read_cropped_video(video_path: Path, expected_frames: int, crop: tuple[int, int, int, int]) -> np.ndarray:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    y0, y1, x0, x1 = crop
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != expected_frames:
        cap.release()
        raise ValueError(f"Frame count mismatch for {video_path}: video={frame_count}, replay={expected_frames}")

    frames = []
    for frame_idx in range(expected_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        frame_bgr = cv2.resize(frame_bgr, DEFAULT_RESIZE, interpolation=cv2.INTER_AREA)
        height, width = frame_bgr.shape[:2]
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            cap.release()
            raise ValueError(f"Invalid crop {crop} for frame size {height}x{width} in {video_path}")
        cropped_bgr = frame_bgr[y0:y1, x0:x1]
        frames.append(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def write_demo(
    data_group: h5py.Group,
    demo_name: str,
    images: np.ndarray,
    eef_proprio: np.ndarray,
    eef: np.ndarray,
    hand_right_qpos: np.ndarray,
    hand_right_cmd_qpos: np.ndarray,
    compression: str | None,
    attrs: dict[str, str],
) -> int:
    if not (len(images) == len(eef_proprio) == len(eef) == len(hand_right_qpos) == len(hand_right_cmd_qpos)):
        raise ValueError(
            f"Length mismatch for {demo_name}: images={len(images)}, eef_proprio={len(eef_proprio)}, "
            f"eef={len(eef)}, hand_right_qpos={len(hand_right_qpos)}, "
            f"hand_right_cmd_qpos={len(hand_right_cmd_qpos)}"
        )

    demo_group = data_group.create_group(demo_name)
    demo_group.attrs["num_samples"] = int(len(images))
    for key, value in attrs.items():
        demo_group.attrs[key] = value

    obs_group = demo_group.create_group("obs")
    actions_group = demo_group.create_group("actions")
    obs_group.create_dataset("images", data=images, compression=compression)
    obs_group.create_dataset("eef_proprio", data=eef_proprio.astype(np.float32))
    obs_group.create_dataset("hand_right_qpos", data=hand_right_qpos.astype(np.float32))
    actions_group.create_dataset("eef", data=eef.astype(np.float32))
    actions_group.create_dataset("hand_right_cmd_qpos", data=hand_right_cmd_qpos.astype(np.float32))
    return int(len(images))


def main() -> int:
    args = parse_args()
    data_root = normalize_data_root(args.data_root)
    args.data_root = data_root
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_root / f"{infer_task_name(data_root)}_replay_dataset.hdf5"
    )
    compression = None if str(args.compression).lower() in {"", "none", "null"} else args.compression

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    episodes = discover_episodes(data_root, args.episodes)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise RuntimeError(
            f"No replay.json files found under {data_root / 'Hand_Poses'}. "
            "Run run_aria.sh / replay_trajectory_seg_aria first."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with h5py.File(output, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs["env_args"] = json.dumps(ENV_ARGS)

        for demo_idx, episode in enumerate(episodes):
            replay_path = resolve_trajectory_path(args, episode)
            if not replay_path.is_file():
                raise FileNotFoundError(f"Trajectory JSON not found: {replay_path}")
            video_path = resolve_video_path(args, episode)
            eef_proprio, eef, hand_right_qpos, hand_right_cmd_qpos = load_replay_arrays(replay_path)
            images = read_cropped_video(video_path, expected_frames=len(eef), crop=tuple(args.crop))
            total += write_demo(
                data_group=data_group,
                demo_name=f"demo_{demo_idx}",
                images=images,
                eef_proprio=eef_proprio,
                eef=eef,
                hand_right_qpos=hand_right_qpos,
                hand_right_cmd_qpos=hand_right_cmd_qpos,
                compression=compression,
                attrs={
                    "episode": episode,
                    "trajectory_json": str(replay_path),
                    "trajectory_source": args.trajectory_source,
                    "video_path": str(video_path),
                    "video_source": args.video_source,
                },
            )
            print(
                f"[DONE] demo_{demo_idx}: episode={episode} samples={len(eef)} "
                f"image_shape={tuple(images.shape[1:])}"
            )

        data_group.attrs["total"] = int(total)

    print(f"[DONE] wrote {output} demos={len(episodes)} total_samples={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
