"""Convert custom self-collected data into SPIDER gigahand raw layout.

Target layout (for use_example_dataset=False in gigahand.py):
  raw/gigahand/objectposes/{scene_name}/{object_name}/{participant}-{scene}_{sequence_id}/pose/optimized_pose.json
  raw/gigahand/object_meshes/publish/{scene_name}/{object_name}/{object_stem}.obj
  raw/gigahand/handposes/{participant}-{scene}/params/{sequence_id[1:]}.json
  raw/gigahand/handposes/{participant}-{scene}/keypoints_3d/{sequence_id[1:]}/chosen_frames_left.json
  raw/gigahand/handposes/{participant}-{scene}/keypoints_3d/{sequence_id[1:]}/chosen_frames_right.json

Usage:
  uv run spider/process_datasets/convert_to_gigahand_layout.py \
    --dataset-dir /path/to/example_datasets \
    --manifest /path/to/manifest.json

Manifest schema (list[object]):
[
  {
    "participant": "p36",
    "scene": "tea",
    "scene_name": "17_instruments",
    "object_name": "ukelele_scan",
    "sequence_id": "0010",

    "object_pose_json": "/abs/path/optimized_pose.json",
    "object_mesh_obj": "/abs/path/mesh.obj",
    "hand_params_json": "/abs/path/params_10.json",

    "chosen_frames_left_json": "/abs/path/chosen_frames_left.json",   // optional
    "chosen_frames_right_json": "/abs/path/chosen_frames_right.json"  // optional
  }
]

Notes:
- If chosen_frames_* are missing, they will be auto-generated from params length.
- params json must contain top-level keys: "left", "right", and each side has
  array-like key "poses" with frame dimension in axis-0.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import tyro

import spider


class Args:
    dataset_dir: str = f"{spider.ROOT}/../example_datasets"
    manifest: str = ""
    use_symlink: bool = False
    overwrite: bool = False


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, use_symlink: bool, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    _ensure_parent(dst)

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _infer_frame_ids_from_params(params_json: Path) -> list[int]:
    data = _load_json(params_json)
    if "left" not in data or "right" not in data:
        raise ValueError("hand_params_json must contain top-level keys 'left' and 'right'")
    left = data["left"]
    right = data["right"]
    if "poses" not in left or "poses" not in right:
        raise ValueError("hand_params_json.left/right must each contain key 'poses'")
    n_left = len(left["poses"])
    n_right = len(right["poses"])
    n = min(n_left, n_right)
    return list(range(n))


def _sequence_subdir(sequence_id: str) -> str:
    # SPIDER gigahand.py uses sequence_id[1:]
    if len(sequence_id) < 2:
        raise ValueError(f"sequence_id must be at least 2 chars, got: {sequence_id}")
    return sequence_id[1:]


def main(args: Args) -> None:
    if not args.manifest:
        raise ValueError("--manifest is required")

    dataset_dir = Path(args.dataset_dir).resolve()
    raw_root = dataset_dir / "raw" / "gigahand"
    manifest_path = Path(args.manifest).resolve()
    entries = _load_json(manifest_path)
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a JSON list")

    for idx, e in enumerate(entries):
        participant = e["participant"]
        scene = e["scene"]
        scene_name = e["scene_name"]
        object_name = e["object_name"]
        sequence_id = e["sequence_id"]

        object_pose_json = Path(e["object_pose_json"]).resolve()
        object_mesh_obj = Path(e["object_mesh_obj"]).resolve()
        hand_params_json = Path(e["hand_params_json"]).resolve()

        chosen_left_src = Path(e["chosen_frames_left_json"]).resolve() if e.get("chosen_frames_left_json") else None
        chosen_right_src = Path(e["chosen_frames_right_json"]).resolve() if e.get("chosen_frames_right_json") else None

        seq_subdir = _sequence_subdir(sequence_id)
        hand_group = f"{participant}-{scene}"
        seq_group = f"{participant}-{scene}_{sequence_id}"

        # 1) object pose
        dst_obj_pose = (
            raw_root
            / "objectposes"
            / scene_name
            / object_name
            / seq_group
            / "pose"
            / "optimized_pose.json"
        )
        _copy_or_link(object_pose_json, dst_obj_pose, args.use_symlink, args.overwrite)

        # 2) object mesh in expected publish path
        object_stem = object_name.split("_")[0]
        dst_obj_mesh = (
            raw_root / "object_meshes" / "publish" / scene_name / object_name / f"{object_stem}.obj"
        )
        _copy_or_link(object_mesh_obj, dst_obj_mesh, args.use_symlink, args.overwrite)

        # 3) hand params
        dst_params = raw_root / "handposes" / hand_group / "params" / f"{seq_subdir}.json"
        _copy_or_link(hand_params_json, dst_params, args.use_symlink, args.overwrite)

        # 4) chosen frames
        chosen_dir = raw_root / "handposes" / hand_group / "keypoints_3d" / seq_subdir
        dst_left = chosen_dir / "chosen_frames_left.json"
        dst_right = chosen_dir / "chosen_frames_right.json"

        if chosen_left_src is not None and chosen_right_src is not None:
            _copy_or_link(chosen_left_src, dst_left, args.use_symlink, args.overwrite)
            _copy_or_link(chosen_right_src, dst_right, args.use_symlink, args.overwrite)
        else:
            frame_ids = _infer_frame_ids_from_params(hand_params_json)
            _save_json(dst_left, frame_ids)
            _save_json(dst_right, frame_ids)

        print(
            f"[{idx+1}/{len(entries)}] prepared {participant}-{scene}-{sequence_id} "
            f"-> {raw_root}"
        )

    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)
