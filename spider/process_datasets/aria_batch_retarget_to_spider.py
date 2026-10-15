"""Batch bridge Aria retarget outputs to SPIDER processed format.

This script is intentionally additive: it does not modify the source Aria/TACO
datasets and does not alter existing converters. It reads EgoEngine
retarget.json files and writes SPIDER processed trajectories plus copied visual
meshes.

For mustard/drawer/flower:
  right hand + one right object (tool)

For hammer:
  right hand + right object (hammer tool), still embodiment_type="right".
  The fixed shelf is saved as extra task metadata and a copied mesh path, but it
  is not put into the left object slot so downstream RL stays single-arm.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh
import tyro

import spider
from spider.io import get_mesh_dir, get_processed_data_dir
from spider.process_datasets.aria_retarget_to_spider import (
    _as_np,
    _identity_pose,
    _load_world_hand_poses_from_all_pose,
    _mat4_to_pose_wxyz,
)


TASK_MESHES = {
    "mustard": "mustard.obj",
    "drawer": "cube_convex.obj",
    "flower": "windex.obj",
    "hammer": "hammer_cut_1.5x.stl",
}

HAMMER_SHELF_MESH = "hammer_shelf.obj"


def _episode_key(path: Path) -> int:
    match = re.search(r"_(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def _infer_task_name(data_root: Path, task_name: str) -> str:
    if task_name:
        return task_name
    for part in data_root.parts:
        if part in TASK_MESHES:
            return part
    raise ValueError("--task-name is required when it cannot be inferred from data root")


def _resolve_data_root(data_root: str) -> Path:
    root = Path(data_root).resolve()
    if not (root / "Hand_Poses").is_dir() and (root / "data" / "Hand_Poses").is_dir():
        root = root / "data"
    if not (root / "Hand_Poses").is_dir():
        raise FileNotFoundError(f"Aria data root missing Hand_Poses: {root}")
    return root


def _copy_mesh_to_visual(src_mesh: Path, dst_visual: Path, overwrite: bool) -> None:
    if dst_visual.exists() and not overwrite:
        return
    dst_visual.parent.mkdir(parents=True, exist_ok=True)
    if src_mesh.suffix.lower() == ".obj":
        shutil.copy2(src_mesh, dst_visual)
        return
    mesh = trimesh.load(src_mesh, force="mesh", process=False)
    mesh.export(dst_visual)


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _dummy_left_tracks(n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ident = _identity_pose()
    qpos_wrist_left = np.repeat(ident[None, :], n_frames, axis=0)
    qpos_finger_left = np.zeros((n_frames, 5, 7), dtype=np.float64)
    qpos_obj_left = np.repeat(ident[None, :], n_frames, axis=0)
    qpos_wrist_left[:, :3] = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    qpos_finger_left[:, :, :3] = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    qpos_finger_left[:, :, 3:] = ident[3:][None, None, :]
    qpos_obj_left[:, :3] = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    return qpos_wrist_left, qpos_finger_left, qpos_obj_left


def _convert_one(
    *,
    retarget_json: Path,
    all_pose_json: Path,
    w2cam_npy: Path,
    dataset_dir: Path,
    dataset_name: str,
    task_name: str,
    task: str,
    data_id: int,
    embodiment_type: str,
    mesh_root: Path,
    ref_dt: float,
    base_offset_xyz: tuple[float, float, float],
    arena_diff_xyz: tuple[float, float, float],
    overwrite: bool,
) -> Path:
    processed_dir = Path(
        get_processed_data_dir(
            dataset_dir=str(dataset_dir),
            dataset_name=dataset_name,
            robot_type="mano",
            embodiment_type=embodiment_type,
            task=task,
            data_id=data_id,
        )
    )
    out_npz = processed_dir / "trajectory_keypoints.npz"
    if out_npz.exists() and not overwrite:
        print(f"[SKIP] exists: {out_npz}")
        return out_npz

    data = json.load(retarget_json.open("r", encoding="utf-8"))
    traj = data.get("traj", [])
    if not traj:
        raise ValueError(f"No traj in {retarget_json}")

    wrist_w, tips_w = _load_world_hand_poses_from_all_pose(all_pose_json, w2cam_npy)
    n_frames = min(len(traj), wrist_w.shape[0], tips_w.shape[0])
    world_off = np.asarray(base_offset_xyz, dtype=np.float64) + np.asarray(arena_diff_xyz, dtype=np.float64)

    qpos_wrist_right = wrist_w[:n_frames].copy()
    qpos_finger_right = tips_w[:n_frames].copy()
    qpos_wrist_right[:, :3] += world_off[None, :]
    qpos_finger_right[:, :, :3] += world_off[None, None, :]

    qpos_obj_right = np.zeros((n_frames, 7), dtype=np.float64)
    for i, fr in enumerate(traj[:n_frames]):
        tool_T = _as_np(fr.get("tool", np.eye(4))).reshape(4, 4)
        qpos_obj_right[i] = _mat4_to_pose_wxyz(tool_T)
    qpos_obj_right[:, :3] += world_off[None, :]

    qpos_wrist_left, qpos_finger_left, qpos_obj_left = _dummy_left_tracks(n_frames)
    left_mesh_rel = None
    extra_objects = {}

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )

    right_object_name = f"aria_{task_name}_tool"
    right_mesh_dir = Path(get_mesh_dir(str(dataset_dir), dataset_name, right_object_name))
    _copy_mesh_to_visual(mesh_root / TASK_MESHES[task_name], right_mesh_dir / "visual.obj", overwrite)
    right_mesh_rel = str(right_mesh_dir.relative_to(dataset_dir))

    if task_name == "hammer":
        shelf_object_name = "aria_hammer_shelf"
        shelf_mesh_dir = Path(get_mesh_dir(str(dataset_dir), dataset_name, shelf_object_name))
        _copy_mesh_to_visual(mesh_root / HAMMER_SHELF_MESH, shelf_mesh_dir / "visual.obj", overwrite)
        shelf_pose0 = _as_np(traj[0].get("target", np.eye(4))).reshape(4, 4).copy()
        shelf_pose0[:3, 3] += world_off
        extra_objects["hammer_shelf"] = {
            "mesh_dir": str(shelf_mesh_dir.relative_to(dataset_dir)),
            "pose": shelf_pose0.tolist(),
            "source": "retarget.traj[0].target",
        }

    table_height_z = None
    table_pose = None
    desk = data.get("init_state", {}).get("desk")
    if isinstance(desk, list) and len(desk) == 4:
        table_pose = np.asarray(desk, dtype=np.float64).copy()
        table_pose[:3, 3] += world_off
        table_height_z = float(table_pose[2, 3])

    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": right_mesh_rel,
        "left_object_mesh_dir": left_mesh_rel,
        "extra_objects": extra_objects,
        "ref_dt": float(ref_dt),
        "source": {
            "retarget_json": str(retarget_json),
            "all_pose_json": str(all_pose_json),
            "w2cam_npy": str(w2cam_npy),
        },
    }
    if table_height_z is not None:
        task_info["table_height_z"] = table_height_z
    if table_pose is not None:
        task_info["table_pose"] = table_pose.tolist()
    _write_json(processed_dir.parent / "task_info.json", task_info, overwrite)

    print(f"[OK] {task_name} {retarget_json.parts[-3]}/1 -> {out_npz}")
    return out_npz


def main(
    aria_data_root: str,
    task_name: str = "",
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "aria",
    task: str = "",
    embodiment_type: str = "",
    mesh_root: str = "/home/ycl/projects/workspace/egoengine/rl_tracking_dev/aria_objects/meshes",
    ref_dt: float = 0.02,
    base_offset_xyz: tuple[float, float, float] = (-0.7, 0.0, 0.0),
    arena_diff_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    overwrite: bool = False,
) -> None:
    data_root = _resolve_data_root(aria_data_root)
    task_name = _infer_task_name(data_root, task_name)
    if task_name not in TASK_MESHES:
        raise ValueError(f"task_name must be one of {sorted(TASK_MESHES)}")

    dataset_dir_p = Path(dataset_dir).resolve()
    mesh_root_p = Path(mesh_root).resolve()
    task = task or task_name
    embodiment_type = embodiment_type or "right"

    hand_root = data_root / "Hand_Poses"
    cam_root = data_root / "Egocentric_Camera_Parameters"
    episodes = sorted([p for p in hand_root.iterdir() if p.is_dir()], key=_episode_key)
    if not episodes:
        raise FileNotFoundError(f"No episodes under {hand_root}")

    ok = 0
    skipped = 0
    failed = 0
    for ep_dir in episodes:
        ep = ep_dir.name
        all_pose = ep_dir / "1" / "all_pose.json"
        retarget = ep_dir / "1" / "retarget.json"
        w2cam = cam_root / ep / "1" / "upd_egocentric_frame_extrinsic.npy"
        if not (all_pose.is_file() and retarget.is_file() and w2cam.is_file()):
            print(f"[SKIP] missing inputs for {ep}/1")
            skipped += 1
            continue
        try:
            data_id = _episode_key(ep_dir)
            _convert_one(
                retarget_json=retarget,
                all_pose_json=all_pose,
                w2cam_npy=w2cam,
                dataset_dir=dataset_dir_p,
                dataset_name=dataset_name,
                task_name=task_name,
                task=task,
                data_id=data_id,
                embodiment_type=embodiment_type,
                mesh_root=mesh_root_p,
                ref_dt=ref_dt,
                base_offset_xyz=base_offset_xyz,
                arena_diff_xyz=arena_diff_xyz,
                overwrite=overwrite,
            )
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {ep}/1: {exc}")
            failed += 1

    print(
        f"[SUMMARY] task={task_name} embodiment={embodiment_type} "
        f"ok={ok} skipped={skipped} failed={failed} total={len(episodes)}"
    )


if __name__ == "__main__":
    tyro.cli(main)
