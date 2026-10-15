"""Bridge Aria retarget output to SPIDER processed format.

This creates:
- processed/{dataset_name}/mano/{embodiment_type}/{task}/{data_id}/trajectory_keypoints.npz
- processed/{dataset_name}/mano/{embodiment_type}/{task}/task_info.json
- processed/{dataset_name}/assets/objects/{object_name}/visual.obj

So downstream SPIDER steps can run unchanged:
  decompose_fast -> generate_xml -> ik -> run_mjwp
"""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

import spider
from spider.io import get_mesh_dir, get_processed_data_dir


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _mat4_to_pose_wxyz(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate([p, quat_wxyz], axis=0)


def _identity_pose() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _find_mesh_from_env_xml(env_xml: Path) -> Path:
    tree = ET.parse(env_xml)
    root = tree.getroot()

    # Prefer tool mesh for aria mustard single-object setup
    preferred_names = [
        "tool_textured_cm_mesh",
        "tool_textured_convex_mesh",
        "tool",
    ]

    meshes = root.findall(".//mesh")
    # First pass: preferred names
    for pname in preferred_names:
        for m in meshes:
            name = m.attrib.get("name", "")
            file = m.attrib.get("file", "")
            if pname in name and file:
                return Path(file)

    # Second pass: any mesh path containing 'mustard'
    for m in meshes:
        file = m.attrib.get("file", "")
        if "mustard" in file.lower():
            return Path(file)

    raise FileNotFoundError(
        f"Cannot find object mesh in env xml: {env_xml}. Pass --object_mesh explicitly."
    )


def _extract_fingertips_world(frame: dict, side: str = "right") -> np.ndarray:
    # Expect dict with keys thumb_tip/index_tip/... and xyz values
    tips = frame.get("fingertips_world", {})
    if not isinstance(tips, dict) or len(tips) == 0:
        # fallback: all zeros
        return np.zeros((5, 3), dtype=np.float64)

    order = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    out = np.zeros((5, 3), dtype=np.float64)
    for i, k in enumerate(order):
        v = tips.get(k, [0.0, 0.0, 0.0])
        out[i] = _as_np(v)
    return out


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _plane_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12):
    v1 = b - a
    v2 = c - a
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < eps:
        raise ValueError("Hand points are collinear; wrist normal undefined.")
    return n / norm


TIP_ORDER = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
ARIA = {
    "wrist": 5,
    "thumb_prev": 7,
    "index_prev": 10,
    "middle_prev": 13,
    "ring_prev": 16,
    "pinky_prev": 19,
    "index1": 8,
    "pinky1": 17,
    "index3": 10,
    "pinky3": 19,
    "tips": [0, 1, 2, 3, 4],
}
WRIST_ALIGN_R = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
# Wrist target-frame remap used by SPIDER IK wrist orientation objective only:
# x'=y, y'=-x, z'=z
WRIST_OPT_REMAP_R = np.array([
    [ 0.,  1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0., -1.],
], dtype=np.float64)

def _build_hand_poses_from_points_21(pts21: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build wrist + five fingertip 4x4 poses in camera frame from Aria 21 points.
    This mirrors egoengine/retargeting/show_aria_new.py.
    """
    wrist_idx = ARIA["wrist"]
    hand_back_idx = ARIA["pinky3"]
    finger_root_idx = [ARIA["index1"], ARIA["pinky1"]]
    tip_to_last = {
        "thumb_tip": (0, 7),
        "index_tip": (1, 10),
        "middle_tip": (2, 13),
        "ring_tip": (3, 16),
        "pinky_tip": (4, 19),
    }

    a = pts21[wrist_idx]
    b = pts21[hand_back_idx]
    c = np.mean(pts21[finger_root_idx], axis=0)

    wrist_y = _plane_normal(a, b, c)
    wrist_z = a - b
    wrist_z /= np.linalg.norm(wrist_z) + 1e-10
    wrist_x = np.cross(wrist_y, wrist_z)
    if np.linalg.norm(wrist_x) < 1e-10:
        wrist_x = np.cross(wrist_z, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(wrist_x) < 1e-10:
            wrist_x = np.cross(wrist_z, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    wrist_x /= np.linalg.norm(wrist_x) + 1e-10

    Tw = np.eye(4, dtype=np.float64)
    Tw[:3, 0] = wrist_x
    Tw[:3, 1] = wrist_y
    Tw[:3, 2] = wrist_z
    Tw[:3, 3] = a
    Tw[:3, :3] = Tw[:3, :3] @ WRIST_ALIGN_R

    out = {"wrist": Tw}
    wrist_y_aligned = Tw[:3, 1]

    for name, (tip_id, last_id) in tip_to_last.items():
        if name != "thumb_tip":
            z = pts21[last_id] - pts21[tip_id]
            z /= np.linalg.norm(z) + 1e-10
            y = wrist_y_aligned.copy()
            x = np.cross(y, z)
            if np.linalg.norm(x) < 1e-10:
                x = np.cross(y, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                if np.linalg.norm(x) < 1e-10:
                    x = np.cross(y, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            x /= np.linalg.norm(x) + 1e-10
        else:
            x = pts21[tip_id] - pts21[last_id]
            x /= np.linalg.norm(x) + 1e-10
            y = wrist_y_aligned.copy()
            z = -1.0 * np.cross(y, x)
            if np.linalg.norm(z) < 1e-10:
                z = np.cross(x, y)
            z /= np.linalg.norm(z) + 1e-10

        Tt = np.eye(4, dtype=np.float64)
        Tt[:3, 0] = x
        Tt[:3, 1] = y
        Tt[:3, 2] = z
        Tt[:3, 3] = pts21[tip_id]
        out[name] = Tt
    return out


def _load_world_hand_poses_from_all_pose(
    all_pose_json: Path,
    w2cam_npy: Path,
    hand_z_rot_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      wrist_world: (T,7) [pos(3)+quat_wxyz(4)]
      tips_world:  (T,5,7) [pos(3)+quat_wxyz(4)] in TIP_ORDER
    """
    data = json.load(all_pose_json.open("r", encoding="utf-8"))
    frames = data.get("frames", [])
    w2cam = np.load(w2cam_npy).astype(np.float64)  # (T,4,4)
    cam2w = np.linalg.inv(w2cam)

    Tn = min(len(frames), cam2w.shape[0])
    if Tn == 0:
        raise ValueError("No valid frame from all_pose_json or w2cam")

    if abs(float(hand_z_rot_deg)) > 1e-9:
        print(
            "[WARN] hand_z_rot_deg is ignored: using show_aria_new.py fixed +90deg Z on hand keypoints."
        )

    Rz = _rot_z(np.pi / 2.0)
    seq_cam: list[dict[str, np.ndarray]] = []
    last_valid = None

    for t in range(Tn):
        fr = frames[t]
        arr = fr.get("right_camera") or fr.get("right_device")
        if arr is None:
            pts = np.full((21, 3), np.nan, dtype=np.float64)
        else:
            pts = np.asarray(arr, dtype=np.float64)
            if pts.shape != (21, 3):
                pts = np.full((21, 3), np.nan, dtype=np.float64)

        if np.isfinite(pts).all():
            pts = pts @ Rz.T
            built = _build_hand_poses_from_points_21(pts)
            seq_cam.append(built)
            last_valid = built
            continue

        if last_valid is not None:
            seq_cam.append(last_valid)
            continue

        found = None
        for j in range(t + 1, Tn):
            frj = frames[j]
            arrj = frj.get("right_camera") or frj.get("right_device")
            if arrj is None:
                continue
            ptsj = np.asarray(arrj, dtype=np.float64)
            if ptsj.shape == (21, 3) and np.isfinite(ptsj).all():
                ptsj = ptsj @ Rz.T
                found = _build_hand_poses_from_points_21(ptsj)
                break

        if found is not None:
            seq_cam.append(found)
            last_valid = found
        else:
            pts_fallback = np.nan_to_num(pts) @ Rz.T
            built = _build_hand_poses_from_points_21(pts_fallback)
            seq_cam.append(built)
            last_valid = built

    wrist_world = np.zeros((Tn, 7), dtype=np.float64)
    tips_world = np.zeros((Tn, 5, 7), dtype=np.float64)
    for t in range(Tn):
        c2w = cam2w[t]
        hand_w = {k: (c2w @ Tc) for k, Tc in seq_cam[t].items()}
        hand_w["wrist"] = hand_w["wrist"].copy()
        hand_w["wrist"][:3, :3] = hand_w["wrist"][:3, :3] @ WRIST_OPT_REMAP_R
        wrist_world[t] = _mat4_to_pose_wxyz(hand_w["wrist"])
        for i, name in enumerate(TIP_ORDER):
            tips_world[t, i] = _mat4_to_pose_wxyz(hand_w[name])
    return wrist_world, tips_world


def _estimate_wrist_from_tips(tips_xyz: np.ndarray) -> np.ndarray:
    # Simple wrist proxy: mean fingertip position + small backward z offset
    p = tips_xyz.mean(axis=0)
    p = p.copy()
    p[2] -= 0.04
    return np.concatenate([p, np.array([1.0, 0.0, 0.0, 0.0])], axis=0)


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "gigahand",
    task: str = "mustard_901_1",
    data_id: int = 0,
    embodiment_type: str = "right",  # right | left | bimanual
    retarget_json: str = "",
    env_xml: str = "",
    all_pose_json: str = "",
    w2cam_npy: str = "",
    object_mesh: str = "",  # optional: override mesh path directly
    object_name: str = "mustard_901_1",
    ref_dt: float = 0.02,
    base_offset_xyz: tuple[float, float, float] = (-0.7, 0.0, 0.0),
    arena_diff_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hand_z_rot_deg: float = 0.0,
) -> None:
    if not retarget_json:
        raise ValueError("--retarget_json is required")

    dataset_dir = Path(dataset_dir).resolve()
    retarget_path = Path(retarget_json).resolve()
    if not retarget_path.exists():
        raise FileNotFoundError(retarget_path)

    data = json.load(retarget_path.open("r", encoding="utf-8"))
    traj = data.get("traj", [])
    if not isinstance(traj, list) or len(traj) == 0:
        raise ValueError("retarget_json has no non-empty 'traj' list")

    N_ret = len(traj)
    world_off = np.asarray(base_offset_xyz, dtype=np.float64) + np.asarray(
        arena_diff_xyz, dtype=np.float64
    )

    # Build right-hand tracks from all_pose+w2cam (preferred) or retarget fallback
    qpos_wrist_right = None
    qpos_finger_right = None

    if all_pose_json and w2cam_npy:
        wrist_w, tips_w = _load_world_hand_poses_from_all_pose(
            Path(all_pose_json).resolve(),
            Path(w2cam_npy).resolve(),
            hand_z_rot_deg=float(hand_z_rot_deg),
        )
        N = min(N_ret, wrist_w.shape[0])
        qpos_wrist_right = wrist_w[:N].copy()
        qpos_finger_right = tips_w[:N].copy()
        qpos_wrist_right[:, :3] += world_off[None, :]
        qpos_finger_right[:, :, :3] += world_off[None, None, :]
    else:
        N = N_ret
        qpos_finger_right = np.zeros((N, 5, 7), dtype=np.float64)
        qpos_wrist_right = np.zeros((N, 7), dtype=np.float64)

        ident_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for i, fr in enumerate(traj[:N]):
            tips = _extract_fingertips_world(fr, side="right")
            for f in range(5):
                qpos_finger_right[i, f, :3] = tips[f]
                qpos_finger_right[i, f, 3:] = ident_quat
            qpos_wrist_right[i] = _estimate_wrist_from_tips(tips)
        qpos_wrist_right[:, :3] += world_off[None, :]
        qpos_finger_right[:, :, :3] += world_off[None, None, :]

    qpos_wrist_right = np.asarray(qpos_wrist_right, dtype=np.float64)
    qpos_finger_right = np.asarray(qpos_finger_right, dtype=np.float64)
    qpos_obj_right = np.zeros((N, 7), dtype=np.float64)

    # left side placeholders (kept for pipeline compatibility)
    qpos_finger_left = np.zeros((N, 5, 7), dtype=np.float64)
    qpos_wrist_left = np.zeros((N, 7), dtype=np.float64)
    qpos_obj_left = np.zeros((N, 7), dtype=np.float64)

    ident_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for i, fr in enumerate(traj[:N]):
        tool_T = _as_np(fr.get("tool", np.eye(4))).reshape(4, 4)
        qpos_obj_right[i] = _mat4_to_pose_wxyz(tool_T)
    qpos_obj_right[:, :3] += world_off[None, :]

    # keep left placeholders far away for right-only tasks
    qpos_wrist_left[:] = _identity_pose()[None, :]
    qpos_wrist_left[:, :3] = np.array([2.0, 2.0, 2.0])
    qpos_finger_left[:, :, :3] = np.array([2.0, 2.0, 2.0])
    qpos_finger_left[:, :, 3:] = ident_quat[None, None, :]
    qpos_obj_left[:] = _identity_pose()[None, :]
    qpos_obj_left[:, :3] = np.array([2.0, 2.0, 2.0])

    # Resolve and copy object mesh as visual.obj
    if object_mesh:
        src_mesh = Path(object_mesh).resolve()
    else:
        if not env_xml:
            raise ValueError("Either --object_mesh or --env_xml must be provided")
        src_mesh = _find_mesh_from_env_xml(Path(env_xml).resolve())
        if not src_mesh.is_absolute():
            # relative to env xml folder
            src_mesh = (Path(env_xml).resolve().parent / src_mesh).resolve()

    if not src_mesh.exists():
        raise FileNotFoundError(f"Object mesh not found: {src_mesh}")

    mesh_dir = Path(
        get_mesh_dir(
            dataset_dir=str(dataset_dir),
            dataset_name=dataset_name,
            object_name=object_name,
        )
    )
    mesh_dir.mkdir(parents=True, exist_ok=True)
    dst_visual = mesh_dir / "visual.obj"
    shutil.copy2(src_mesh, dst_visual)

    # Save trajectory_keypoints.npz
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
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        processed_dir / "trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )

    # Parse desk/table height from retarget (if available)
    table_height_z = None
    table_pose = None
    init_state = data.get("init_state", {})
    desk_init = init_state.get("desk")
    if isinstance(desk_init, list) and len(desk_init) == 4:
        try:
            table_pose = np.asarray(desk_init, dtype=np.float64).copy()
            table_pose[:3, 3] += world_off
            table_height_z = float(desk_init[2][3])
            table_height_z += float(world_off[2])
        except Exception:
            table_height_z = None
            table_pose = None
    if table_height_z is None and len(traj) > 0:
        desk0 = traj[0].get("desk")
        if isinstance(desk0, list) and len(desk0) == 4:
            try:
                table_pose = np.asarray(desk0, dtype=np.float64).copy()
                table_pose[:3, 3] += world_off
                table_height_z = float(desk0[2][3]) + float(world_off[2])
            except Exception:
                table_height_z = None
                table_pose = None

    # Save task_info.json in parent folder of data_id
    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": str(mesh_dir.relative_to(dataset_dir)),
        "left_object_mesh_dir": None,
        "ref_dt": float(ref_dt),
    }
    if table_height_z is not None:
        task_info["table_height_z"] = table_height_z
    if table_pose is not None:
        task_info["table_pose"] = table_pose.tolist()

    task_info_path = processed_dir.parent / "task_info.json"
    with task_info_path.open("w", encoding="utf-8") as f:
        json.dump(task_info, f, indent=2)

    print(f"[OK] wrote {processed_dir / 'trajectory_keypoints.npz'}")
    print(f"[OK] wrote {task_info_path}")
    print(f"[OK] copied mesh to {dst_visual}")
    if table_height_z is not None:
        print(f"[INFO] table_height_z={table_height_z:.4f}")
    print(f"[INFO] world_offset_xyz={world_off.tolist()}")
    if all_pose_json and w2cam_npy:
        print("[INFO] hand source: all_pose.json + w2cam (show_aria_new geometry)")
        print(
            f"[INFO] hand_z_rot_deg={float(hand_z_rot_deg):.1f} (ignored; using fixed +90deg keypoint rotation)"
        )
        try:
            q0 = qpos_wrist_right[0, 3:7]
            r = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
            e = r.as_euler("xyz", degrees=True)
            print(
                "[INFO] wrist0_euler_xyz_deg="
                f"[{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]"
            )
        except Exception:
            pass
    else:
        print("[INFO] hand source: retarget fingertips fallback (wrist estimated)")
    print(
        f"[INFO] frames={N}, embodiment={embodiment_type}, task={task}, data_id={data_id}"
    )


if __name__ == "__main__":
    tyro.cli(main)
