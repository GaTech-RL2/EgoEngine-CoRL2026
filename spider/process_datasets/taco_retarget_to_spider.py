"""Bridge TACO retarget output to SPIDER processed format (bimanual xhand).

Input (TACO retarget.json):
- traj[t]["robot_cfg"]: joint name -> scalar
- traj[t]["tool"]: 4x4 pose
- traj[t]["target"]: 4x4 pose
- init_state["desk"] or traj[t]["desk"]: 4x4 table pose

Output:
- processed/{dataset}/mano/{embodiment}/{task}/{data_id}/trajectory_keypoints.npz
- processed/{dataset}/mano/{embodiment}/{task}/task_info.json
- processed/{dataset}/assets/objects/{object_name}/visual.obj (right=tool, left=target)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

import spider
from spider.io import get_mesh_dir, get_processed_data_dir


def _mat4_to_pose_wxyz(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate([p, quat_wxyz], axis=0)


def _xmat9_to_wxyz(xmat9: np.ndarray) -> np.ndarray:
    R = np.asarray(xmat9, dtype=np.float64).reshape(3, 3)
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _as_pose(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(4, 4)
    return a


def _resolve_mesh_from_env(env_xml: Path, name_hint: str) -> Path:
    tree = ET.parse(env_xml)
    root = tree.getroot()
    for m in root.findall(".//mesh"):
        name = m.attrib.get("name", "")
        file = m.attrib.get("file", "")
        if name_hint in name and file:
            p = Path(file)
            if not p.is_absolute():
                p = (env_xml.parent / p).resolve()
            return p
    raise FileNotFoundError(f"Cannot find mesh with hint='{name_hint}' in {env_xml}")


def _body_pose_wxyz(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    pos = np.asarray(data.xpos[bid], dtype=np.float64)
    quat = _xmat9_to_wxyz(data.xmat[bid])
    return np.concatenate([pos, quat], axis=0)


def _geom_pose_wxyz(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"Geom not found: {geom_name}")
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    quat = _xmat9_to_wxyz(data.geom_xmat[gid])
    return np.concatenate([pos, quat], axis=0)


RIGHT_WRIST_BODY = "gripper0_right_x_right_hand_root"
LEFT_WRIST_BODY = "gripper0_left_x_left_hand_root"

RIGHT_TIP_GEOMS = [
    "gripper0_right_right_hand_thumb_rota_tip_col",
    "gripper0_right_right_hand_index_rota_tip_col",
    "gripper0_right_right_hand_mid_tip_col",
    "gripper0_right_right_hand_ring_tip_col",
    "gripper0_right_right_hand_pinky_tip_col",
]
LEFT_TIP_GEOMS = [
    "gripper0_left_left_hand_thumb_rota_tip_col",
    "gripper0_left_left_hand_index_rota_tip_col",
    "gripper0_left_left_hand_mid_tip_col",
    "gripper0_left_left_hand_ring_tip_col",
    "gripper0_left_left_hand_pinky_tip_col",
]


def _pose7_from_T(T: np.ndarray) -> np.ndarray:
    return _mat4_to_pose_wxyz(np.asarray(T, dtype=np.float64).reshape(4, 4))


def _load_mano_hand_from_pkl(
    hand_pkl: str,
    hand_shape_pkl: str,
    side: str,
    hand_loader_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MANO pkl and return wrist/tips poses in world frame.

    Returns:
      wrist: (T,7)
      tips:  (T,5,7)  (thumb,index,middle,ring,pinky)
    """
    root = Path(hand_loader_root).resolve()
    # Add local egoengine packages so we can import hand_pose_loader directly.
    sys.path.insert(0, str(root / "retargeting"))
    sys.path.insert(0, str(root / "manopth"))
    sys.path.insert(0, str(root / "manotorch"))

    from hand_pose_loader import mano_params_to_hand_info, read_hand_shape

    beta = read_hand_shape(hand_shape_pkl)
    _, _, hand_se3 = mano_params_to_hand_info(
        hand_pose_path=hand_pkl,
        mano_beta=np.asarray(beta, dtype=np.float32),
        side=side,
        return_frames=True,
        device="cuda:0",
    )
    hand_se3 = np.asarray(hand_se3, dtype=np.float64)  # (T,21,4,4)
    if hand_se3.ndim != 4 or hand_se3.shape[1] < 21:
        raise ValueError(f"Unexpected MANO frame shape: {hand_se3.shape}")

    T = hand_se3.shape[0]
    wrist = np.zeros((T, 7), dtype=np.float64)
    tips = np.zeros((T, 5, 7), dtype=np.float64)
    # hand_pose_loader order: wrist=0, tips=16..20
    tip_ids = [16, 17, 18, 19, 20]
    for t in range(T):
        wrist[t] = _pose7_from_T(hand_se3[t, 0])
        for i, jid in enumerate(tip_ids):
            tips[t, i] = _pose7_from_T(hand_se3[t, jid])
    return wrist, tips


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "gigahand",
    task: str = "brush_eraser_bowl_20231027_037",
    data_id: int = 0,
    embodiment_type: str = "bimanual",  # should be bimanual for TACO
    retarget_json: str = "",
    env_xml: str = "",
    right_object_name: str = "taco_tool",
    left_object_name: str = "taco_target",
    right_hand_pkl: str = "",
    left_hand_pkl: str = "",
    right_hand_shape_pkl: str = "",
    left_hand_shape_pkl: str = "",
    hand_loader_root: str = "/home/ycl/projects/workspace/egoengine",
    ref_dt: float = 0.02,
    base_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    arena_diff_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    if not retarget_json:
        raise ValueError("--retarget_json is required")
    if not env_xml:
        raise ValueError("--env_xml is required")
    if embodiment_type != "bimanual":
        raise ValueError("TACO bridge is intended for bimanual tasks.")

    retarget_path = Path(retarget_json).resolve()
    env_path = Path(env_xml).resolve()
    if not retarget_path.exists():
        raise FileNotFoundError(retarget_path)
    if not env_path.exists():
        raise FileNotFoundError(env_path)

    data = json.load(retarget_path.open("r", encoding="utf-8"))
    traj = data.get("traj", [])
    if not isinstance(traj, list) or len(traj) == 0:
        raise ValueError("retarget_json has no non-empty 'traj' list")

    model = mujoco.MjModel.from_xml_path(str(env_path))
    mjd = mujoco.MjData(model)

    N = len(traj)
    world_off = np.asarray(base_offset_xyz, dtype=np.float64) + np.asarray(
        arena_diff_xyz, dtype=np.float64
    )

    qpos_wrist_right = np.zeros((N, 7), dtype=np.float64)
    qpos_wrist_left = np.zeros((N, 7), dtype=np.float64)
    qpos_finger_right = np.zeros((N, 5, 7), dtype=np.float64)
    qpos_finger_left = np.zeros((N, 5, 7), dtype=np.float64)
    qpos_obj_right = np.zeros((N, 7), dtype=np.float64)  # tool
    qpos_obj_left = np.zeros((N, 7), dtype=np.float64)   # target

    used_mano_pkl = False
    if (
        right_hand_pkl
        and left_hand_pkl
        and right_hand_shape_pkl
        and left_hand_shape_pkl
    ):
        rw, rf = _load_mano_hand_from_pkl(
            hand_pkl=right_hand_pkl,
            hand_shape_pkl=right_hand_shape_pkl,
            side="right",
            hand_loader_root=hand_loader_root,
        )
        lw, lf = _load_mano_hand_from_pkl(
            hand_pkl=left_hand_pkl,
            hand_shape_pkl=left_hand_shape_pkl,
            side="left",
            hand_loader_root=hand_loader_root,
        )
        N = min(N, rw.shape[0], lw.shape[0], rf.shape[0], lf.shape[0])
        qpos_wrist_right = rw[:N].copy()
        qpos_finger_right = rf[:N].copy()
        qpos_wrist_left = lw[:N].copy()
        qpos_finger_left = lf[:N].copy()
        qpos_obj_right = np.zeros((N, 7), dtype=np.float64)
        qpos_obj_left = np.zeros((N, 7), dtype=np.float64)
        used_mano_pkl = True

    for t, fr in enumerate(traj):
        if t >= N:
            break
        rcfg = fr.get("robot_cfg", {})
        # Reset qpos and set by joint names from robot_cfg
        mjd.qpos[:] = 0.0
        for jname, jval in rcfg.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qadr = model.jnt_qposadr[jid]
            mjd.qpos[qadr] = float(jval)
        mujoco.mj_forward(model, mjd)

        if not used_mano_pkl:
            qpos_wrist_right[t] = _body_pose_wxyz(model, mjd, RIGHT_WRIST_BODY)
            qpos_wrist_left[t] = _body_pose_wxyz(model, mjd, LEFT_WRIST_BODY)
            for i, gname in enumerate(RIGHT_TIP_GEOMS):
                qpos_finger_right[t, i] = _geom_pose_wxyz(model, mjd, gname)
            for i, gname in enumerate(LEFT_TIP_GEOMS):
                qpos_finger_left[t, i] = _geom_pose_wxyz(model, mjd, gname)

        qpos_obj_right[t] = _mat4_to_pose_wxyz(_as_pose(fr["tool"]))
        qpos_obj_left[t] = _mat4_to_pose_wxyz(_as_pose(fr["target"]))

    # apply world offset to positions
    qpos_wrist_right[:, :3] += world_off[None, :]
    qpos_wrist_left[:, :3] += world_off[None, :]
    qpos_finger_right[:, :, :3] += world_off[None, None, :]
    qpos_finger_left[:, :, :3] += world_off[None, None, :]
    qpos_obj_right[:, :3] += world_off[None, :]
    qpos_obj_left[:, :3] += world_off[None, :]

    # save trajectory keypoints
    dataset_dir_p = Path(dataset_dir).resolve()
    processed_dir = Path(
        get_processed_data_dir(
            dataset_dir=str(dataset_dir_p),
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

    # copy visual meshes for tool/target
    tool_mesh = _resolve_mesh_from_env(env_path, "tool_")
    target_mesh = _resolve_mesh_from_env(env_path, "target_")
    right_mesh_dir = Path(
        get_mesh_dir(
            dataset_dir=str(dataset_dir_p),
            dataset_name=dataset_name,
            object_name=right_object_name,
        )
    )
    left_mesh_dir = Path(
        get_mesh_dir(
            dataset_dir=str(dataset_dir_p),
            dataset_name=dataset_name,
            object_name=left_object_name,
        )
    )
    right_mesh_dir.mkdir(parents=True, exist_ok=True)
    left_mesh_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tool_mesh, right_mesh_dir / "visual.obj")
    shutil.copy2(target_mesh, left_mesh_dir / "visual.obj")

    # table info
    table_pose = None
    table_height_z = None
    init_state = data.get("init_state", {})
    desk = init_state.get("desk")
    if isinstance(desk, list) and len(desk) == 4:
        Tdesk = np.asarray(desk, dtype=np.float64).reshape(4, 4)
        Tdesk[:3, 3] += world_off
        # Retargeting saves the desk URDF tabletop link center. Spider task_info
        # expects tabletop top height, so lift by half of the 0.05 m tabletop.
        Tdesk[2, 3] += 0.025
        table_pose = Tdesk.tolist()
        table_height_z = float(Tdesk[2, 3])

    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": str(right_mesh_dir.relative_to(dataset_dir_p)),
        "left_object_mesh_dir": str(left_mesh_dir.relative_to(dataset_dir_p)),
        "ref_dt": float(ref_dt),
    }
    if table_pose is not None:
        task_info["table_pose"] = table_pose
    if table_height_z is not None:
        task_info["table_height_z"] = table_height_z
    task_info_path = processed_dir.parent / "task_info.json"
    with task_info_path.open("w", encoding="utf-8") as f:
        json.dump(task_info, f, indent=2)

    print(f"[OK] wrote {processed_dir / 'trajectory_keypoints.npz'}")
    print(f"[OK] wrote {task_info_path}")
    print(f"[OK] copied right mesh to {right_mesh_dir / 'visual.obj'}")
    print(f"[OK] copied left mesh to {left_mesh_dir / 'visual.obj'}")
    if used_mano_pkl:
        print("[INFO] hand source: right_hand.pkl + left_hand.pkl (MANO)")
    else:
        print("[INFO] hand source: retarget.json robot_cfg FK")
    print(f"[INFO] frames={N}, embodiment={embodiment_type}, task={task}, data_id={data_id}")


if __name__ == "__main__":
    tyro.cli(main)
