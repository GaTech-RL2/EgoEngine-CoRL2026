from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from generate_xml import generate_scene_for_replay

def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[[1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[[3, 0, 1, 2]]


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r1 = R.from_quat(_quat_wxyz_to_xyzw(q1))
    r2 = R.from_quat(_quat_wxyz_to_xyzw(q2))
    return _quat_xyzw_to_wxyz((r1 * r2).as_quat())


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    r = R.from_quat(_quat_wxyz_to_xyzw(q))
    return _quat_xyzw_to_wxyz(r.inv().as_quat())


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> R:
    # right.xml base chain:
    # roll about +Z, then pitch about +X, then yaw about -Y.
    return (
        R.from_rotvec(np.array([0.0, 0.0, 1.0]) * roll)
        * R.from_rotvec(np.array([1.0, 0.0, 0.0]) * pitch)
        * R.from_rotvec(np.array([0.0, -1.0, 0.0]) * yaw)
    )


def _solve_base_rpy(target_wxyz: np.ndarray, init_rpy: np.ndarray) -> np.ndarray:
    target_rot = R.from_quat(_quat_wxyz_to_xyzw(target_wxyz))

    def residual(x: np.ndarray) -> np.ndarray:
        cur = _rpy_to_rot(float(x[0]), float(x[1]), float(x[2]))
        return (cur.inv() * target_rot).as_rotvec()

    res = least_squares(
        residual,
        x0=np.asarray(init_rpy, dtype=np.float64),
        method="lm",
        max_nfev=100,
    )
    return res.x.astype(np.float64)


def _mat4_to_pose_wxyz(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = T[:3, 3]
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    return np.concatenate([p, q_wxyz], axis=0)


def _load_json(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_eef7(frame: dict[str, Any], key: str, frame_idx: int) -> np.ndarray:
    v = np.asarray(frame.get(key, []), dtype=np.float64)
    if v.shape[0] != 7:
        raise ValueError(
            f"Frame {frame_idx}: missing or invalid '{key}' (expected 7D [x,y,z,qw,qx,qy,qz])."
        )
    return v


def _infer_taco_root(input_json: Path) -> Path:
    # .../TACO/Hand_Poses/.../retarget.json -> .../TACO
    parts = input_json.parts
    if "TACO" in parts:
        idx = parts.index("TACO")
        return Path(*parts[: idx + 1])
    # fallback from expected layout
    return input_json.parents[4]


def _resolve_object_mesh_paths(data_json: dict[str, Any], input_json: Path, object_model_dir: Path | None) -> tuple[Path, Path]:
    info = data_json.get("info", {}) if isinstance(data_json.get("info"), dict) else {}
    tool_id = str(info.get("tool_id", "")).strip()
    target_id = str(info.get("target_id", "")).strip()
    if not tool_id or not target_id:
        raise ValueError("retarget.json missing info.tool_id / info.target_id")

    if object_model_dir is None:
        taco_root = _infer_taco_root(input_json)
        object_model_dir = taco_root / "Object_Models" / "object_models_released"

    tool_mesh = object_model_dir / f"{tool_id}_cm.obj"
    target_mesh = object_model_dir / f"{target_id}_cm.obj"

    if not tool_mesh.exists():
        raise FileNotFoundError(f"tool mesh not found: {tool_mesh}")
    if not target_mesh.exists():
        raise FileNotFoundError(f"target mesh not found: {target_mesh}")

    return tool_mesh, target_mesh


def _get_processed_dirs(
    dataset_dir: str,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    task: str,
    data_id: int,
) -> tuple[Path, Path]:
    task_dir = Path(dataset_dir) / "processed" / dataset_name / robot_type / embodiment_type / task
    data_dir = task_dir / str(data_id)
    return task_dir, data_dir


def main() -> None:
    parser = argparse.ArgumentParser("Replay xhand + real TACO objects + desk from retarget.json")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--hand_xml",
        type=str,
        default="/home/ycl/projects/workspace/spider/spider/assets/robots/xhand/bimanual.xml",
    )
    parser.add_argument(
        "--desk_urdf",
        type=str,
        default="/home/ycl/projects/workspace/egoengine/retargeting/assets/desk/desk.urdf",
    )
    parser.add_argument("--object_model_dir", type=str, default="")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--cam_name", type=str, default="right_track")
    parser.add_argument(
        "--video_camera",
        type=str,
        default="front_camera",
        help="MuJoCo camera name for video rendering.",
    )
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument(
        "--video_cycles",
        type=int,
        default=1,
        help="How many full trajectory cycles to record (1 = one pass).",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Run headless conversion / rendering without opening MuJoCo viewer window.",
    )
    parser.add_argument("--dataset_dir", type=str, default="/home/ycl/projects/workspace/spider/example_datasets")
    parser.add_argument("--dataset_name", type=str, default="gigahand")
    parser.add_argument("--robot_type", type=str, default="xhand")
    parser.add_argument("--embodiment_type", type=str, default="bimanual")
    parser.add_argument("--task", type=str, default="brush_eraser_bowl_20231027_037")
    parser.add_argument("--data_id", type=int, default=0)
    parser.add_argument(
        "--write_processed",
        action="store_true",
        help="Deprecated and ignored. Processed outputs are always written.",
    )
    args = parser.parse_args()
    input_json = Path(args.input_json).expanduser()
    hand_xml = Path(args.hand_xml).expanduser()
    desk_urdf = Path(args.desk_urdf).expanduser()
    object_model_dir = Path(args.object_model_dir).expanduser() if args.object_model_dir else None

    # Output directory follows input retarget.json directory.
    # Example:
    #   input:  .../20230927_027/retarget.json
    #   output: .../20230927_027/{scene.xml, scene_eq.xml, task_info.json, trajectory_kinematic.npz, ik.mp4}
    task_dir = input_json.parent
    data_dir = input_json.parent
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if not input_json.exists():
        raise FileNotFoundError(f"input_json not found: {input_json}")
    if not hand_xml.exists():
        raise FileNotFoundError(f"hand_xml not found: {hand_xml}")

    data_json = _load_json(input_json)
    traj = data_json.get("traj", [])
    if not isinstance(traj, list) or len(traj) == 0:
        raise ValueError(f"Invalid traj in: {input_json}")

    tool_mesh, target_mesh = _resolve_object_mesh_paths(data_json, input_json, object_model_dir)
    print(f"[INFO] tool mesh:   {tool_mesh}")
    print(f"[INFO] target mesh: {target_mesh}")
    print(f"[INFO] desk urdf:   {desk_urdf}")
    print(f"[INFO] save video:  {data_dir / 'ik.mp4'} (camera={args.video_camera})")
    init_state = data_json.get("init_state", {})
    init_desk_pose = init_state.get("desk", None)
    corrected_desk_pose = None
    if init_desk_pose is not None:
        corrected_desk_pose = np.asarray(init_desk_pose, dtype=np.float64).reshape(4, 4).copy()
        # Retargeting stores the URDF tabletop link pose at its center, while Spider
        # expects the tabletop top height. The desk tabletop thickness is 0.05 m.
        corrected_desk_pose[2, 3] += 0.025
    scene_files = generate_scene_for_replay(
        hand_xml=str(hand_xml),
        output_dir=str(task_dir),
        embodiment_type=args.embodiment_type,
        robot_type=args.robot_type,
        right_collision_mesh_files=[str(tool_mesh)],
        left_collision_mesh_files=[str(target_mesh)]
        if args.embodiment_type in ["left", "bimanual"]
        else [],
        right_visual_mesh_file=str(tool_mesh),
        left_visual_mesh_file=str(target_mesh)
        if args.embodiment_type in ["left", "bimanual"]
        else None,
        right_contact_pos=np.zeros((5, 3), dtype=np.float64),
        left_contact_pos=np.zeros((5, 3), dtype=np.float64),
        table_pose=corrected_desk_pose,
        table_height_z=None,
        ref_dt=1.0 / max(float(args.hz), 1e-6),
        sim_dt=0.01,
    )
    model = mujoco.MjModel.from_xml_path(scene_files["scene_xml"])
    data = mujoco.MjData(model)

    sid_right_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    sid_left_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    if sid_right_palm < 0:
        raise ValueError("right_palm site not found in hand xml")
    if sid_left_palm < 0:
        raise ValueError("left_palm site not found in hand xml")
    q_right_link_palm = np.asarray(model.site_quat[sid_right_palm], dtype=np.float64)
    q_left_link_palm = np.asarray(model.site_quat[sid_left_palm], dtype=np.float64)

    right_base_joint_names = [
        "R_forearm_tx_link_joint",
        "R_forearm_ty_link_joint",
        "R_forearm_tz_link_joint",
        "R_forearm_roll_link_joint",
        "R_forearm_pitch_link_joint",
        "R_forearm_yaw_link_joint",
    ]
    left_base_joint_names = [
        "L_forearm_tx_link_joint",
        "L_forearm_ty_link_joint",
        "L_forearm_tz_link_joint",
        "L_forearm_roll_link_joint",
        "L_forearm_pitch_link_joint",
        "L_forearm_yaw_link_joint",
    ]
    right_base_joint_ids = []
    for name in right_base_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        right_base_joint_ids.append(jid)
    left_base_joint_ids = []
    for name in left_base_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        left_base_joint_ids.append(jid)

    sample_cfg = traj[0].get("robot_cfg", {}) if isinstance(traj[0], dict) else {}
    cfg_joint_names = []
    for jname in sample_cfg.keys():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            cfg_joint_names.append(jname)

    jid_right_object = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    jid_left_object = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_object_joint")
    qadr_right_object = int(model.jnt_qposadr[jid_right_object]) if jid_right_object >= 0 else -1
    qadr_left_object = int(model.jnt_qposadr[jid_left_object]) if jid_left_object >= 0 else -1

    if not all(("right_eef" in fr and "left_eef" in fr) for fr in traj):
        raise ValueError(
            "retarget.json must contain both 'right_eef' and 'left_eef' in every traj frame. "
            "Please regenerate json with updated show_human_new/show_aria_new."
        )

    dt = 1.0 / max(float(args.hz), 1e-6)
    prev_rpy_right = np.zeros(3, dtype=np.float64)
    prev_rpy_left = np.zeros(3, dtype=np.float64)

    # Precompute qpos sequence once; reuse for both writing and replay.
    T = len(traj)
    qpos_seq = np.zeros((T, model.nq), dtype=np.float64)
    for i in range(T):
        fr = traj[i]
        right_eef = _require_eef7(fr, "right_eef", i)
        p_world_right_palm = right_eef[:3]
        q_world_right_palm = right_eef[3:7]
        q_world_right_link = _quat_mul_wxyz(
            q_world_right_palm, _quat_inv_wxyz(q_right_link_palm)
        )
        rpy_right = _solve_base_rpy(q_world_right_link, prev_rpy_right)
        prev_rpy_right = rpy_right
        right_base_vals = np.array(
            [
                p_world_right_palm[0],
                p_world_right_palm[1],
                p_world_right_palm[2],
                rpy_right[0],
                rpy_right[1],
                rpy_right[2],
            ],
            dtype=np.float64,
        )

        left_eef = _require_eef7(fr, "left_eef", i)
        p_world_left_palm = left_eef[:3]
        q_world_left_palm = left_eef[3:7]
        q_world_left_link = _quat_mul_wxyz(
            q_world_left_palm, _quat_inv_wxyz(q_left_link_palm)
        )
        rpy_left = _solve_base_rpy(q_world_left_link, prev_rpy_left)
        prev_rpy_left = rpy_left
        left_base_vals = np.array(
            [
                p_world_left_palm[0],
                p_world_left_palm[1],
                p_world_left_palm[2],
                rpy_left[0],
                rpy_left[1],
                rpy_left[2],
            ],
            dtype=np.float64,
        )

        data.qpos[:] = 0.0
        for jid, val in zip(right_base_joint_ids, right_base_vals):
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = float(val)
        for jid, val in zip(left_base_joint_ids, left_base_vals):
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = float(val)

        rcfg = fr.get("robot_cfg", {})
        for jname in cfg_joint_names:
            if jname in rcfg:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                qadr = int(model.jnt_qposadr[jid])
                data.qpos[qadr] = float(rcfg[jname])

        tool_pose = fr.get("tool", None)
        target_pose = fr.get("target", None)
        if tool_pose is not None and qadr_right_object >= 0:
            data.qpos[qadr_right_object : qadr_right_object + 7] = _mat4_to_pose_wxyz(
                np.asarray(tool_pose, dtype=np.float64)
            )
        if target_pose is not None and qadr_left_object >= 0:
            data.qpos[qadr_left_object : qadr_left_object + 7] = _mat4_to_pose_wxyz(
                np.asarray(target_pose, dtype=np.float64)
            )

        qpos_seq[i] = data.qpos.copy()

    # Build qvel with MuJoCo-differentiated positions.
    qvel_seq = np.zeros((T, model.nv), dtype=np.float64)
    for i in range(T - 1):
        qvel = np.zeros((model.nv,), dtype=np.float64)
        mujoco.mj_differentiatePos(model, qvel, dt, qpos_seq[i], qpos_seq[i + 1])
        qvel_seq[i] = qvel
    if T > 1:
        qvel_seq[-1] = qvel_seq[-2]

    ctrl_seq = np.zeros((T, model.nu), dtype=np.float64)
    if model.nu > 0:
        take = min(model.nu, model.nq)
        ctrl_seq[:, :take] = qpos_seq[:, :take]

    # Dummy contact labels for RL format compatibility.
    contact_seq = np.zeros((T, 10), dtype=np.float64)
    contact_pos_seq = np.zeros((T, 10, 3), dtype=np.float64)

    scene_path = Path(scene_files["scene_xml"])
    scene_eq_path = Path(scene_files["scene_eq_xml"])

    traj_path = data_dir / "trajectory_kinematic.npz"
    np.savez_compressed(
        traj_path,
        qpos=qpos_seq.astype(np.float32),
        qvel=qvel_seq.astype(np.float32),
        ctrl=ctrl_seq.astype(np.float32),
        contact=contact_seq.astype(np.float32),
        contact_pos=contact_pos_seq.astype(np.float32),
        frequency=np.asarray([float(args.hz)], dtype=np.float32),
    )

    task_info_path = Path(scene_files["task_info_json"])
    task_info = {}
    if task_info_path.exists():
        try:
            task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
        except Exception:
            task_info = {}
    task_info["ref_dt"] = dt
    task_info.setdefault("sim_dt", 0.01)
    task_info_path.write_text(json.dumps(task_info, indent=2), encoding="utf-8")
    print(f"[OK] wrote {scene_path}")
    print(f"[OK] wrote {scene_eq_path}")
    print(f"[OK] wrote {traj_path}")
    print(f"[OK] updated {task_info_path}")

    video_path = data_dir / "ik.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.video_camera)
    if cam_id < 0:
        raise ValueError(f"Video camera not found in model: {args.video_camera}")
    req_w = int(args.video_width)
    req_h = int(args.video_height)
    max_w = int(model.vis.global_.offwidth)
    max_h = int(model.vis.global_.offheight)
    out_w = min(req_w, max_w)
    out_h = min(req_h, max_h)
    if out_w != req_w or out_h != req_h:
        print(
            f"[WARN] requested video {req_w}x{req_h} exceeds framebuffer {max_w}x{max_h}; "
            f"using {out_w}x{out_h}."
        )
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        float(args.hz),
        (int(out_w), int(out_h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    renderer = mujoco.Renderer(model, width=out_w, height=out_h)

    try:
        n = len(traj)
        i = 0
        cycles_done = 0
        if args.no_viewer:
            while True:
                data.qpos[:] = qpos_seq[i]
                data.qvel[:] = qvel_seq[i]
                mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera=args.video_camera)
                rgb = renderer.render()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                i += 1
                if i >= n:
                    i = 0
                    cycles_done += 1
                    if args.video_cycles > 0 and cycles_done >= int(args.video_cycles):
                        break
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    data.qpos[:] = qpos_seq[i]
                    data.qvel[:] = qvel_seq[i]
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    renderer.update_scene(data, camera=args.video_camera)
                    rgb = renderer.render()
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                    time.sleep(dt)
                    i += 1
                    if i >= n:
                        i = 0
                        cycles_done += 1
                        if args.video_cycles > 0 and cycles_done >= int(args.video_cycles):
                            break
    finally:
        writer.release()
        renderer.close()
        print(f"[INFO] video saved to: {video_path}")


if __name__ == "__main__":
    main()
