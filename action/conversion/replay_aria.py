from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from generate_xml import generate_scene_for_replay


ARIA_MESH_DIR = Path(
    "/home/ycl/projects/workspace/egoengine/rl_tracking_dev/aria_objects/meshes"
)

TASK_MESHES = {
    "mustard": {
        "visual": ARIA_MESH_DIR / "mustard.obj",
        "collision": [ARIA_MESH_DIR / "mustard_convex.obj"],
    },
    "drawer": {
        "visual": ARIA_MESH_DIR / "cube_convex.obj",
        "collision": [ARIA_MESH_DIR / "cube_convex.obj"],
    },
    "flower": {
        "visual": ARIA_MESH_DIR / "windex.obj",
        "collision": [ARIA_MESH_DIR / "windex.obj"],
    },
    "hammer": {
        "visual": ARIA_MESH_DIR / "hammer_cut_1.5x.stl",
        "collision": sorted((ARIA_MESH_DIR / "hammer_collision").glob("*.obj")),
    },
}

OBJECT_DENSITY = 1000.0
OBJECT_ARMATURE = 0.0001
OBJECT_FRICTIONLOSS = 0.0001
OBJECT_FRICTION = "1 1 0.1"
OBJECT_SOLREF = "0.02 1"
HAMMER_SHELF_DENSITY = 1196.9

HAMMER_SHELF = {
    "visual": ARIA_MESH_DIR / "hammer_shelf.obj",
    "collision": sorted((ARIA_MESH_DIR / "hammer_shelf_collision").glob("*.obj")),
}


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
    return np.concatenate([p, q_xyzw[[3, 0, 1, 2]]], axis=0)


def _pose_wxyz_from_mat4_text(T: np.ndarray) -> tuple[str, str]:
    pose = _mat4_to_pose_wxyz(T)
    pos = " ".join(f"{v:.9g}" for v in pose[:3])
    quat = " ".join(f"{v:.9g}" for v in pose[3:])
    return pos, quat


def _set_camera_pos(scene_xml: Path, camera_name: str, pos: np.ndarray) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    camera = root.find(f".//camera[@name='{camera_name}']")
    if camera is None:
        raise ValueError(f"Camera not found in XML: {camera_name}")
    camera.set("pos", " ".join(f"{v:.9g}" for v in np.asarray(pos, dtype=np.float64)))
    tree.write(scene_xml, encoding="utf-8", xml_declaration=False)


def _trajectory_tool_center(traj: list[dict[str, Any]], task_name: str) -> np.ndarray:
    pts = []
    for fr in traj:
        tool = fr.get("tool")
        if tool is not None:
            pts.append(np.asarray(tool, dtype=np.float64).reshape(4, 4)[:3, 3])
        if task_name == "hammer":
            target = fr.get("target")
            if target is not None:
                pts.append(np.asarray(target, dtype=np.float64).reshape(4, 4)[:3, 3])
    if not pts:
        return np.array([0.55, -0.25, 0.80], dtype=np.float64)
    return np.median(np.stack(pts, axis=0), axis=0)


def _load_json(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_eef7(frame: dict[str, Any], key: str, frame_idx: int) -> np.ndarray:
    v = np.asarray(frame.get(key, []), dtype=np.float64)
    if v.shape[0] != 7:
        raise ValueError(
            f"Frame {frame_idx}: missing or invalid '{key}' "
            "(expected 7D [x,y,z,qw,qx,qy,qz])."
        )
    return v


def _ensure_paths(paths: list[Path]) -> list[str]:
    out = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        out.append(str(p))
    return out


def _task_from_retarget(data_json: dict[str, Any], explicit: str) -> str:
    if explicit:
        return explicit
    info = data_json.get("info", {}) if isinstance(data_json.get("info"), dict) else {}
    task_name = str(info.get("task_name", "")).strip()
    if task_name:
        return task_name
    raise ValueError("Cannot infer task name; pass --task-name")


def _inject_hammer_shelf(scene_xml: Path, shelf_pose: np.ndarray) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Invalid MuJoCo XML: {scene_xml}")

    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "hammer_shelf_visual",
            "file": str(HAMMER_SHELF["visual"].resolve()),
            "scale": "1 1 1",
        },
    )
    for i, mesh_path in enumerate(HAMMER_SHELF["collision"]):
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": f"hammer_shelf_collision_{i}",
                "file": str(mesh_path.resolve()),
                "scale": "1 1 1",
            },
        )

    pos, quat = _pose_wxyz_from_mat4_text(shelf_pose)
    body = ET.SubElement(worldbody, "body", {"name": "hammer_shelf", "pos": pos, "quat": quat})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "hammer_shelf_visual",
            "type": "mesh",
            "mesh": "hammer_shelf_visual",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "0",
            "rgba": "0.75 0.1 0.02 1",
        },
    )
    for i, _ in enumerate(HAMMER_SHELF["collision"]):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"hammer_shelf_collision_{i}",
                "type": "mesh",
                "mesh": f"hammer_shelf_collision_{i}",
                "contype": "1",
                "conaffinity": "1",
                "condim": "4",
                "density": f"{HAMMER_SHELF_DENSITY:g}",
                "friction": OBJECT_FRICTION,
                "solref": OBJECT_SOLREF,
                "group": "3",
                "rgba": "0.75 0.1 0.02 1",
            },
        )

    tree.write(scene_xml, encoding="utf-8", xml_declaration=False)


def _iter_retarget_jsons(data_root: Path, task_name: str, episodes: list[str]) -> list[Path]:
    hand_root = data_root / "Hand_Poses"
    if not hand_root.exists():
        raise FileNotFoundError(hand_root)
    if episodes:
        candidates = []
        for ep in episodes:
            candidates.extend(sorted((hand_root / ep).glob("*/retarget.json")))
        return candidates
    return sorted(hand_root.glob(f"{task_name}_*/1/retarget.json"))


def _body_mass(model: mujoco.MjModel, body_name: str) -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return 0.0
    return float(model.body_mass[bid])


def _print_object_masses(model: mujoco.MjModel, task_name: str) -> dict[str, float]:
    masses = {"tool": _body_mass(model, "right_object")}
    if task_name == "hammer":
        masses["hammer_shelf"] = _body_mass(model, "hammer_shelf")
    msg = ", ".join(f"{name}={mass:.6f} kg" for name, mass in masses.items())
    print(f"[INFO] object masses: {msg}")
    return masses


def convert_one(args: argparse.Namespace, input_json: Path) -> bool:
    data_json = _load_json(input_json)
    task_name = _task_from_retarget(data_json, args.task_name)
    if task_name not in TASK_MESHES:
        raise ValueError(f"Unsupported Aria task: {task_name}")

    data_dir = input_json.parent
    outputs = [
        data_dir / "scene.xml",
        data_dir / "scene_eq.xml",
        data_dir / "task_info.json",
        data_dir / "trajectory_kinematic.npz",
        data_dir / "ik.mp4",
    ]
    if not args.overwrite and all(p.exists() for p in outputs):
        print(f"[SKIP] outputs already exist: {input_json}")
        return False

    traj = data_json.get("traj", [])
    if not isinstance(traj, list) or len(traj) == 0:
        raise ValueError(f"Invalid traj in: {input_json}")

    mesh_info = TASK_MESHES[task_name]
    collision_meshes = _ensure_paths(list(mesh_info["collision"]))
    visual_mesh = Path(mesh_info["visual"])
    if not visual_mesh.exists():
        raise FileNotFoundError(visual_mesh)

    init_state = data_json.get("init_state", {})
    init_desk_pose = init_state.get("desk", None)
    corrected_desk_pose = None
    if init_desk_pose is not None:
        corrected_desk_pose = np.asarray(init_desk_pose, dtype=np.float64).reshape(4, 4).copy()
        corrected_desk_pose[2, 3] += 0.025

    scene_files = generate_scene_for_replay(
        hand_xml=str(Path(args.hand_xml).expanduser()),
        output_dir=str(data_dir),
        embodiment_type="right",
        robot_type=args.robot_type,
        right_collision_mesh_files=collision_meshes,
        left_collision_mesh_files=[],
        right_visual_mesh_file=str(visual_mesh),
        left_visual_mesh_file=None,
        right_contact_pos=np.zeros((5, 3), dtype=np.float64),
        left_contact_pos=np.zeros((5, 3), dtype=np.float64),
        table_pose=corrected_desk_pose,
        table_height_z=None,
        object_density=OBJECT_DENSITY,
        object_armature=OBJECT_ARMATURE,
        object_frictionloss=OBJECT_FRICTIONLOSS,
        object_mesh_scale=1.0,
        ref_dt=1.0 / max(float(args.hz), 1e-6),
        sim_dt=0.01,
    )

    shelf_pose = None
    if task_name == "hammer":
        _ensure_paths([HAMMER_SHELF["visual"], *HAMMER_SHELF["collision"]])
        shelf_pose = np.asarray(traj[0].get("target"), dtype=np.float64).reshape(4, 4)
        _inject_hammer_shelf(Path(scene_files["scene_xml"]), shelf_pose)
        _inject_hammer_shelf(Path(scene_files["scene_eq_xml"]), shelf_pose)

    if not args.no_auto_center_camera:
        center = (
            np.asarray(args.camera_lookat, dtype=np.float64)
            if args.camera_lookat is not None
            else _trajectory_tool_center(traj, task_name)
        )
        camera_pos = (
            np.asarray(args.camera_pos, dtype=np.float64)
            if args.camera_pos is not None
            else center + np.asarray(args.camera_offset, dtype=np.float64)
        )
        _set_camera_pos(
            Path(scene_files["scene_xml"]),
            args.video_camera,
            camera_pos,
        )
        _set_camera_pos(
            Path(scene_files["scene_eq_xml"]),
            args.video_camera,
            camera_pos,
        )
        print(
            "[INFO] camera "
            f"{args.video_camera}: pos={camera_pos.tolist()} "
            f"center={center.tolist()} orientation=unchanged"
        )

    model = mujoco.MjModel.from_xml_path(scene_files["scene_xml"])
    data = mujoco.MjData(model)
    object_masses = _print_object_masses(model, task_name)

    sid_right_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    if sid_right_palm < 0:
        raise ValueError("right_palm site not found in hand xml")
    q_right_link_palm = np.asarray(model.site_quat[sid_right_palm], dtype=np.float64)

    right_base_joint_names = [
        "R_forearm_tx_link_joint",
        "R_forearm_ty_link_joint",
        "R_forearm_tz_link_joint",
        "R_forearm_roll_link_joint",
        "R_forearm_pitch_link_joint",
        "R_forearm_yaw_link_joint",
    ]
    right_base_joint_ids = []
    for name in right_base_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        right_base_joint_ids.append(jid)

    sample_cfg = traj[0].get("robot_cfg", {}) if isinstance(traj[0], dict) else {}
    cfg_joint_names = []
    for jname in sample_cfg.keys():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            cfg_joint_names.append(jname)

    jid_right_object = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    qadr_right_object = int(model.jnt_qposadr[jid_right_object]) if jid_right_object >= 0 else -1

    dt = 1.0 / max(float(args.hz), 1e-6)
    prev_rpy_right = np.zeros(3, dtype=np.float64)
    T = len(traj)
    qpos_seq = np.zeros((T, model.nq), dtype=np.float64)
    for i, fr in enumerate(traj):
        right_eef = _require_eef7(fr, "right_eef", i)
        p_world_right_palm = right_eef[:3]
        q_world_right_palm = right_eef[3:7]
        q_world_right_link = _quat_mul_wxyz(
            q_world_right_palm, _quat_inv_wxyz(q_right_link_palm)
        )
        rpy_right = _solve_base_rpy(q_world_right_link, prev_rpy_right)
        prev_rpy_right = rpy_right

        data.qpos[:] = 0.0
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
        for jid, val in zip(right_base_joint_ids, right_base_vals):
            data.qpos[int(model.jnt_qposadr[jid])] = float(val)

        rcfg = fr.get("robot_cfg", {})
        for jname in cfg_joint_names:
            if jname in rcfg:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                data.qpos[int(model.jnt_qposadr[jid])] = float(rcfg[jname])

        tool_pose = fr.get("tool", None)
        if tool_pose is not None and qadr_right_object >= 0:
            data.qpos[qadr_right_object : qadr_right_object + 7] = _mat4_to_pose_wxyz(
                np.asarray(tool_pose, dtype=np.float64)
            )

        qpos_seq[i] = data.qpos.copy()

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

    contact_seq = np.zeros((T, 10), dtype=np.float64)
    contact_pos_seq = np.zeros((T, 10, 3), dtype=np.float64)

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
    task_info["task_name"] = task_name
    task_info["ref_dt"] = dt
    task_info.setdefault("sim_dt", 0.01)
    task_info["embodiment_type"] = "right"
    task_info["right_object_visual_mesh"] = str(visual_mesh)
    task_info["right_object_collision_meshes"] = collision_meshes
    task_info["object_density"] = OBJECT_DENSITY
    task_info["object_armature"] = OBJECT_ARMATURE
    task_info["object_frictionloss"] = OBJECT_FRICTIONLOSS
    task_info["object_friction"] = OBJECT_FRICTION
    task_info["object_solref"] = OBJECT_SOLREF
    task_info["object_masses_kg"] = object_masses
    if shelf_pose is not None:
        task_info["extra_objects"] = {
            "hammer_shelf": {
                "visual_mesh": str(HAMMER_SHELF["visual"]),
                "collision_meshes": [str(p) for p in HAMMER_SHELF["collision"]],
                "pose": shelf_pose.tolist(),
                "fixed": True,
                "density": HAMMER_SHELF_DENSITY,
            }
        }
    task_info_path.write_text(json.dumps(task_info, indent=2), encoding="utf-8")

    print(f"[OK] wrote {scene_files['scene_xml']}")
    print(f"[OK] wrote {scene_files['scene_eq_xml']}")
    print(f"[OK] wrote {traj_path}")
    print(f"[OK] updated {task_info_path}")

    video_path = data_dir / "ik.mp4"
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.video_camera)
    if cam_id < 0:
        raise ValueError(f"Video camera not found in model: {args.video_camera}")
    req_w = int(args.video_width)
    req_h = int(args.video_height)
    out_w = min(req_w, int(model.vis.global_.offwidth))
    out_h = min(req_h, int(model.vis.global_.offheight))
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.hz),
        (int(out_w), int(out_h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    renderer = mujoco.Renderer(model, width=out_w, height=out_h)
    try:
        cycles_done = 0
        i = 0
        if args.no_viewer:
            while True:
                data.qpos[:] = qpos_seq[i]
                data.qvel[:] = qvel_seq[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=args.video_camera)
                rgb = renderer.render()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                i += 1
                if i >= T:
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
                    if i >= T:
                        i = 0
                        cycles_done += 1
                        if args.video_cycles > 0 and cycles_done >= int(args.video_cycles):
                            break
    finally:
        writer.release()
        renderer.close()
        print(f"[INFO] video saved to: {video_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        "Generate TACO-style replay outputs for Aria retarget.json demos."
    )
    parser.add_argument("--input_json", type=str, default="")
    parser.add_argument("--aria-data-root", type=str, default="")
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--episodes", nargs="*", default=[])
    parser.add_argument(
        "--hand_xml",
        type=str,
        default="/home/ycl/projects/workspace/spider/spider/assets/robots/xhand/right.xml",
    )
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--video_camera", type=str, default="front_camera")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--video_cycles", type=int, default=1)
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World camera position for ik.mp4. Orientation is unchanged.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Object center used only to compute camera position with --camera-offset.",
    )
    parser.add_argument(
        "--camera-offset",
        type=float,
        nargs=3,
        default=[0.95, -0.2, 0.5],
        metavar=("DX", "DY", "DZ"),
        help="Camera position offset from object center when --camera-pos is omitted.",
    )
    parser.add_argument(
        "--no-auto-center-camera",
        action="store_true",
        help="Keep the camera generated by generate_xml.py unchanged.",
    )
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--robot_type", type=str, default="xhand")
    args = parser.parse_args()

    if args.input_json:
        inputs = [Path(args.input_json).expanduser()]
    else:
        if not args.aria_data_root or not args.task_name:
            raise ValueError("Pass either --input_json or both --aria-data-root and --task-name")
        inputs = _iter_retarget_jsons(
            Path(args.aria_data_root).expanduser(),
            args.task_name,
            list(args.episodes),
        )

    ok = 0
    skipped = 0
    failed = 0
    for input_json in inputs:
        try:
            wrote = convert_one(args, input_json)
            if wrote:
                ok += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {input_json}: {exc}")
            if args.input_json:
                raise
    print(f"[SUMMARY] ok={ok} skipped={skipped} failed={failed} total={len(inputs)}")


if __name__ == "__main__":
    main()
