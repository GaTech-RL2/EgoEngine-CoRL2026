# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run IK for the given hand type and mode.

Input data format: npz file which contains qpos for key frames.


Author: Chaoyi Pan
Date: 2025-07-06
"""

import json
import os

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from mujoco import MjSpec
from omegaconf import DictConfig, OmegaConf
from scipy import signal

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _offset_palm_target_pos(
    wrist_pos: np.ndarray, wrist_quat_wxyz: np.ndarray, offset_m: float = 0.03
) -> np.ndarray:
    """Shift wrist target backward along palm direction by offset_m.

    Assumes palm direction is the wrist frame local +Z axis.
    """
    R = _quat_wxyz_to_rotmat(wrist_quat_wxyz)
    palm_dir_world = R[:, 2]
    return wrist_pos - palm_dir_world * offset_m


def _load_obj_mesh(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ mesh vertices and triangulated faces."""
    verts = []
    faces = []
    with open(obj_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    continue
                idxs = []
                for p in parts:
                    vtok = p.split("/")[0]
                    if not vtok:
                        continue
                    vid = int(vtok)
                    if vid > 0:
                        idxs.append(vid - 1)  # OBJ is 1-based
                    else:
                        idxs.append(vid)  # negative index, resolve later
                if len(idxs) < 3:
                    continue
                # Triangulate polygon faces using fan triangulation.
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if len(verts) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
        )
    verts = np.asarray(verts, dtype=np.float64)
    if len(faces) == 0:
        return verts, np.zeros((0, 3), dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    faces = np.where(faces < 0, faces + verts.shape[0], faces)
    valid = np.all((faces >= 0) & (faces < verts.shape[0]), axis=1)
    return verts, faces[valid]


def _nearest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Closest point on triangle ABC to point p."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * bc

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + v * ab + w * ac


def _nearest_point_on_vertices(
    p_world: np.ndarray, verts_world: np.ndarray
) -> tuple[np.ndarray, float]:
    if verts_world is None or verts_world.shape[0] == 0:
        return p_world, np.inf
    d2 = np.sum((verts_world - p_world[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return verts_world[idx], float(np.sqrt(d2[idx]))


def _nearest_point_on_mesh(
    p_world: np.ndarray, verts_world: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, float]:
    """Closest point on triangle mesh; falls back to vertices if faces are missing."""
    if verts_world is None or verts_world.shape[0] == 0:
        return p_world, np.inf
    if faces is None or faces.shape[0] == 0:
        return _nearest_point_on_vertices(p_world, verts_world)

    best_p = p_world
    best_d2 = np.inf
    for f in faces:
        q = _nearest_point_on_triangle(
            p_world, verts_world[f[0]], verts_world[f[1]], verts_world[f[2]]
        )
        d2 = float(np.sum((q - p_world) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_p = q
    return best_p, float(np.sqrt(best_d2))


def add_mocap_bodies(
    mjspec: MjSpec,
    sites_for_mimic: list[str],
    mocap_bodies: list[str],
    robot_conf: DictConfig = None,
    add_equality_constraint: bool = True,
):
    """Add mocap bodies to the model specification.
    Source: https://github.com/robfiras/loco-mujoco

    Args:
        mjspec (MjSpec): The model specification.
        sites_for_mimic (List[str]): The sites to mimic.
        mocap_bodies (List[str]): The names of the mocap bodies to be added to the model specification.
        mocap_bodies_init_pos: The initial positions of the mocap bodies.
        add_equality_constraint (bool): Whether to add equality constraints between the sites and the mocap bodies.

    """
    if robot_conf is not None and robot_conf.optimization_params.disable_joint_limits:
        for j in mjspec.joints:
            j.limited = False

    for j in mjspec.joints:
        j.actfrclimited = 0

    if robot_conf is not None and robot_conf.optimization_params.disable_collisions:
        for g in mjspec.geoms:
            g.contype = 0
            g.conaffinity = 0

    for mb_name in mocap_bodies:
        b_handle = mjspec.worldbody.add_body(name=mb_name, mocap=True)
        if "palm" in mb_name or "object" in mb_name:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.02, 0.03],
                rgba=[0.0, 1.0, 0.0, 0.5],
                group=1,
            )
        else:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],
                rgba=[0.0, 1.0, 0.0, 0.5],
                group=1,
            )

    if add_equality_constraint:
        for b1, b2 in zip(sites_for_mimic, mocap_bodies, strict=False):
            if robot_conf is not None:
                eq_type = getattr(
                    mujoco.mjtEq,
                    robot_conf.site_joint_matches[b1].equality_constraint_type,
                )
                torque_scale = robot_conf.site_joint_matches[b1].torque_scale
            else:
                eq_type = mujoco.mjtEq.mjEQ_CONNECT
                torque_scale = 1.0

            constraint_data = np.zeros(11)
            constraint_data[10] = torque_scale
            e = mjspec.add_equality(
                name=f"{b1}_{b2}_equality_constraint",
                type=eq_type,
                name1=b1,
                name2=b2,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=constraint_data,
            )

            if robot_conf is not None:
                if hasattr(robot_conf.site_joint_matches[b1], "solref"):
                    test = len(robot_conf.site_joint_matches[b1].solref)
                    assert len(robot_conf.site_joint_matches[b1].solref) == 2, (
                        "solref must be a list of length 2"
                    )
                    e.solref = robot_conf.site_joint_matches[b1].solref
                if hasattr(robot_conf.site_joint_matches[b1], "solimp"):
                    assert len(robot_conf.site_joint_matches[b1].solimp) == 5, (
                        "solimp must be a list of length 5"
                    )
                    e.solimp = robot_conf.site_joint_matches[b1].solimp

    return mjspec


def get_robot_sites(robot_type: str, embodiment_type: str):
    if robot_type in ["allegro", "metahand"]:
        sites_in_robot = [
            "right_palm",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_thumb_tip",
            "left_palm",
            "left_ring_tip",
            "left_middle_tip",
            "left_index_tip",
            "left_thumb_tip",
            "right_object",
            "left_object",
        ]
    else:
        sites_in_robot = [
            "right_palm",
            "right_thumb_tip",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_pinky_tip",
            "left_palm",
            "left_thumb_tip",
            "left_index_tip",
            "left_middle_tip",
            "left_ring_tip",
            "left_pinky_tip",
            "right_object",
            "left_object",
        ]
    if embodiment_type == "right":
        sites_in_robot = [s for s in sites_in_robot if "right" in s]
    elif embodiment_type == "left":
        sites_in_robot = [s for s in sites_in_robot if "left" in s]
    return sites_in_robot


# parameters
def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    save_video: bool = False,
    enable_collision: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
    data_id: int = 0,
    open_hand: bool = False,
    contact_detection_step_threshold: int = 3,
    finger_solimp_width: float = 0.01,
    wrist_solimp_width: float = 10.0,
    wrist_torque_scale: float = 10.0,
    object_solimp_width: float = 0.001,
    max_num_initial_guess: int = 8,
    average_frame_size: int = 3,
    aggregate_contact: bool = False,
    z_offset: float = 0.0,
    mesh_contact_threshold: float = 0.02,  # meters
    contact_height_eps: float = 0.005,  # object z above min z -> contact stage
):
    # resolved processed directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    # load task info (for object convex mesh dirs)
    task_info_path = f"{processed_dir_robot}/../task_info.json"
    if not os.path.exists(task_info_path):
        raise FileNotFoundError(f"task_info.json not found: {task_info_path}")
    with open(task_info_path, encoding="utf-8") as f:
        task_info = json.load(f)
    # load model from processed scene
    model_path = f"{processed_dir_robot}/../scene.xml"
    # NOTE: sites in robot should follow the order of the xml file
    sites_in_robot = get_robot_sites(robot_type, embodiment_type)

    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left = loaded_data["qpos_obj_left"][start_idx:end_idx]
    try:
        contact_left = loaded_data["contact_left"][start_idx:end_idx]
        contact_right = loaded_data["contact_right"][start_idx:end_idx]
    except KeyError as e:
        raise KeyError(
            "Missing contact data in trajectory_keypoints.npz. "
            "Please run detect_contact first and ensure "
            "contact_left/contact_right are saved."
        ) from e
    try:
        contact_pos_left_raw = loaded_data["contact_pos_left"]
        contact_pos_right_raw = loaded_data["contact_pos_right"]
    except KeyError as e:
        raise KeyError(
            "Missing contact_pos data in trajectory_keypoints.npz. "
            "Please run detect_contact first and ensure "
            "contact_pos_left/contact_pos_right are saved."
        ) from e
    H_contact = qpos_finger_right.shape[0]
    # contact_pos can be either static (5,3) or per-frame (H,5,3)
    if contact_pos_right_raw.ndim == 2:
        contact_pos_right = np.tile(contact_pos_right_raw[None, :, :], (H_contact, 1, 1))
    else:
        contact_pos_right = contact_pos_right_raw[start_idx:end_idx]
    if contact_pos_left_raw.ndim == 2:
        contact_pos_left = np.tile(contact_pos_left_raw[None, :, :], (H_contact, 1, 1))
    else:
        contact_pos_left = contact_pos_left_raw[start_idx:end_idx]
    contact_ref = np.concatenate([contact_right, contact_left], axis=1)
    if aggregate_contact:
        # Aggregate contact per hand side (not across both hands).
        # right hand: channels [0:5], left hand: channels [5:10]
        if embodiment_type == "right":
            right_any = np.any(contact_right > 0.5, axis=1, keepdims=True).astype(
                contact_right.dtype
            )
            contact_ref[:, :5] = np.repeat(right_any, 5, axis=1)
            contact_ref[:, 5:] = 0.0
        elif embodiment_type == "left":
            left_any = np.any(contact_left > 0.5, axis=1, keepdims=True).astype(
                contact_left.dtype
            )
            contact_ref[:, :5] = 0.0
            contact_ref[:, 5:] = np.repeat(left_any, 5, axis=1)
        elif embodiment_type == "bimanual":
            right_any = np.any(contact_right > 0.5, axis=1, keepdims=True).astype(
                contact_right.dtype
            )
            left_any = np.any(contact_left > 0.5, axis=1, keepdims=True).astype(
                contact_left.dtype
            )
            contact_ref[:, :5] = np.repeat(right_any, 5, axis=1)
            contact_ref[:, 5:] = np.repeat(left_any, 5, axis=1)
        else:
            raise ValueError(f"Invalid embodiment_type: {embodiment_type}")
    # get the first contact frame where contact_left turns to 1 (two 1s consecutive)
    first_contact_frame_left = np.zeros(5) + qpos_finger_right.shape[0]
    first_contact_frame_right = np.zeros(5) + qpos_finger_left.shape[0]
    for j in range(5):
        for i in range(contact_detection_step_threshold, len(contact_left)):
            if contact_left[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_left[j] = i
                break
        for i in range(contact_detection_step_threshold, len(contact_right)):
            if contact_right[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_right[j] = i
                break

    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset
    # Height-based contact stage (per side).
    right_contact_stage = (
        qpos_obj_right[:, 2] > (np.min(qpos_obj_right[:, 2]) + contact_height_eps)
    )
    left_contact_stage = (
        qpos_obj_left[:, 2] > (np.min(qpos_obj_left[:, 2]) + contact_height_eps)
    )

    # Load object visual mesh (local frame) for mesh-distance based contact logic.
    right_object_verts_local = np.zeros((0, 3), dtype=np.float64)
    right_object_faces = np.zeros((0, 3), dtype=np.int64)
    left_object_verts_local = np.zeros((0, 3), dtype=np.float64)
    left_object_faces = np.zeros((0, 3), dtype=np.int64)
    if embodiment_type in ["right", "bimanual"]:
        right_mesh_dir_rel = task_info.get("right_object_mesh_dir")
        if right_mesh_dir_rel:
            right_mesh_file = os.path.join(dataset_dir, right_mesh_dir_rel, "visual.obj")
            if os.path.isfile(right_mesh_file):
                right_object_verts_local, right_object_faces = _load_obj_mesh(
                    right_mesh_file
                )
            else:
                loguru.logger.warning(f"Right object mesh not found: {right_mesh_file}")
    if embodiment_type in ["left", "bimanual"]:
        left_mesh_dir_rel = task_info.get("left_object_mesh_dir")
        if left_mesh_dir_rel:
            left_mesh_file = os.path.join(dataset_dir, left_mesh_dir_rel, "visual.obj")
            if os.path.isfile(left_mesh_file):
                left_object_verts_local, left_object_faces = _load_obj_mesh(left_mesh_file)
            else:
                loguru.logger.warning(f"Left object mesh not found: {left_mesh_file}")

    # load model
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)

    # NOTE: sites for mimic should follow the order of data
    index_map = {}
    cnt = 0
    for sides in ["right", "left"]:
        for body_name in [
            "palm",
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]:
            index_map[f"{sides}_{body_name}"] = {
                "qpos_idx": cnt,
                "mocap_idx": -1,
                "eq_constraint_idx": -1,
            }
            cnt += 1
    # add objects
    index_map["right_object"] = {
        "qpos_idx": cnt,
        "mocap_idx": -1,
        "eq_constraint_idx": -1,
    }
    cnt += 1
    index_map["left_object"] = {
        "qpos_idx": cnt,
        "mocap_idx": -1,
        "eq_constraint_idx": -1,
    }
    cnt += 1

    sites_for_mimic = [
        "right_palm",
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
        "left_palm",
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
        "right_object",
        "left_object",
    ]

    # special case: allegro hand
    if robot_type in ["allegro", "metahand"]:
        sites_for_mimic.remove("right_pinky_tip")
        sites_for_mimic.remove("left_pninky_tip")

    if embodiment_type == "right":
        sites_for_mimic = [s for s in sites_for_mimic if "right" in s]
    elif embodiment_type == "left":
        sites_for_mimic = [s for s in sites_for_mimic if "left" in s]

    site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, s)
        for s in sites_for_mimic
    ]
    # make sure all site_ids are valid, i.e. no -1
    assert all(site_id != -1 for site_id in site_ids), f"site_ids: {site_ids}"
    mano2mimic_site_idx = []
    for s in sites_for_mimic:
        # find robot site name
        for site_name in sites_in_robot:
            if site_name == s:
                mano2mimic_site_idx.append(sites_in_robot.index(site_name))

    # create mocap sites for retargeting
    site_joint_matches = {}
    for key in sites_for_mimic:
        if "palm" in key:  # palm: strong rotation constraint, weak position constraint
            constraint_type = "mjEQ_WELD"
            solimp = [0.0, 0.95, wrist_solimp_width, 0.5, 2.0]
            torque_scale = wrist_torque_scale
            solref = [0.02, 1.0]
        elif "object" in key:  # object: strong position and rotation constraint
            constraint_type = "mjEQ_WELD"
            torque_scale = 10.0
            solimp = [0.9, 0.95, object_solimp_width, 0.5, 2.0]
            solref = [0.002, 1.0]
        else:  # finger: weak rotation constraint, strong position constraint (but weaker than object)
            constraint_type = "mjEQ_CONNECT"
            if "thumb" in key or "index" in key or "middle" in key:
                width_scale = 1.0
            else:
                width_scale = 3.0
            solimp = [0.0, 0.95, finger_solimp_width * width_scale, 0.5, 2.0]
            solref = [0.01, 1.0]
            torque_scale = 1.0
        site_joint_matches[key] = {
            "equality_constraint_type": constraint_type,
            "torque_scale": torque_scale,
            "solref": solref,
            "solimp": solimp,
        }

    robot_conf = OmegaConf.create(
        {
            "optimization_params": {
                "disable_joint_limits": False,
                "disable_collisions": not enable_collision,
            },
            "site_joint_matches": site_joint_matches,
        }
    )
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mj_spec = mujoco.MjSpec.from_file(model_path)

    # ================================
    # add constraints to the free body
    # ================================
    mjspec = add_mocap_bodies(
        mj_spec,
        sites_for_mimic,
        target_mocap_bodies,
        robot_conf,
        add_equality_constraint=True,
    )

    # ================================
    # add constraints to relative bodies, i.e. stick the object to the finger
    # ================================
    finger_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    sides = {
        "right": ["right"],
        "left": ["left"],
        "bimanual": ["right", "left"],
    }[embodiment_type]

    # add position sensor to sites_for_mimic
    for i in range(len(sites_for_mimic)):
        site_name = sites_for_mimic[i]
        mjspec.add_sensor(
            name=f"pos_{site_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site_name,
        )

    mj_model_ik = mj_spec.compile()
    mj_model_ik.opt.timestep = sim_dt
    mj_model_ik.opt.iterations = 20
    mj_model_ik.opt.ls_iterations = 50
    if not enable_collision:
        mj_model_ik.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model_ik.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    mj_data_ik = mujoco.MjData(mj_model_ik)

    # update index_map
    for target_mocap_body in target_mocap_bodies:
        body_name = target_mocap_body[18:]
        body_id = mujoco.mj_name2id(
            mj_model_ik, mujoco.mjtObj.mjOBJ_BODY, target_mocap_body
        )
        mocap_id = mj_model_ik.body_mocapid[body_id]
        index_map[body_name]["mocap_idx"] = mocap_id
        # print(body_name, body_id, mocap_id)
    # set object position
    if embodiment_type == "bimanual":
        mj_data_ik.qpos[-14:-7] = qpos_obj_right[0]
        mj_data_ik.qpos[-7:] = qpos_obj_left[0]
    elif embodiment_type == "right":
        mj_data_ik.qpos[-7:] = qpos_obj_right[0]
    elif embodiment_type == "left":
        mj_data_ik.qpos[-7:] = qpos_obj_left[0]

    # set the mocap sites to the tip positions
    # for i, site_id in enumerate(site_ids):
    mujoco.mj_forward(mj_model_ik, mj_data_ik)
    for i in range(len(sites_for_mimic)):
        site_id = site_ids[i]
        site_name = sites_for_mimic[i]
        mano_id = mano2mimic_site_idx[i]
        mocap_id = i
        # Warm-start from human wrist pose (position + orientation).
        ref_pos = qpos_ref[0, mano_id, :3]
        ref_quat = qpos_ref[0, mano_id, 3:]
        if "palm" in site_name:
            ref_pos = _offset_palm_target_pos(ref_pos, ref_quat, offset_m=-0.02)
        mj_data_ik.mocap_pos[mocap_id] = ref_pos
        mj_data_ik.mocap_quat[mocap_id] = ref_quat

    # rollout mujoco
    # reference dt inferred from MANO keypoint data spacing if available; default to 0.02
    rate_limiter = RateLimiter(1 / ref_dt)
    H = qpos_finger_right.shape[0]
    cnt = 0
    if save_video:
        import imageio

        mj_model_ik.vis.global_.offwidth = 720
        mj_model_ik.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model_ik, height=480, width=720)

    run_viewer = get_viewer(show_viewer, mj_model_ik, mj_data_ik)

    # random initial guess to find a stable initial pose

    ref_mocap_ids = []
    ref_site_ids = []
    track_site_ids = []
    track_site_names = []
    # get track site ids
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None and name.startswith("track"):
            track_site_ids.append(sid)
            track_site_names.append(name)
    for sid, track_name in zip(track_site_ids, track_site_names, strict=False):
        ref_name = track_name.replace("track", "ref")
        # get mocap id of ref site
        mocap_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, ref_name)
        mocap_id = mj_model.body_mocapid[mocap_body_id]
        ref_mocap_ids.append(mocap_id)
        # get site id of ref site
        ref_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ref_name)
        ref_site_ids.append(ref_site_id)

    with run_viewer() as gui:
        cnt = 0
        while cnt < H:
            frame_contact_right = np.zeros(5, dtype=np.float64)
            frame_contact_left = np.zeros(5, dtype=np.float64)
            right_verts_world = None
            left_verts_world = None
            right_faces = None
            left_faces = None
            if right_object_verts_local.shape[0] > 0:
                obj_qidx = index_map["right_object"]["qpos_idx"]
                obj_pos = qpos_ref[cnt, obj_qidx, :3]
                obj_R = _quat_wxyz_to_rotmat(qpos_ref[cnt, obj_qidx, 3:])
                right_verts_world = right_object_verts_local @ obj_R.T + obj_pos
                right_faces = right_object_faces
            if left_object_verts_local.shape[0] > 0:
                obj_qidx = index_map["left_object"]["qpos_idx"]
                obj_pos = qpos_ref[cnt, obj_qidx, :3]
                obj_R = _quat_wxyz_to_rotmat(qpos_ref[cnt, obj_qidx, 3:])
                left_verts_world = left_object_verts_local @ obj_R.T + obj_pos
                left_faces = left_object_faces

            if cnt == 0:
                # reset distance cost
                cost_sum = 0.0
                # reset data buffer
                best_qpos_init = np.zeros(mj_model_ik.nq)
                best_qpos_diff_sum = np.inf
                for i in range(max_num_initial_guess):
                    mj_data_ik.qpos[:] = np.random.rand(mj_model_ik.nq)
                    mj_data_ik.qvel[:] = np.zeros(mj_model_ik.nv)
                    mj_data_ik.ctrl[:] = np.random.rand(mj_model_ik.nu)
                    mujoco.mj_forward(mj_model_ik, mj_data_ik)
                    mocap_id_list = []
                    qpos_id_list = []
                    for k, v in index_map.items():
                        if v["mocap_idx"] != -1:
                            mocap_idx = v["mocap_idx"]
                            qpos_idx = v["qpos_idx"]
                            # Warm-start stage: use human wrist pose as initialization target.
                            ref_pos = qpos_ref[cnt, qpos_idx, :3]
                            ref_quat = qpos_ref[cnt, qpos_idx, 3:]
                            if "palm" in k:
                                ref_pos = _offset_palm_target_pos(
                                    ref_pos, ref_quat, offset_m=0.02
                                )
                            mj_data_ik.mocap_pos[mocap_idx] = ref_pos
                            if "tip" in k:
                                mj_data_ik.mocap_pos[mocap_idx] += (
                                    np.random.randn(3) * 0.002
                                )
                            mj_data_ik.mocap_quat[mocap_idx] = ref_quat
                    nq_obj = 14 if embodiment_type == "bimanual" else 7
                    qpos_diff_sum = 0.0
                    for i in range(30):
                        mj_data_ik.ctrl[:] = mj_data_ik.qpos[:-nq_obj].copy()
                        mujoco.mj_step(mj_model_ik, mj_data_ik)
                    # compute mocap diff
                    for mocap_id, qpos_id in zip(
                        mocap_id_list, qpos_id_list, strict=False
                    ):
                        mocap_pos = mj_data_ik.mocap_pos[mocap_id]
                        mocap_quat = mj_data_ik.mocap_quat[mocap_id]
                        qpos_pos = qpos_ref[cnt, qpos_id, :3]
                        qpos_quat = qpos_ref[cnt, qpos_id, 3:]
                        mocap_diff = np.linalg.norm(mocap_pos - qpos_pos)
                        qpos_diff = np.linalg.norm(mocap_quat - qpos_quat)
                        qpos_diff_sum += mocap_diff + qpos_diff
                    mj_data.qpos[:] = mj_data_ik.qpos.copy()
                    mj_data.qvel[:] = mj_data_ik.qvel.copy() * 0.0
                    mj_data.ctrl[:] = mj_data_ik.qpos[:-nq_obj].copy()
                    mujoco.mj_forward(mj_model, mj_data)
                    for i in range(30):
                        mujoco.mj_step(mj_model, mj_data)
                        qpos_diff_sum += np.linalg.norm(mj_data.qpos - mj_data_ik.qpos)
                    if qpos_diff_sum < best_qpos_diff_sum:
                        best_qpos_init = mj_data_ik.qpos.copy()
                        best_qpos_diff_sum = qpos_diff_sum
                loguru.logger.info(f"best_qpos_diff_sum: {best_qpos_diff_sum}")
                mj_data_ik.qpos[:] = best_qpos_init
                mj_data_ik.qvel[:] = 0.0
                mj_data_ik.ctrl[:] = mj_data_ik.qpos[:-nq_obj].copy()
                mujoco.mj_step(mj_model_ik, mj_data_ik)
                qpos_list = []
                contact_pos_list = []
                contact_list = []
                images = []

            mujoco.mj_forward(mj_model_ik, mj_data_ik)
            for k, v in index_map.items():
                if v["mocap_idx"] != -1:
                    target_pos = qpos_ref[cnt, v["qpos_idx"], :3].copy()
                    target_quat = qpos_ref[cnt, v["qpos_idx"], 3:].copy()
                    if "palm" in k:
                        target_pos = _offset_palm_target_pos(
                            target_pos, target_quat, offset_m=0.02
                        )
                    # Mesh-threshold contact logic:
                    # if fingertip is close enough to object mesh, project it
                    # onto mesh and mark as contact.
                    if "_tip" in k:
                        finger_map = {
                            "thumb": 0,
                            "index": 1,
                            "middle": 2,
                            "ring": 3,
                            "pinky": 4,
                        }
                        fid = None
                        for fn, fi in finger_map.items():
                            if fn in k:
                                fid = fi
                                break
                        if fid is not None:
                            if k.startswith("right_") and right_verts_world is not None:
                                if right_contact_stage[cnt]:
                                    # In-air stage: force all five fingers onto mesh.
                                    p_proj, _ = _nearest_point_on_mesh(
                                        target_pos, right_verts_world, right_faces
                                    )
                                    target_pos = p_proj
                                    frame_contact_right[fid] = 1.0
                                else:
                                    p_proj, dist = _nearest_point_on_mesh(
                                        target_pos, right_verts_world, right_faces
                                    )
                                    if dist < mesh_contact_threshold:
                                        target_pos = p_proj
                                        frame_contact_right[fid] = 1.0
                            elif k.startswith("left_") and left_verts_world is not None:
                                if left_contact_stage[cnt]:
                                    # In-air stage: force all five fingers onto mesh.
                                    p_proj, _ = _nearest_point_on_mesh(
                                        target_pos, left_verts_world, left_faces
                                    )
                                    target_pos = p_proj
                                    frame_contact_left[fid] = 1.0
                                else:
                                    p_proj, dist = _nearest_point_on_mesh(
                                        target_pos, left_verts_world, left_faces
                                    )
                                    if dist < mesh_contact_threshold:
                                        target_pos = p_proj
                                        frame_contact_left[fid] = 1.0
                    mj_data_ik.mocap_pos[v["mocap_idx"]] = target_pos
                    mj_data_ik.mocap_quat[v["mocap_idx"]] = target_quat

            for _ in range(max(1, int(ref_dt / sim_dt))):
                mujoco.mj_step(mj_model_ik, mj_data_ik)

            # set site position and set it to ref mocap position (use original mj_model and mj_data)
            mj_data.qpos[:] = mj_data_ik.qpos.copy()
            mj_data.qvel[:] = 0.0
            nq_obj = 14 if embodiment_type == "bimanual" else 7
            mj_data.ctrl[:] = mj_data_ik.qpos[:-nq_obj].copy()

            # override joint position according to contact state
            if open_hand:
                for side in ["right", "left"]:
                    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                        # get joint index
                        joint_ids = []
                        for jid in range(mj_model.njnt):
                            joint_name = mujoco.mj_id2name(
                                mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid
                            )
                            if side in joint_name and finger in joint_name:
                                joint_ids.append(jid)
                        if len(joint_ids) > 0:
                            for joint_idx in joint_ids:
                                current_joint_pos = mj_data.qpos[joint_idx]
                                zero_joint_pos = 0.0
                                # Map sides and fingers to their respective indices
                                side_map = {
                                    "right": first_contact_frame_right,
                                    "left": first_contact_frame_left,
                                }
                                finger_map = {
                                    "thumb": 0,
                                    "index": 1,
                                    "middle": 2,
                                    "ring": 3,
                                    "pinky": 4,
                                }

                                contact_frame_list = side_map[side]
                                finger_idx = finger_map[finger]
                                contact_frame = contact_frame_list[finger_idx]

                                # Use smooth transition with clipping
                                ratio = np.clip(cnt / max(contact_frame, 1), 0.0, 1.0)
                                ratio = 1.0 - np.cos(ratio * np.pi * 0.5)
                                joint_pos = (
                                    ratio * current_joint_pos
                                    + (1 - ratio) * zero_joint_pos
                                )
                                mj_data.qpos[joint_idx] = joint_pos

            mujoco.mj_kinematics(mj_model, mj_data)
            for i in range(len(ref_mocap_ids)):
                mocap_id = ref_mocap_ids[i]
                track_site_id = track_site_ids[i]
                track_site_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
                )
                mj_data.mocap_pos[mocap_id] = mj_data.site_xpos[track_site_id].copy()

            contact = np.zeros(len(track_site_ids))
            contact_map = {
                "right_thumb": 0,
                "right_index": 1,
                "right_middle": 2,
                "right_ring": 3,
                "right_pinky": 4,
                "left_thumb": 5,
                "left_index": 6,
                "left_middle": 7,
                "left_ring": 8,
                "left_pinky": 9,
            }
            for i in range(len(track_site_ids)):
                track_site_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_ids[i]
                )
                for k, v in contact_map.items():
                    if k in track_site_name and "object" in track_site_name:
                        if v < 5:
                            contact[i] = frame_contact_right[v]
                        else:
                            contact[i] = frame_contact_left[v - 5]
                        break

            mujoco.mj_forward(mj_model, mj_data)
            contact_pos_list.append(mj_data.mocap_pos.copy())
            # get contact list
            # logic: for each track_site, check its corresponding object site (e.g. track site named "track_hand_right_index_tip" should correspond to "track_object_right_index_tip")
            # similarly, "track_object_right_index_tip" should correspond to "track_hand_right_index_tip"
            # after find its corresponding object site, check if the distance between the two sites is less than 0.01, if so, set contact to 1, otherwise set contact to 0
            # contact order should follow track site definition order
            # contact size is equal to check sites number
            # for i in range(len(track_site_ids)):
            #     track_site_id = track_site_ids[i]
            #     track_site_pos = mj_data.site_xpos[track_site_id].copy()
            #     track_site_name = mujoco.mj_id2name(
            #         mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
            #     )
            #     if "hand" in track_site_name:
            #         match_site_name = track_site_name.replace("hand", "object")
            #     elif "object" in track_site_name:
            #         match_site_name = track_site_name.replace("object", "hand")
            #     else:
            #         raise ValueError(f"Invalid track site name: {track_site_name}")
            #     match_site_id = mujoco.mj_name2id(
            #         mj_model, mujoco.mjtObj.mjOBJ_SITE, match_site_name
            #     )
            #     match_site_pos = mj_data.site_xpos[match_site_id].copy()
            #     if np.linalg.norm(track_site_pos - match_site_pos) < 0.01:
            #         contact[i] = 1
            #     else:
            #         contact[i] = 0
            contact_list.append(contact)

            # get contact point distance
            for i in range(len(sites_for_mimic)):
                site_name = sites_for_mimic[i]

            qpos_list.append(mj_data.qpos.copy())
            if save_video:
                opt = mujoco.MjvOption()
                # opt.sitegroup[4] = 1
                renderer.update_scene(data=mj_data, camera="front", scene_option=opt)
                images.append(renderer.render())
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()
            cnt += 1
            if cnt == H:
                cost_mean = cost_sum / H
                if show_viewer:
                    # check if the rollout is good, if so, break
                    user_input = input("Is the rollout good? (y/n): ")
                    if user_input.lower() == "y":
                        break
                    else:
                        cnt = 0
                else:
                    break

        file_dir = processed_dir_robot
        os.makedirs(file_dir, exist_ok=True)
        if save_video:
            imageio.mimsave(
                f"{file_dir}/visualization_ik.mp4",
                images,
                fps=int(1 / ref_dt),
            )
            loguru.logger.info(
                f"Saved visualization video to {file_dir}/visualization_ik.mp4"
            )

        qpos_list = np.array(qpos_list)

        # average filter
        def moving_average_filter(signal_data, window_size=5):
            return np.convolve(
                signal_data, np.ones(window_size) / window_size, mode="valid"
            )

        # Apply moving average filter
        filtered_qpos_list = np.zeros(
            (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
        )
        for i in range(qpos_list.shape[1]):
            filtered_qpos_list[:, i] = moving_average_filter(
                qpos_list[:, i], average_frame_size
            )
        qpos_list = filtered_qpos_list

        def low_pass_filter(signal_data, cutoff_frequency=10, order=4):
            nyquist = 0.5 * (1 / ref_dt)
            normal_cutoff = cutoff_frequency / nyquist
            b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
            return signal.filtfilt(b, a, signal_data)

        # Apply low pass filter
        # for i in range(qpos_list.shape[1]):
        #     qpos_list[:, i] = low_pass_filter(qpos_list[:, i])

        H = qpos_list.shape[0]
        # get qvel
        qvel_list = np.zeros((H - 1, mj_model_ik.nv))
        for i in range(1, H):
            mujoco.mj_differentiatePos(
                mj_model_ik,
                qvel_list[i - 1, :],
                ref_dt,
                qpos_list[i - 1, :],
                qpos_list[i, :],
            )
        qpos_list = qpos_list[1:]
        contact_pos_list = np.array(contact_pos_list)[1:]
        contact_list = np.array(contact_list)[1:]
        # Keep only object-track channels in saved contact/contact_pos.
        # This avoids mixing hand-track channels (zeros) in right/left mode.
        object_track_idx_right = [
            i for i, n in enumerate(track_site_names) if n.startswith("track_object_right_")
        ]
        object_track_idx_left = [
            i for i, n in enumerate(track_site_names) if n.startswith("track_object_left_")
        ]
        if embodiment_type == "right":
            keep_idx = object_track_idx_right
        elif embodiment_type == "left":
            keep_idx = object_track_idx_left
        elif embodiment_type == "bimanual":
            keep_idx = object_track_idx_right + object_track_idx_left
        else:
            raise ValueError(f"Invalid embodiment_type: {embodiment_type}")
        if len(keep_idx) == 0:
            loguru.logger.warning(
                "No track_object_* channels found; keeping original contact channels."
            )
        else:
            contact_list = contact_list[:, keep_idx]
            contact_pos_list = contact_pos_list[:, keep_idx, :]
        assert qpos_list.shape[0] == qvel_list.shape[0]

        # directly rollout ctrl to get qpos_rollout
        mj_model.opt.timestep = ref_dt
        mj_data.qpos[:] = qpos_list[0]
        mj_data.qvel[:] = qvel_list[0]
        mj_data.ctrl[:] = qpos_list[0][:-nq_obj]
        mujoco.mj_step(mj_model, mj_data)
        H = qpos_list.shape[0]
        qpos_rollout = np.zeros((H, mj_model.nq))
        qvel_rollout = np.zeros((H, mj_model.nv))
        qpos_rollout[0] = qpos_list[0]
        for i in range(1, H):
            mj_data.ctrl[:] = qpos_list[i][:-nq_obj]
            noise = np.random.randn(mj_model.nu) * 0.2
            noise[:6] *= 0.0
            noise[22:28] *= 0.0
            mj_data.ctrl[:] += noise
            mujoco.mj_step(mj_model, mj_data)
            qpos_rollout[i] = mj_data.qpos.copy()

        out_npz = f"{file_dir}/trajectory_kinematic.npz"
        np.savez(
            out_npz,
            qpos=qpos_list,
            qpos_rollout=qpos_rollout,
            qvel=qvel_list,
            contact=contact_list,
            contact_pos=contact_pos_list,
            frequency=1 / ref_dt,
        )
        out_npz = f"{file_dir}/trajectory_ikrollout.npz"
        np.savez(
            out_npz,
            qpos=qpos_rollout,
        )
        loguru.logger.info(f"Saved {out_npz}")


if __name__ == "__main__":
    tyro.cli(main)
