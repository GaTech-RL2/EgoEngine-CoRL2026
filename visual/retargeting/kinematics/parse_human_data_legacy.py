import sys

import numpy as np
import pickle
import torch
import mano
from mano.utils import Mesh
from scipy.spatial.transform import Rotation as R
# import open3d as o3d
from glob import glob
import os
import os.path as osp
import trimesh
from pb_visualizer import robot_model_pk, PybulletVisualizer
from rby1_kinematics import track_one_traj


def create_obj_urdf(obj_path, scale=(1, 1, 1)):
    """
    Create a minimal URDF file for a given .obj mesh.

    Args:
        obj_path (str): Absolute or relative path to the .obj file.
        scale (tuple): Scale to apply to the mesh.

    Returns:
        urdf_path (str): Path to the generated URDF file.
    """
    obj_path = os.path.abspath(obj_path)
    mesh_dir = os.path.dirname(obj_path)
    mesh_name = os.path.basename(obj_path)

    urdf_content = f"""<?xml version="1.0"?>
<robot name="dynamic_obj">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_name}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_name}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

    urdf_name = mesh_name.replace(".obj", ".urdf")
    # Create temp directory to store URDF and copy OBJ alongside
    urdf_path = f"{mesh_dir}/{urdf_name}"

    # Write URDF
    with open(urdf_path, "w") as f:
        f.write(urdf_content)

    return urdf_path

def estimate_human_base_frame(tool_pose, target_pose):

    x_offset = 0.8 # this should consider the reachability of robot arm
    z_offset = 0.8

    xy_vec = tool_pose[:3, 3] - target_pose[:3, 3]
    # xy_vec[2] = 0

    y_axis = -xy_vec / (np.linalg.norm(xy_vec) + 1e-18)
    Rz_cw = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    x_axis = (Rz_cw @ y_axis.reshape((3, 1)))[:, 0]
    z_axis = np.cross(x_axis, y_axis)

    base_rot = np.array([x_axis, y_axis, z_axis]).T

    cent_pos = (tool_pose[:3, 3] + target_pose[:3, 3]) / 2.
    # cent_pos[2] = 0

    base_pos = cent_pos + (-x_axis * x_offset) + (-z_axis * z_offset)

    base_pose = np.eye(4)
    base_pose[:3, 3] = base_pos
    base_pose[:3, :3] = base_rot
    return base_pose

def transfer_poses_to_local(lh_seq, rh_seq, tool_pose, target_pose, base_pose):
    def proj_dict(pose_dict, base_pose):
        ret = {}
        for k, v in pose_dict.items():
            ret[k] = np.linalg.inv(base_pose) @ v
        return ret

    for i in range(len(lh_seq)):
        lh_seq[i] = proj_dict(lh_seq[i], base_pose)
        rh_seq[i] = proj_dict(rh_seq[i], base_pose)

    tool_pose = np.linalg.inv(base_pose)[None] @ tool_pose
    target_pose = np.linalg.inv(base_pose)[None] @ target_pose

    return lh_seq, rh_seq, tool_pose, target_pose


def load_data(data_dir, mesh_dir, viz=False):

    target_pose_path = glob(f"{data_dir}/target_*.npy")[0]
    tool_pose_path = glob(f"{data_dir}/tool_*.npy")[0]

    target_obj_id = osp.basename(target_pose_path).split(".")[0].split("_")[1]
    tool_obj_id = osp.basename(tool_pose_path).split(".")[0].split("_")[1]
    target_obj_path = f"{mesh_dir}/{target_obj_id}_cm.obj"
    tool_obj_path = f"{mesh_dir}/{tool_obj_id}_cm.obj"

    target_urdf = create_obj_urdf(target_obj_path, scale=(0.01, 0.01, 0.01))
    tool_urdf = create_obj_urdf(tool_obj_path, scale=(0.01, 0.01, 0.01))

    left_hand_path = f"{data_dir}/left_hand.pkl"
    right_hand_path = f"{data_dir}/right_hand.pkl"

    left_hand = pickle.load(open(left_hand_path, "rb"))
    right_hand = pickle.load(open(right_hand_path, "rb"))

    lh_joint_seq = []
    rh_joint_seq = []

    for i, ((_, lv), (_, rv)) in enumerate(zip(left_hand.items(), right_hand.items())):

        lh_targets = generate_hand_model(hand_pose=lv["hand_pose"],
                                         hand_trans=lv["hand_trans"],
                                         is_rhand=False,
                                         vis=viz)

        rh_targets = generate_hand_model(hand_pose=rv["hand_pose"],
                                         hand_trans=rv["hand_trans"],
                                         is_rhand=True,
                                         vis=viz)

        lh_joint_seq.append(lh_targets)
        rh_joint_seq.append(rh_targets)


    target_pose = np.load(target_pose_path)
    tool_pose = np.load(tool_pose_path)

    print("target pose: ", target_pose[0])
    print("tool pose: ", tool_pose[0])
    human_base_pose = estimate_human_base_frame(tool_pose=tool_pose[0], target_pose=target_pose[0])
    print("human base pose: ", human_base_pose)

    lh_joint_seq, rh_joint_seq, tool_pose, target_pose = transfer_poses_to_local(
        lh_joint_seq, rh_joint_seq, tool_pose, target_pose, human_base_pose
    )

    target_quat = R.from_matrix(target_pose[:, :3, :3]).as_quat()
    tool_quat = R.from_matrix(tool_pose[:, :3, :3]).as_quat()

    target_pose = np.concatenate([target_pose[:, :3, 3], target_quat], axis=1)
    tool_pose = np.concatenate([tool_pose[:, :3, 3], tool_quat], axis=1)

    return {"lh_seq": lh_joint_seq, "rh_seq": rh_joint_seq,
            "target_obj_pose": target_pose, "tool_obj_pose": tool_pose,
            "target_urdf": target_urdf, "tool_urdf": tool_urdf
            }

def generate_hand_model(hand_pose, hand_trans, is_rhand, vis=True):
    def _add_pose(pose):
        frame = trimesh.creation.axis(origin_size=0.005, axis_length=0.01)
        frame.apply_transform(pose)

        return frame

    if is_rhand:
        model_path = './models/MANO_RIGHT.pkl'
    else:
        model_path = "./models/MANO_LEFT.pkl"

    n_comps = 45
    batch_size = 1

    rh_model = mano.load(model_path=model_path,
                         is_rhand=is_rhand,
                         num_pca_comps=n_comps,
                         batch_size=batch_size,
                         flat_hand_mean=False)

    betas = torch.zeros(batch_size, 10) * .1
    pose = hand_pose[3:].reshape((batch_size, n_comps))
    global_orient = hand_pose[:3].reshape((batch_size, 3))
    transl = hand_trans.reshape((batch_size, 3))

    output = rh_model(betas=betas,
                      global_orient=global_orient,
                      hand_pose=pose,
                      transl=transl,
                      return_verts=True,
                      return_tips=True)

    joint_trs = output.joints[0]
    mano_joint_names = [
        "wrist",  # 0
        "thumb1",  # 1
        "thumb2",  # 2
        "thumb3",  # 3
        "index1",  # 4
        "index2",  # 5
        "index3",  # 6
        "middle1",  # 7
        "middle2",  # 8
        "middle3",  # 9
        "ring1",  # 10
        "ring2",  # 11
        "ring3",  # 12
        "pinky1",  # 13
        "pinky2",  # 14
        "pinky3",  # 15
        "thumb_tip",  # 16
        "index_tip",  # 17
        "middle_tip",  # 18
        "ring_tip",  # 19
        "pinky_tip"  # 20
    ]

    mats = np.array([R.from_rotvec(hand_pose[i*3:i*3+3]).as_matrix() for i in range(n_comps//3+1)]) # (16, 3, 3)
    se3_poses = np.expand_dims(np.eye(4), axis=0).repeat(repeats=21, axis=0)
    se3_poses[:len(mats), :3, :3] = mats
    se3_poses[:, :3, 3] = joint_trs

    # rotate pose for robot
    if is_rhand:
        h2r_base = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    else:
        h2r_base = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    se3_poses[0, :3, :3] = se3_poses[0, :3, :3] @ np.linalg.inv(h2r_base)

    # adjust wrist
    z_offset = -0.08
    adj_wrist_offset = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, z_offset], [0, 0, 0, 1]])
    se3_poses[0] = se3_poses[0] @ adj_wrist_offset

    ret = {}
    for i, name in enumerate(mano_joint_names):
        ret[name] = se3_poses[i]

    if vis:
        scene = trimesh.Scene()
        pose_meshes = [_add_pose(p) for p in se3_poses]
        h_meshes = rh_model.hand_meshes(output)  #
        for mesh in pose_meshes + h_meshes:
            scene.add_geometry(mesh)
        scene.show()

    return ret

def retarget_one_frame(left_hand_model, right_hand_model, rby1_model, init_q_dict,
                       rh_targets, lh_targets, link_mapping):
    res = {}

    hand_retarget = ["thumb_tip", "index_tip", "middle_tip",
                     "ring_tip", "pinky_tip", ]
    right_hand_base_pose = rh_targets["wrist"]
    left_hand_base_pose = lh_targets["wrist"]

    pos_th = 0.01
    rot_th = np.pi*2
    num_trials = 128

    for k in hand_retarget:
        robo_link_name = link_mapping[k].format("right")
        v = (np.linalg.inv(right_hand_base_pose) @ rh_targets[k]).astype("float32")
        # the target pose must be in base link coordinate! the original target are represented in (0, 0, 0) coordinate,
        # so make sure the base link in urdf also centered at (0, 0, 0)
        sol = right_hand_model.compute_partial_ik(target_pose_robot=v,
                                                  link_name=robo_link_name,
                                                  init_q_dict=init_q_dict,
                                                  num_trials=num_trials,
                                                  pos_tol=pos_th,
                                                  rot_tol=rot_th,
                                                  fixed_dof_names=[],
                                                  root_link="right_hand_link"
                                                  )
        res.update(sol)

    # for k in hand_retarget:
    #     robo_link_name = link_mapping[k].format("left")
    #     v = (np.linalg.inv(left_hand_base_pose) @ lh_targets[k]).astype("float32")
    #     # the target pose must be in base link coordinate! the original target are represented in (0, 0, 0) coordinate,
    #     # so make sure the base link in urdf also centered at (0, 0, 0)
    #     sol = left_hand_model.compute_partial_ik(target_pose_robot=v,
    #                                              link_name=robo_link_name,
    #                                              init_q_dict=init_q_dict,
    #                                              num_trials=num_trials,
    #                                              pos_tol=pos_th,
    #                                              rot_tol=rot_th,
    #                                              fixed_dof_names=[],
    #                                              root_link=f"left_hand_link",
    #                                              )
    #     res.update(sol)

    # left_ee_link = link_mapping["wrist"].format("left")
    # sol = rby1_model.compute_partial_ik(target_pose_robot=left_hand_base_pose,
    #                                      link_name=left_ee_link,
    #                                      init_q_dict=init_q_dict,
    #                                      num_trials=num_trials,
    #                                      pos_tol=0.005,
    #                                      rot_tol=0.01,
    #                                      fixed_dof_names=["torso_3", "torso_4"],
    #                                      bias_sampling=True,
    #                                      # root_link="base_link",
    #                                      )
    # res.update(sol)

    init_q_dict.update(res)

    right_ee_link = link_mapping["wrist"].format("right")
    sol = rby1_model.compute_partial_ik(target_pose_robot=right_hand_base_pose,
                                        link_name=right_ee_link,
                                        init_q_dict=init_q_dict,
                                        num_trials=num_trials,
                                        pos_tol=0.005,
                                        rot_tol=0.01,
                                        fixed_dof_names=["torso_3", "torso_4"],
                                        bias_sampling=True,
                                        do_clipping=False,
                                        # root_link="base_link",
                                        )
    res.update(sol)

    for k in init_q_dict:
        if k not in res:
            res[k] = init_q_dict[k]

    return res

def retarget_to_robot():
    data = load_data(data_dir="/Users/shuocheng/Downloads/Hand_Poses/(brush, brush, box)/20230928_046",
                     mesh_dir="/Users/shuocheng/Downloads/object_models_released")

    left_hand_joints = ['left_hand_thumb_bend_joint', 'left_hand_thumb_rota_joint1', 'left_hand_thumb_rota_joint2',
                        'left_hand_index_bend_joint', 'left_hand_index_joint1', 'left_hand_index_joint2',
                        'left_hand_mid_joint1', 'left_hand_mid_joint2', 'left_hand_ring_joint1',
                        'left_hand_ring_joint2', 'left_hand_pinky_joint1', 'left_hand_pinky_joint2']
    right_hand_joints = ['right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2',
                         'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
                         'right_hand_mid_joint1', 'right_hand_mid_joint2', 'right_hand_ring_joint1',
                         'right_hand_ring_joint2', 'right_hand_pinky_joint1', 'right_hand_pinky_joint2']
    rby1_joints = ["left_wheel", "right_wheel", "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "head_0",
                   "head_1","left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5",
                   "left_arm_6", "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
                   "right_arm_5", "right_arm_6",]


    init_q = np.array([0] * (len(right_hand_joints+right_hand_joints+rby1_joints)), dtype="float32")

    hand2robot_link_mapping = {"thumb_tip": "{}_hand_thumb_rota_tip",
                               "index_tip": "{}_hand_index_rota_tip",
                               "middle_tip": "{}_hand_mid_tip",
                               "ring_tip": "{}_hand_ring_tip",
                               "pinky_tip": "{}_hand_pinky_tip",
                               "wrist": "{}_hand_link",
                               }
    rh_seq = data["rh_seq"]
    lh_seq = data["lh_seq"]

    left_hand_urdf = "./assets/xhand/xhand_left_new/urdf/xhand_left.urdf"
    right_hand_urdf = "./assets/xhand/xhand_right_new/urdf/xhand_right.urdf"
    # intg_robot_urdf = "./assets/rbyxhand2/robot/model_modified.urdf"
    intg_robot_urdf = "./assets/rbyxhand_v2/model_modified.urdf"

    left_hand_model = robot_model_pk(urdf_path=left_hand_urdf)
    right_hand_model = robot_model_pk(urdf_path=right_hand_urdf)
    rby1_model = robot_model_pk(urdf_path=intg_robot_urdf)

    init_q_dict = dict(zip(left_hand_joints + right_hand_joints + rby1_joints, init_q.tolist()))
    init_q_dict["torso_0"] = -0.6
    init_q_dict["torso_1"] = 1.2
    init_q_dict["torso_2"] = -0.6

    target_pose = data["target_obj_pose"]
    tool_pose = data["tool_obj_pose"]

    # robot_viz = PybulletVisualizer([intg_robot_urdf], [[0, 0, 0]], [[0, 0, 0, 1]])
    # robot_viz = PybulletVisualizer([right_hand_urdf, left_hand_urdf])
    robot_viz = PybulletVisualizer({"robot": intg_robot_urdf,
                                              "target": data["target_urdf"],
                                              "tool": data["tool_urdf"]},
                                   offsets=[
                                       [0, 0, 0],
                                       target_pose[0, :3].tolist(),
                                       tool_pose[0, :3].tolist()
                                   ],
                                   quats=[
                                       [0, 0, 0, 1],
                                       target_pose[0, 3:].tolist(),
                                       tool_pose[0, 3:].tolist()
                                   ])


    # visualize init scenario
    # robot_viz.set_joint_configuration(init_q_dict)
    # robot_viz.run()

    retarget_traj = []

    n = len(rh_seq)
    for i in range(n):
        rh_targets = rh_seq[i]
        lh_targets = lh_seq[i]

        res = retarget_one_frame(left_hand_model=left_hand_model,
                                 right_hand_model=right_hand_model,
                                 rby1_model=rby1_model,
                                 init_q_dict=init_q_dict,
                                 rh_targets=rh_targets,
                                 lh_targets=lh_targets,
                                 link_mapping=hand2robot_link_mapping
                                 )

        robot_viz.set_joint_configuration(res)
        robot_viz.set_object_pose(tool_pose[i], "tool")

        reached_poses = rby1_model.compute_fk(res, ["left_hand_link", "right_hand_link"])
        reached_poses = list(reached_poses.values())

        robot_viz.visualize_poses([rh_targets["wrist"], lh_targets["wrist"]]+reached_poses)

        # Run the simulation for 10 seconds
        robot_viz.run(duration=1)

        init_q_dict = res.copy()
        retarget_traj.append(res)



if __name__ == '__main__':

    retarget_to_robot()
