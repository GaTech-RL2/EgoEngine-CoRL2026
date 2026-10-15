import glob
import os.path
import time


from urdfpy import URDF
import pybullet as p
import pybullet_data
import numpy as np


class robot_model_urdf:
    def __init__(self, urdf_path):

        robot = URDF.load(urdf_path)
        print("All links:", [l.name for l in robot.links])
        print("All joints:", [j.name for j in robot.joints])

        self.robot = robot

        joint_values = {joint.name: 0.0 for joint in robot.actuated_joints}

        fk_results = robot.link_fk(cfg=joint_values, )
        print(fk_results)
        # Visualize the robot with trimesh viewer
        # trimesh shows the robot with the FK transforms applied
        robot.show(cfg=joint_values)


        assert 0

    def forward_kinematics(self, joint_conf, names):
        assert len(joint_conf) == len(self.robot_chain.links), self.robot_chain
        ret = {}
        for name in names:
            eef_id = self.name2index[name]
            fk_result = self.robot_chain.forward_kinematics(joint_conf, end_effector_link_index=eef_id)
            ret[name] = (fk_result)

        return ret

class robot_model_pybullet:
    def __init__(self, urdf_path, base_pos, base_quat, gui=False):

        if gui:
            p.connect(p.GUI)  # or p.DIRECT for headless
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to find common URDFs

        plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path,
                                   basePosition=base_pos,
                                   baseOrientation=base_quat,
                                   useFixedBase=True)

        self.link_name2id = {}
        self.link_name2id[-1] = "base"

        # Loop through all joints
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")  # child link name
            self.link_name2id[link_name] = i

        self.debug_items = []

    def get_link_poses(self, joint_positions, link_names):

        for idx, pos in enumerate(joint_positions):
            p.resetJointState(self.robot_id, idx, targetValue=pos)

        ret = {}
        for link_name in link_names:
            link_id = self.link_name2id[link_name]
            info = p.getLinkState(bodyUniqueId=self.robot_id,
                                  linkIndex=link_id)
            world_pos = info[4]
            world_ori = info[5]
            mat = p.getMatrixFromQuaternion(world_ori)
            mat = np.array(mat).reshape((3, 3))

            pose = np.eye(4)
            pose[:3, 3] = world_pos
            pose[:3, :3] = mat

            ret[link_name] = pose

        return ret

    def clear_debug_items(self):
        for item in self.debug_items:
            p.removeUserDebugItem(item)
        self.debug_items = []

    def draw_frame(self, pose):
        self.clear_debug_items()

        t = pose[:3, 3]
        R = pose[:3, :3]
        line_len = 0.5
        x_axis = p.addUserDebugLine(lineFromXYZ=t,
                                    lineToXYZ=R[:3, 0] * line_len + t,
                                    lineColorRGB=[1, 0, 0],
                                    lineWidth=0.1,
                                    lifeTime=0,
                                    )
        y_axis = p.addUserDebugLine(lineFromXYZ=t,
                                    lineToXYZ=R[:3, 1] * line_len + t,
                                    lineColorRGB=[0, 1, 0],
                                    lineWidth=0.1,
                                    lifeTime=0,
                                    )
        z_axis = p.addUserDebugLine(lineFromXYZ=t,
                                    lineToXYZ=R[:3, 2] * line_len + t,
                                    lineColorRGB=[0, 0, 1],
                                    lineWidth=0.1,
                                    lifeTime=0,
                                    )
        self.debug_items.extend([x_axis, y_axis, z_axis])

    def visualize_all_poses(self, poses):
        for pose in poses:
            self.draw_frame(pose)


def check_all_shapes(src_dir, ):
    all_data = glob.glob(f"{src_dir}/*.npy")
    for fp in all_data:
        data = np.load(fp, allow_pickle=True)
        print(os.path.basename(fp), data.shape)


def check_data(root_dir, frame_id, env_id):
    src_dir = f"{root_dir}/{frame_id:06d}"

    check_all_shapes(src_dir)

    robot_pos = np.load(f"{src_dir}/rby1_robot_pos.npy", allow_pickle=True)[env_id]
    robot_quat = np.load(f"{src_dir}/rby1_robot_quat.npy", allow_pickle=True)[env_id]

    joint_pos = np.load(f"{src_dir}/rby1_robot_joint_pos.npy", allow_pickle=True)[env_id]

    robot_model = robot_model_pybullet("../assets/rbyxhand/model_modified.urdf", gui=True,
                                       base_pos=robot_pos, base_quat=robot_quat)
    eef_names = ["link_left_arm_6", "left_xhand_palm",
                 # "left_xhand_thumb_tip",
                 # "link_right_arm_6", "right_xhand_palm",
                 # "right_xhand_thumb_tip"
                 ]

    poses = robot_model.get_link_poses(joint_pos, link_names=eef_names)
    print(len(poses))
    robot_model.visualize_all_poses(list(poses.values()))

    for _ in range(100000):
        x = 10**2 + 100
        time.sleep(0.001)


check_data(root_dir="/Users/shuocheng/Downloads/debug_env_18-13-06",
           frame_id=50, env_id=4)