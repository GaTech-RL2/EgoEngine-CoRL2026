import gc

import numpy as np
import sys
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from termcolor import cprint

class robot_model_pk():
    def __init__(self, urdf_path, base_pos, base_quat, isaac_dof_names_ordered, ):

        self.device = "cpu"
        self.urdf_path = urdf_path

        self.base_pos = base_pos
        self.base_quat = base_quat

        robot_pose = np.eye(4)
        robot_pose[:3, 3] = base_pos
        robot_pose[:3, :3] = R.from_quat(base_quat).as_matrix()
        self.robot_pose = robot_pose.astype("float32")

        self.isaac_dof_names_ord = isaac_dof_names_ordered

    def compute_fk(self, joint_positions, link_names):

        chain = pk.build_chain_from_urdf(open(self.urdf_path, "rb").read())
        pk_joint_names_ord = chain.get_joint_parameter_names()

        assert len(joint_positions.shape) == 2, joint_positions.shape
        with torch.no_grad():
            query_q = []
            for jname in pk_joint_names_ord:
                ind = self.isaac_dof_names_ord.index(jname)
                query_q.append(joint_positions[:, ind])
            query_q = np.array(query_q).T
            ret = chain.forward_kinematics(query_q, )
            robot_pose = self.robot_pose[None]

            res = {}
            for k in ret:
                if k not in link_names:
                    continue

                link_pose = ret[k].get_matrix().cpu().numpy()
                res[k] = robot_pose @ link_pose
                # transform to world frame

        gc.collect()

        return res

    def _sample_configs(self, seed_q, num_samp, delta, q_low, q_high):
        assert len(seed_q.shape) == 1, seed_q.shape

        n = len(seed_q)
        samples = np.random.uniform(0, 1, size=(num_samp, n))

        low_ = np.array(seed_q - delta * 0.5)
        high_ = np.array(seed_q + delta * 0.5)

        low_[low_<q_low] = q_low[low_<q_low]
        high_[high_>q_high] = q_high[high_>q_high]

        low_ = low_.reshape((1, n))
        high_ = high_.reshape((1, n))

        samp_q = (high_ - low_) * samples + low_

        return samp_q.astype("float32")

    def compute_ik(self, target_pose_world, link_name, init_q, fixed_dof_names,
                   num_trials=128, max_iter=500, root_link="", bias_sampling=True):


        assert len(init_q.shape) == 1, init_q.shape
        assert len(init_q) == len(self.isaac_dof_names_ord)

        with torch.no_grad():

            chain = pk.build_serial_chain_from_urdf(open(self.urdf_path, "rb").read(),
                                                    link_name,
                                                    root_link_name=root_link)

            active_joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
            q_low, q_high = chain.get_joint_limits()

            assert len(q_low) == len(active_joint_names)
            # modify bound
            for fixed_dof_name in fixed_dof_names:
                if fixed_dof_name in active_joint_names:
                    to_ind = active_joint_names.index(fixed_dof_name)
                    from_ind = self.isaac_dof_names_ord.index(fixed_dof_name)
                    val = init_q[from_ind]
                    q_low[to_ind] = q_high[to_ind] = val

            q_low = np.array(q_low)
            q_high = np.array(q_high)
            joint_limits = torch.tensor([q_low, q_high]).T

            world2robot = np.linalg.inv(self.robot_pose)
            target_pose_robot = (world2robot @ target_pose_world).astype("float32")

            if bias_sampling:

                seed_q = [init_q[self.isaac_dof_names_ord.index(name)] for name in active_joint_names]
                seed_q = np.array(seed_q)
                init_configs = self._sample_configs(seed_q=seed_q, num_samp=num_trials,
                                                    delta=np.pi/3, q_low=q_low, q_high=q_high)
                init_configs = torch.tensor(init_configs)
            else:
                init_configs = None

            # create the IK object
            # see the constructor for more options and their explanations, such as convergence tolerances
            ik = pk.PseudoInverseIK(chain,
                                    max_iterations=max_iter,
                                    retry_configs=init_configs,
                                    num_retries=num_trials,
                                    joint_limits=joint_limits,
                                    pos_tolerance=0.005,
                                    rot_tolerance=0.01,
                                    early_stopping_any_converged=False,
                                    early_stopping_no_improvement="all",
                                    optimizer_method="Adam",
                                    debug=False,
                                    lr=0.05)

            pos = torch.tensor(target_pose_robot[None, :3, 3])
            rot = torch.tensor(target_pose_robot[None, :3, :3])
            target_pose_robot_tf = pk.Transform3d(pos=pos, rot=rot, )
            sol = ik.solve(target_pose_robot_tf)

            assert sol.solutions.shape[0] == 1, sol.solutions.shape

            if torch.all(~sol.converged):
                partial_solutions = sol.solutions[0]
                cprint("[Warning]: No Converged Solution!", "red")
            else:
                partial_solutions = sol.solutions[0, sol.converged[0]]
                cprint(f"Found {len(partial_solutions)} solutions!", "cyan")

            M = partial_solutions.shape[0]
            candidates = np.array(init_q)[None].repeat(repeats=M, axis=0)

            for i, dof_name in enumerate(active_joint_names):
                ind = self.isaac_dof_names_ord.index(dof_name)
                candidates[:, ind] = partial_solutions[:, i]

            sol_dist = np.linalg.norm(init_q - candidates, axis=1, keepdims=False)
            best_sol_id = np.argmin(sol_dist)
            best_sol = candidates[best_sol_id]

        gc.collect()

        return best_sol


    def compute_partial_ik(self, target_pose_world, link_name, init_q_dict, fixed_dof_names,
                           pos_tol=0.005, rot_tol=0.01, num_trials=128, max_iter=500,
                           root_link="", bias_sampling=True):

        # assume the root link is at origin

        with torch.no_grad():
            chain = pk.build_serial_chain_from_urdf(open(self.urdf_path, "rb").read(),
                                                    link_name,
                                                    root_link_name=root_link)


            active_joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
            q_low, q_high = chain.get_joint_limits()
            if root_link:
                init_q = np.array([init_q_dict[name] for name in active_joint_names])
                hand_poses = chain.forward_kinematics(init_q, False)

                print(hand_poses)

                root_rel_T = hand_poses[root_link].get_matrix().cpu().numpy()[0]
                target_pose_world = root_rel_T @ np.array(target_pose_world)

                print(root_rel_T, '??')

            assert len(q_low) == len(active_joint_names)

            # modify bound
            for fixed_dof_name in fixed_dof_names:
                if fixed_dof_name in active_joint_names:
                    to_ind = active_joint_names.index(fixed_dof_name)
                    val = init_q_dict[fixed_dof_name]
                    q_low[to_ind] = q_high[to_ind] = val

            q_low = np.array(q_low)
            q_high = np.array(q_high)
            joint_limits = np.array([q_low, q_high])
            joint_limits = torch.tensor(joint_limits).T

            world2robot = np.linalg.inv(self.robot_pose)

            target_pose_robot = (world2robot @ target_pose_world).astype("float32")

            if bias_sampling:
                seed_q = [init_q_dict[name] for name in active_joint_names]
                seed_q = np.array(seed_q)
                init_configs = self._sample_configs(seed_q=seed_q, num_samp=num_trials,
                                                    delta=np.pi/3, q_low=q_low, q_high=q_high)
                init_configs = torch.tensor(init_configs)
            else:
                init_configs = None

            # create the IK object
            # see the constructor for more options and their explanations, such as convergence tolerances
            ik = pk.PseudoInverseIK(chain,
                                    max_iterations=max_iter,
                                    retry_configs=init_configs,
                                    num_retries=num_trials,
                                    joint_limits=joint_limits,
                                    pos_tolerance=pos_tol,
                                    rot_tolerance=rot_tol,
                                    early_stopping_any_converged=False,
                                    early_stopping_no_improvement="all",
                                    optimizer_method="Adam",
                                    debug=False,
                                    lr=0.01)

            pos = torch.tensor(target_pose_robot[None, :3, 3])
            rot = torch.tensor(target_pose_robot[None, :3, :3])
            target_pose_robot_tf = pk.Transform3d(pos=pos, rot=rot, )
            sol = ik.solve(target_pose_robot_tf)

            assert sol.solutions.shape[0] == 1, sol.solutions.shape

            print(sol.err_pos, '?')

            if torch.all(~sol.converged):
                partial_solutions = sol.solutions[0]
                cprint("[Warning]: No Converged Solution!", "red")
            else:
                partial_solutions = sol.solutions[0, sol.converged[0]]
                cprint(f"Found {len(partial_solutions)} solutions!", "cyan")

            partial_solutions = partial_solutions.cpu().numpy()



            seed_q = np.array([init_q_dict[name] for name in active_joint_names])
            sol_dist = np.linalg.norm(seed_q - partial_solutions, axis=1, keepdims=False)
            best_sol_id = np.argmin(sol_dist)
            best_sol = partial_solutions[best_sol_id]

            best_sol_dict = dict(zip(active_joint_names, best_sol.tolist()))

        gc.collect()
        return best_sol_dict


if __name__ == '__main__':

    urdf_path = "assets/rbyxhand/model_modified.urdf"

    dof_names = [
        "left_wheel",
        "right_wheel",
        "torso_0",
        "torso_1",
        "torso_2",
        "torso_3",
        "torso_4",
        "head_0",
        "head_1",
        "left_arm_0",
        "left_arm_1",
        "left_arm_2",
        "left_arm_3",
        "left_arm_4",
        "left_arm_5",
        "left_arm_6",
        "left_xhand_index_q0",
        "left_xhand_index_q1",
        "left_xhand_index_q2",
        "left_xhand_middle_q1",
        "left_xhand_middle_q2",
        "left_xhand_pinky_q1",
        "left_xhand_pinky_q2",
        "left_xhand_ring_q1",
        "left_xhand_ring_q2",
        "left_xhand_thumb_q0",
        "left_xhand_thumb_q1",
        "left_xhand_thumb_q2",
        "right_arm_0",
        "right_arm_1",
        "right_arm_2",
        "right_arm_3",
        "right_arm_4",
        "right_arm_5",
        "right_arm_6",
        "right_xhand_index_q0",
        "right_xhand_index_q1",
        "right_xhand_index_q2",
        "right_xhand_middle_q1",
        "right_xhand_middle_q2",
        "right_xhand_pinky_q1",
        "right_xhand_pinky_q2",
        "right_xhand_ring_q1",
        "right_xhand_ring_q2",
        "right_xhand_thumb_q0",
        "right_xhand_thumb_q1",
        "right_xhand_thumb_q2"
    ]

    robot_model = robot_model_pk(urdf_path=urdf_path,
                                 base_pos=[-1, 0, 0],
                                 base_quat=[0, 0, 0, 1],
                                 isaac_dof_names_ordered=dof_names)

    root_dir = "/Users/shuocheng/Downloads/debug_track_20-40-21"
    frame_id = 34
    get_full_path = lambda x: f"{root_dir}/{frame_id:06d}/{x}.npy"

    def check_fk():
        link_names = ["right_xhand_base_link", "right_xhand_back_cover"]
        joint_pos = np.load(get_full_path("rby1_robot_joint_pos"))[0]

        # print("check jpos: ", repr(joint_pos))
        # plt.plot(joint_pos)
        # plt.show()

        joint_pos = joint_pos.reshape((1, -1))
        eef_poses = robot_model.compute_fk(joint_pos, link_names=link_names)
        for name in link_names:
            gt_pose = np.load(get_full_path(f"rby1_robot_{name}_pose"))[0]
            print(eef_poses[name])
            print(gt_pose)
    check_fk()

    def check_ik():
        link_names = ["right_xhand_base_link", "right_xhand_back_cover"]
        joint_pos = np.load(get_full_path("rby1_robot_joint_pos"))[0]

        fixed_names = ["left_wheel", "right_wheel", "torso_0", "torso_1", "torso_2",
                       "torso_3", "torso_4", "head_0", "head_1",]

        for name in link_names:
            gt_pose = np.load(get_full_path(f"rby1_robot_{name}_pose"))[0]
            gt_pose = np.array(gt_pose, dtype="float32")

            est_q = robot_model.compute_ik(gt_pose, name, joint_pos, fixed_dof_names=fixed_names)

            print(joint_pos)
            print(est_q)

            plt.plot(joint_pos)
            plt.plot(est_q)
            plt.show()




    check_ik()
