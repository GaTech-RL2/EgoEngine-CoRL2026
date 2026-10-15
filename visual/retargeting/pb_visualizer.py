import pybullet as p
import pybullet_data
import time
import os
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import gc
from termcolor import cprint
from collections import namedtuple


Entity = namedtuple('Entity', ['urdf', 'name', 'pose', 'color', 'fix_base'])

class PoseVisualizer:
    def __init__(self):
        self.debug_items = []

    def draw_pose(self, pose, axis_length=0.2, duration=0, line_width=2, clear_previous=False):
        """
        Draws a pose as a coordinate frame in PyBullet.

        Args:
            pose (np.array): 4x4 transformation matrix.
            axis_length (float): Length of each axis.
            duration (float): How long the frame lasts (0 = forever).
            line_width (float): Line width.
            clear_previous (bool): If True, removes previously drawn frames.
        """
        if clear_previous:
            self.clear()

        origin = pose[:3, 3]
        R = pose[:3, :3]

        x_axis = origin + R[:, 0] * axis_length
        y_axis = origin + R[:, 1] * axis_length
        z_axis = origin + R[:, 2] * axis_length

        self.debug_items.append(p.addUserDebugLine(origin, x_axis, [1, 0, 0], lineWidth=line_width, lifeTime=duration))  # Red
        self.debug_items.append(p.addUserDebugLine(origin, y_axis, [0, 1, 0], lineWidth=line_width, lifeTime=duration))  # Green
        self.debug_items.append(p.addUserDebugLine(origin, z_axis, [0, 0, 1], lineWidth=line_width, lifeTime=duration))  # Blue

    def clear(self):
        for item in self.debug_items:
            p.removeUserDebugItem(item)
        self.debug_items.clear()

    def draw_points(self, points, colors, point_size=6, duration=0, clear_previous=False):
        if clear_previous:
            self.clear()
        if len(points) == 0:
            return
        pts = [list(pt) for pt in points]
        cols = [list(c) for c in colors]
        self.debug_items.append(
            p.addUserDebugPoints(pts, cols, pointSize=point_size, lifeTime=duration)
        )


def R_geodesic_angle(R1, R2):
    R = R1.T @ R2
    cos_theta = (np.trace(R) - 1.0) * 0.5
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

class PybulletVisualizer:
    def __init__(self, entities: [Entity]):
        """
        Initialize the PyBullet visualizer.

        Args:
            urdf_path (str): Path to the robot URDF file.
            use_fixed_base (bool): Whether the robot should have a fixed base.
        """
        # self.physics_client = p.connect(p.GUI)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
        self.all_ids = {}

        for i, entity in enumerate(entities):
            entity: Entity
            offset = entity.pose[:3, 3]
            quat = R.from_matrix(entity.pose[:3, :3]).as_quat()
            robot_id = p.loadURDF(entity.urdf, basePosition=offset,
                                  baseOrientation=quat, useFixedBase=entity.fix_base)
            self.all_ids[entity.name] = robot_id

            if entity.color is not None:
                p.changeVisualShape(robot_id, -1, rgbaColor=entity.color)

        self._disable_robot_object_collisions()
        self.name2joint = self._get_movable_joint_indices()

        self.pose_viz = PoseVisualizer()

    def _disable_robot_object_collisions(self):
        if "robot" not in self.all_ids:
            return
        robot_ids = [self.all_ids["robot"]] if isinstance(self.all_ids["robot"], int) else self.all_ids["robot"]
        object_names = [name for name in ("tool", "target") if name in self.all_ids]
        for robot_id in robot_ids:
            robot_links = [-1] + list(range(p.getNumJoints(robot_id)))
            for object_name in object_names:
                object_id = self.all_ids[object_name]
                object_links = [-1] + list(range(p.getNumJoints(object_id)))
                for robot_link in robot_links:
                    for object_link in object_links:
                        p.setCollisionFilterPair(robot_id, object_id, robot_link, object_link, 0)


    def _get_movable_joint_indices(self):
        all_indices = {}

        robot_ids = [self.all_ids["robot"]] if isinstance(self.all_ids["robot"], int) else self.all_ids["robot"]

        for robot_id in robot_ids:
            indices = {}
            num_joints = p.getNumJoints(robot_id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)
                joint_type = joint_info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_name = joint_info[1].decode('utf-8')
                    indices[joint_name] = i
            all_indices[robot_id] = indices

        return all_indices

    def set_joint_configuration(self, joint_values):
        """
        Set the robot joint values.

        Args:
            joint_values (list): A list of joint positions (in radians or meters).
        """

        for k, v in joint_values.items():
            robot_id = None
            joint_id = None
            for rid, indices in self.name2joint.items():
                if k in indices:
                    joint_id = indices[k]
                    robot_id = rid
                    break
            if robot_id is None:
                print(f"skip {k}, not found in robot!")
                continue
            print(joint_id, v)
            p.resetJointState(robot_id, joint_id, v)

    def control_robot(self, joint_values, n=240):
        robot_id = self.all_ids["robot"]

        robot_name2indices = self.name2joint[robot_id]
        indices2name = {v:k for k, v in robot_name2indices.items()}
        all_vs = sorted(list(robot_name2indices.values()))

        control_targets = []
        joint_indices = []
        for v in all_vs:
            val = joint_values[indices2name[v]]
            joint_indices.append(v)
            control_targets.append(val)

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=control_targets,
            forces=[500] * len(joint_indices)  # Set appropriate max forces
        )
        for i in range(int(n)):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def set_object_poses(self, poses: dict):
        for k, v in poses.items():
            obj_id = self.all_ids[k]
            trs = v[:3, 3]
            quat = R.from_matrix(v[:3, :3]).as_quat()
            p.resetBasePositionAndOrientation(obj_id, trs, quat)

    def get_object_pose(self, object_names):
        res = {}
        for k in object_names:
            body_id = self.all_ids[k]
            pos, orn = p.getBasePositionAndOrientation(body_id)
            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = R.from_quat(orn).as_matrix()
            res[k] = pose
        return res

    def visualize_poses(self, poses, ax_len):
        for i, pose in enumerate(poses):
            if i == 0:
                self.pose_viz.draw_pose(pose, axis_length=ax_len, clear_previous=True)
            else:
                self.pose_viz.draw_pose(pose, axis_length=ax_len, clear_previous=False)

    def visualize_tip_points(self, points, colors, point_size=6):
        self.pose_viz.draw_points(points, colors, point_size=point_size, clear_previous=False)

    def run(self, duration=None):
        """
        Run the simulation for visualization.

        Args:
            duration (float or None): Time in seconds to run the sim. If None, runs indefinitely.
        """
        start_time = time.time()
        while duration is None or time.time() - start_time < duration:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    
    def wait_until_all_stable(self,
                              v_thresh=0.02,  # m/s
                              w_thresh=0.1,  # rad/s
                              settle_steps=10,  # consecutive steps under thresholds
                              max_steps=240 * 6,  # hard cap, 10 s
                              contact_mode="any"  # "none" | "any" | "allowed"
                              ):
        """
        Step sim until *all* bodies are 'stable'.
        Returns: {body_id: (pos, orn)}
        contact_mode:
          - "none": don't require contact
          - "any": body must touch *something*
          - "allowed": body must touch ground_id or another body in body_ids
        """
        body_ids = []
        for name, oid in self.all_ids.items():
            if name in ["target", "tool"]:
                body_ids.append(oid)

        ground_id = self.plane_id

        stable_counts = {bid: 0 for bid in body_ids}
        allowed = set(body_ids)
        if ground_id is not None:
            allowed.add(ground_id)

        good_now = False

        for _ in range(int(max_steps)):
            p.stepSimulation()
            all_ok = True

            for bid in body_ids:
                try:
                    lin_v, ang_v = p.getBaseVelocity(bid)
                except:
                    continue

                if lin_v is None:  # removed/disabled
                    stable_counts[bid] = 0
                    all_ok = False
                    print("lin_v is none!")
                    continue

                init_pose = self.id2init_pose[bid]
                cur_pos, cur_quat = p.getBasePositionAndOrientation(bid)

                if np.linalg.norm(np.array(cur_pos) - init_pose[:3, 3]) > 0.3:
                    return False

                cur_mat = R.from_quat(cur_quat).as_matrix()
                rot_dis = R_geodesic_angle(cur_mat, init_pose[:3, :3])
                if abs(rot_dis) > np.deg2rad(30):
                    return False

                v_ok = np.linalg.norm(lin_v) < v_thresh
                w_ok = np.linalg.norm(ang_v) < w_thresh
                ok = v_ok and w_ok

                # print("lin_v and ang_v: ", np.linalg.norm(lin_v).round(3), v_ok,
                #                            np.linalg.norm(ang_v).round(3), w_ok)

                if contact_mode != "none":
                    if contact_mode == "any":
                        contact_ok = False
                        _contacts = p.getContactPoints(bodyA=bid)
                        if (_contacts is not None) and len(_contacts) > 0:
                            contact_ok = True
                    else:  # "allowed"
                        contact_ok = False
                        for c in p.getContactPoints(bodyA=bid):
                            other = c[2]  # bodyUniqueIdB
                            if other in allowed and other != bid:
                                contact_ok = True
                                break
                    ok = ok and contact_ok

                    # print("contact ok? ", contact_ok)

                if ok:
                    stable_counts[bid] += 1
                else:
                    stable_counts[bid] = 0
                    all_ok = False

            if all(stable_counts[bid] >= settle_steps for bid in body_ids) and all_ok:
                good_now = True
                break

        return good_now

    def disconnect(self):
        """
        Disconnect from the PyBullet simulation.
        """
        p.disconnect()


class robot_model_pk():
    def __init__(self, urdf_path):

        self.urdf_path = urdf_path
        self.device = "cpu"
        chain = pk.build_chain_from_urdf(open(self.urdf_path, "rb").read())
        chain.print_tree()
        active_joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
        self.active_joint_names = active_joint_names
        q_low, q_high = chain.get_joint_limits()

        print("joint names: ", active_joint_names)
        print(q_low)
        print(q_high)

    def get_joint_names(self):
        return self.active_joint_names.copy()

    def compute_fk(self, joint_pos_dict, link_names):

        chain = pk.build_chain_from_urdf(open(self.urdf_path, "rb").read())
        pk_joint_names_ord = chain.get_joint_parameter_names()

        with torch.no_grad():
            query_q = []
            for jname in pk_joint_names_ord:
                query_q.append(joint_pos_dict[jname])

            query_q = np.array(query_q).T
            ret = chain.forward_kinematics(query_q, )

            res = {}
            for k in ret:
                if k not in link_names:
                    continue
                link_pose = ret[k].get_matrix().cpu().numpy().squeeze(0)
                res[k] = link_pose
                # transform to world frame

        gc.collect()
        return res

    def _sample_configs(self, seed_q, num_samp, delta, q_low, q_high):
        assert len(seed_q.shape) == 1, seed_q.shape

        n = len(seed_q)
        samples = np.random.uniform(0, 1, size=(num_samp, n))

        low_ = np.array(seed_q - delta * 0.5)
        high_ = np.array(seed_q + delta * 0.5)

        low_ = np.maximum(low_, q_low)
        high_ = np.minimum(high_, q_high)

        low_ = low_.reshape((1, n))
        high_ = high_.reshape((1, n))

        samp_rang = high_ - low_
        assert np.all(samp_rang >= 0.), samp_rang

        samp_q = samp_rang * samples + low_

        assert np.all(samp_q >= low_), f"{samp_q} vs {low_} --- {samp_q >= low_}"
        assert np.all(samp_q <= high_), f"{samp_q} vs {high_} --- {samp_q <= high_}"

        return samp_q.astype("float32")

    def compute_partial_ik(self, target_pose_robot, link_name, init_q_dict, fixed_dof_names,
                           pos_tol=0.005, rot_tol=0.01, num_trials=128, max_iter=500,
                           root_link="", do_clipping=False, bias_sampling=True):


        # assume the root link is at origin
        with torch.no_grad():

            chain = pk.build_serial_chain_from_urdf(open(self.urdf_path, "rb").read(),
                                                    link_name,
                                                    root_link_name=root_link)

            active_joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
            q_low, q_high = chain.get_joint_limits()

            # if root_link:
            #     init_q = np.array([init_q_dict[name] for name in active_joint_names])
            #     hand_poses = chain.forward_kinematics(init_q, False)
            #     root_rel_T = hand_poses[root_link].get_matrix().cpu().numpy()[0]
            #     print(root_rel_T)
            #     target_pose_robot = root_rel_T @ np.array(target_pose_robot)

            assert len(q_low) == len(active_joint_names)

            # modify bound
            for fixed_dof_name in fixed_dof_names:
                if fixed_dof_name in active_joint_names:
                    to_ind = active_joint_names.index(fixed_dof_name)
                    val = init_q_dict[fixed_dof_name]
                    q_low[to_ind] = q_high[to_ind] = val

            q_low = np.array(q_low, dtype="float32")
            q_high = np.array(q_high, dtype="float32")

            joint_limits = np.array([q_low, q_high])
            joint_limits = torch.tensor(joint_limits).T

            if bias_sampling:
                seed_q = [init_q_dict[name] for name in active_joint_names]
                seed_q = np.array(seed_q)
                seed_q = np.clip(seed_q, q_low, q_high)
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
                                    optimizer_method="LBFGS",
                                    debug=False,
                                    lr=0.01)



            pos = torch.tensor(target_pose_robot[None, :3, 3]).float()
            rot = torch.tensor(target_pose_robot[None, :3, :3]).float()

            # print("target pose: ", pos, rot)
            # print("link name: ", link_name)

            target_pose_robot_tf = pk.Transform3d(pos=pos, rot=rot, )

            sol = ik.solve(target_pose_robot_tf)

            assert sol.solutions.shape[0] == 1, sol.solutions.shape

            if torch.all(~sol.converged):
                partial_solutions = sol.solutions[0]
                cprint("[Warning]: No Converged Solution!", "red")
                err_pos = sol.err_pos.cpu().numpy()[0]
                best_sol_id = np.argmin(err_pos)
                cprint(f"[Warning]: The Best Err: {err_pos[best_sol_id]}", "red")
                best_sol = partial_solutions[best_sol_id]

            else:
                partial_solutions = sol.solutions[0, sol.converged[0]]

                cprint(f"Found {len(partial_solutions)} solutions!", "cyan")
                partial_solutions = partial_solutions.cpu().numpy()

                err_pos = sol.err_pos[0, sol.converged[0]]
                err_pos = err_pos.cpu().numpy()


                # valid_low = np.all(partial_solutions >= q_low.reshape((1, -1))-1e-4, axis=1)
                # valid_high = np.all(partial_solutions <= q_high.reshape((1, -1))+1e-4, axis=1)
                # valid_ind = valid_low & valid_high
                # print(valid_ind)
                # partial_solutions = partial_solutions[valid_ind]
                # err_pos = err_pos[valid_ind]

                seed_q = np.array([init_q_dict[name] for name in active_joint_names])
                sol_dist = np.linalg.norm(seed_q - partial_solutions, axis=1, keepdims=False)
                best_sol_id = np.argmin(sol_dist)

                assert len(sol_dist) == len(err_pos), f"{sol_dist.shape} vs. {err_pos.shape}"

                best_sol_err = err_pos[best_sol_id]
                best_sol = partial_solutions[best_sol_id]
                cprint(f"Best Solution With Err: {best_sol_err}", "cyan")

            if do_clipping:
                best_sol = np.clip(best_sol, q_low, q_high) # clip will results in drift, need better way to handle this

            best_sol_dict = dict(zip(active_joint_names, best_sol.tolist()))

        gc.collect()
        return best_sol_dict
