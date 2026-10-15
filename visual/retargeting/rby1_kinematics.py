import os.path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from collections import namedtuple

Target = namedtuple(
    'IKTarget',
    ['task_name', 'relative', 'root_name', "link_name", "link_pose", "orientation_weight", "position_weight"],
)
Target.__new__.__defaults__ = (1.0, 1.0)

class MinkSolver:
    def __init__(self, mjcf_path):
        model = mujoco.MjModel.from_xml_path(mjcf_path, )
        self.model = model
        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(model.njnt)]
        self.joint_names = joint_names

    def compute_fk(self, q_dict, link_names):
        q = np.array([q_dict[k] for k in self.joint_names], dtype="float32")
        model = self.model
        configuration = mink.Configuration(model, q)
        link_poses = {}
        for k in link_names:
            pose = configuration.get_transform_frame_to_world(frame_name=k,
                                                              frame_type="body")
            pose = pose.as_matrix()
            link_poses[k] = pose
        return link_poses

    def compute_ik(self, q_dict, ik_targets: [Target], num_iter=20, freq=200.0,
                   fixed_links=["base_link",], solver="quadprog",
                   pos_threshold=1e-4, ori_threshold=1e-4, return_err=False):

        q = np.array([q_dict[k] for k in self.joint_names], dtype="float32")
        model = self.model
        configuration = mink.Configuration(model, q)
        init_cfg = mink.Configuration(model, q)

        rate = RateLimiter(frequency=freq, warn=False)
        # base_task = mink.FrameTask(
        #     frame_name="base_link",
        #     frame_type="body",
        #     position_cost=2.0,
        #     orientation_cost=2.0,
        #     lm_damping=0.5,
        # )
        fixed_tasks = []
        for link_name in fixed_links:
            fix_task = mink.FrameTask(
                frame_name=link_name,
                frame_type="body",
                position_cost=1000,
                orientation_cost=1000,
                lm_damping=0.1,
            )
            fixed_tasks.append(fix_task)

        # # want to fix the base link, minimize movement of base_link


        # want the solution closer to some reference config, e.g., prev config

        damping_task = mink.DampingTask(model, cost=0.01)
        # com_task = mink.ComTask(cost=0.05)

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_arm_5")
        qpos_idx = model.jnt_qposadr[jid]         # index in qpos (nq)
        dof_idx  = model.jnt_dofadr[jid]

        posture_cost = np.zeros(configuration.nv, dtype=float)
        posture_cost[dof_idx] = 0.7

        posture_task = mink.PostureTask(model, cost=posture_cost)

        eef_tasks = {}
        ik_target: Target
        for ik_target in ik_targets:
            # Defaults for most targets
            pos_cost = float(ik_target.position_weight)
            ori_cost = 0.0

            # Orientation-only wrist task
            if ik_target.task_name.endswith("_wrist_ori") or ik_target.task_name.startswith("ori_"):
                pos_cost = 0.0         # no position penalty
                ori_cost = float(ik_target.orientation_weight)  # weight orientation

            if ik_target.relative:
                eef_task = mink.RelativeFrameTask(
                    frame_name=ik_target.link_name,
                    root_name=ik_target.root_name,
                    frame_type="body",
                    root_type="body",
                    position_cost=pos_cost,
                    orientation_cost=ori_cost,
                )
            else:
                eef_task = mink.FrameTask(
                    frame_name=ik_target.link_name,
                    frame_type="body",
                    position_cost=pos_cost,
                    orientation_cost=ori_cost,
                )

            eef_tasks[ik_target.task_name] = eef_task

        limits = [
            mink.ConfigurationLimit(model),
        ]

        converge = lambda err: (bool(np.linalg.norm(err[:3]) <= pos_threshold) &
                                 bool(np.linalg.norm(err[3:]) <= ori_threshold))

        errs = None

        t = 0.
        for iter_i in range(num_iter):
            for task in fixed_tasks:
                task.set_target_from_configuration(init_cfg)

            # base_task.set_target_from_configuration(configuration)

            posture_task.set_target_from_configuration(configuration)
            damping_task.set_target_from_configuration(configuration)
            for ik_target in ik_targets:
                target_pose = mink.SE3.from_matrix(np.array(ik_target.link_pose))
                eef_tasks[ik_target.task_name].set_target(target_pose)

            tasks = [posture_task, damping_task] + list(eef_tasks.values()) + fixed_tasks

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)

            t += rate.dt

            errs = {k: eef_task.compute_error(configuration) for (k, eef_task) in eef_tasks.items()}
            early_stop = True
            for err in errs.values():
                if not converge(err):
                    early_stop = False
                    break
            if early_stop:
                break

        q_res = dict(zip(self.joint_names, configuration.q.tolist()))

        if not return_err:
            return q_res
        else:
            return {"q": q_res, "errors": errs}

if __name__ == '__main__':
    wrist_traj = [[[[-0.30325127606558006, 0.2121411618588196, -0.9290176294588717, 0.5995154214458194],
                    [-0.9096345142466093, 0.21828488290247455, 0.35349077105800336, 0.24545583791017223],
                    [0.2777804473393791, 0.9522630275001024, 0.1267757534331575, 0.8056163617535307],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[-0.19147291947676157, -0.20481856415177355, -0.9599096008069954, 0.629386891779026],
                    [0.9689699924089958, 0.12485788390433432, -0.21341655857368755, -0.1641976466710181],
                    [0.1635639545894913, -0.9709870901420635, 0.17455613900051192, 0.8372979672307141],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.4828500168408918, -0.06972027126565038, -0.872945536586899, 0.6816287841938291],
                    [-0.8568044878512401, 0.2272052717067685, 0.4629285030565967, 0.24340787274413567],
                    [0.1660623270157298, 0.9714686888943703, -0.16944271058456753, 0.893690203320773],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[0.3430604614066034, 0.1443584246906003, -0.9281751612039242, 0.7226303680693559],
                    [0.867180442196241, 0.3245909787740847, 0.377727073435844, -0.24489380527268545],
                    [0.3558053693331646, -0.9344785708270527, -0.013830471611289013, 0.8854519958027256],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.4225410273971911, -0.23416790063227377, -0.8755931928214117, 0.6806941843916706],
                    [-0.7480972323438364, 0.6331349305613685, 0.19882066164797163, 0.2653027624868421],
                    [0.5078112183965661, 0.7390387308493241, -0.4427058424072202, 0.8824966245872325],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[0.11863318304661724, -0.007377970417868957, -0.9929303592238152, 0.6211026347277365],
                    [0.4388509772374425, 0.897072445681485, 0.05205585281574761, -0.12190553492645055],
                    [0.8903463991981517, -0.4419240099898344, 0.10966038918251897, 0.8911773145991604],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.3606593756650454, -0.26321265682024253, -0.8948088493472353, 0.6709102566432239],
                    [-0.639584342573996, 0.7677600784686722, 0.038926796420485056, 0.2846951181179069],
                    [0.6767524868819588, 0.5863450437327975, -0.44524656463058715, 0.8821674339154281],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[0.0619814785623055, -0.14054444867334526, -0.98815207286021, 0.6758031741841524],
                    [0.4928065612429673, 0.8659082487171569, -0.08592766047706574, -0.0808512738400634],
                    [0.8677256865441523, -0.4816419015655818, 0.12293150000728591, 0.9057213349195118],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.35637040524197144, -0.2549481062225584, -0.8989107643347017, 0.6713417934646974],
                    [-0.6100831766310484, 0.7919926974476634, 0.024187798927071524, 0.291843422854558],
                    [0.7057641274800325, 0.5570301503187102, -0.4377823090584343, 0.8818157617797836],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[0.08277611169217466, -0.14072015240376665, -0.9866027158264765, 0.6959331030005523],
                    [0.5039811842372344, 0.8605503733475891, -0.0741281707703194, -0.07603214733760012],
                    [0.8594526629384314, -0.49109316335067854, 0.14215332287545707, 0.9066128142254363],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.35071959754912413, -0.32890103688718636, -0.8768459594849644, 0.678787242568047],
                    [-0.7833902142638582, 0.6150463503387972, 0.08975925560155508, 0.25858506955600874],
                    [0.5097789949529746, 0.718392874078211, -0.4733666622235386, 0.8987313338983356],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[0.10904851795900526, -0.0464601421544024, -0.9929697078750847, 0.6476734943205935],
                    [0.5637695980573633, 0.8254258138184102, 0.02958092331296718, -0.16003239036247718],
                    [0.8182484953176419, -0.5630318889390119, 0.11620427735007884, 0.8898234903225681],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.4176629232685776, -0.3632424387771769, -0.8328574776236021, 0.6717792175661023],
                    [-0.8714005331955267, 0.4134082777822438, 0.2641849190307134, 0.24766913268660284],
                    [0.24834700118556963, 0.8360926956429284, -0.48919496463163553, 0.881775493798911],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[0.2678166236377442, 0.1770627985740818, -0.9470807707371037, 0.7186968530579338],
                    [0.8969545842792196, 0.3075090689124405, 0.31772568532098544, -0.23462091611106295],
                    [0.3474933250160451, -0.9345806592809963, -0.07646124831221449, 0.863808952176041],
                    [0.0, 0.0, 0.0, 1.0]]],
                  [[[-0.35732764167757236, 0.16493571802503737, -0.9193215597647559, 0.6111366174709922],
                    [-0.8148818556352442, 0.42051140893380007, 0.3989695240849424, 0.2535663765489884],
                    [0.4523895292849536, 0.891701297689087, -0.01585722608805421, 0.8102444910855587],
                    [0.0, 0.0, 0.0, 1.0]],
                   [[-0.1866669545442227, -0.2176261327756789, -0.9580361575900368, 0.6202676733903831],
                    [0.9751995082720002, 0.08753489054793683, -0.20337779337578574, -0.1484545445323603],
                    [0.12812191286043317, -0.9722403030999821, 0.19588902623503388, 0.8368213473456645],
                    [0.0, 0.0, 0.0, 1.0]]]]

    track_one_traj(model_path="assets/rbyxhand_v2/rby1_act.xml", traj=wrist_traj)
