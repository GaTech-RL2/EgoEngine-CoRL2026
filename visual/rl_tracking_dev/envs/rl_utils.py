import numpy as np
from robosuite.utils.transform_utils import quat2mat, mat2quat
from termcolor import cprint

def pose2vec7(pose):
    trs = pose[:3, 3]
    quat = mat2quat(pose[:3, :3])
    vec7 = trs.tolist() + quat.tolist()
    return vec7

def soft_thresh(x, shift=0.04, scale=100.0):
    x = np.asarray(x)
    s = scale * (x - shift)
    return float(1.0 / (1.0 + np.exp(s)))


def rotation_difference_degree(R1, R2):

    R_rel = R1.swapaxes(-1, -2) @ R2
    trace = np.trace(R_rel, axis1=-2, axis2=-1)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # For numerical stability
    diff_rad = np.arccos(cos_theta)
    diff_deg = np.rad2deg(diff_rad)
    return diff_deg


def _dense_reward_one_side(side, obs, next_obj_pose, next_robot_cfg, at_the_end):
    assert side in ["left", "right"], side
    if side == "left":
        obj_name = "target"
    else:
        obj_name = "tool"

    cur_obj_pos = obs[f"{obj_name}_pos"] - obs["table_offset"]
    cur_obj_quat = obs[f"{obj_name}_quat"]
    cur_obj_mat = quat2mat(cur_obj_quat)

    arm_jnames = [f"{side}_arm_{i}" for i in range(7)]
    cur_jpos = np.array([obs["robot_config"][jname] for jname in arm_jnames])
    tar_jpos = np.array([next_robot_cfg[jname] for jname in arm_jnames])
    jpos_diff = np.abs(cur_jpos - tar_jpos)
    jpos_avg_div = jpos_diff.mean()

    # r0 = 0.
    # if jpos_diff.max() < 0.3:
    #     r0 = 1.

    r0 = 1./(jpos_avg_div+1)

    r1 = float(obs[f"{side}_contact"])

    r2 = float(at_the_end)
    # if successfully track full trajectory, give 1 reward

    shift_trs = np.linalg.norm(next_obj_pose[:3, 3] - cur_obj_pos,)
    r3 = soft_thresh(shift_trs, 0.05, 100)
    # if object pos matches trajectory, give 1 reward

    shift_orn = rotation_difference_degree(next_obj_pose[:3, :3], cur_obj_mat)
    r4 = soft_thresh(shift_orn, 8, 0.5)
    # if object orn matches trajectory, give 1 reward

    # ref qpos, contact, finish track, obj pos, obj quat
    rewards = [r0, r1, r2, r3, r4]
    weights = [0.1, 0.2, 0.3, 0.3, 0.1]

    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-10)

    cprint(f"reward: {rewards} <-> {weights}", "cyan")

    final_reward = np.array(rewards).dot(weights)

    return final_reward, rewards

def _dense_reward(obs, next_tool_pose, next_target_pose,
                  next_robot_cfg, at_the_end, sides):

    rewards = []
    infos = {}
    for side in sides:
        if side == "left":
            next_pose = next_target_pose
        elif side == "right":
            next_pose = next_tool_pose
        else:
            raise NotImplementedError
        r_side, info = _dense_reward_one_side(side=side, obs=obs,
                                              next_robot_cfg=next_robot_cfg,
                                              next_obj_pose=next_pose,
                                              at_the_end=at_the_end)
        rewards.append(r_side)
        infos[side] = info

    rewards = np.mean(rewards, axis=0, keepdims=False)

    return rewards, infos


def _early_termination_one_side(side, obs, next_robot_cfg, next_obj_pose):
    assert side in ["left", "right"], side
    if side == "left":
        obj_name = "target"
    else:
        obj_name = "tool"

    ROT_DEG_DIFF_TH = 30
    TRS_DIFF_TH = 0.1
    FALLIN_TH = 0.
    ARM_RAD_DIFF_TH = 0.8

    term = 0
    info = {}

    arm_jnames = [f"{side}_arm_{i}" for i in range(7)]
    cur_jpos = np.array([obs["robot_config"][jname] for jname in arm_jnames])
    tar_jpos = np.array([next_robot_cfg[jname] for jname in arm_jnames])
    jpos_diff = np.abs(cur_jpos - tar_jpos)

    if jpos_diff.max() > ARM_RAD_DIFF_TH:
        print("jpos divg!")
        term = 1

    cur_obj_pos = obs[f"{obj_name}_pos"] - obs["table_offset"]
    cur_obj_quat = obs[f"{obj_name}_quat"]
    cur_obj_mat = quat2mat(cur_obj_quat)

    shift_trs = np.linalg.norm(next_obj_pose[:3, 3] - cur_obj_pos)
    shift_orn = rotation_difference_degree(next_obj_pose[:3, :3], cur_obj_mat)

    if cur_obj_pos[2] < FALLIN_TH:
        print("obj fall!")
        term = 1

    if shift_trs > TRS_DIFF_TH:
        print("obj shift!")
        term = 1

    if shift_orn > ROT_DEG_DIFF_TH:
        print("obj orn!")
        term = 1

    info["shift_trs"] = shift_trs
    info["shift_orn"] = shift_orn
    info["obj_pos"] = cur_obj_pos
    info["jpos_diff"] = jpos_diff

    cprint(f"term info: {info}", "yellow")

    return bool(term), info


def _early_termination(obs, next_target_pose, next_tool_pose, next_robot_cfg, sides):
    # this reward is task specific, need to redefine for each task
    terms = []
    term_infos = {}
    for side in sides:
        if side == "left":
            next_pose = next_target_pose
        elif side == "right":
            next_pose = next_tool_pose
        else:
            raise NotImplementedError
        term_side, info = _early_termination_one_side(side=side, obs=obs,
                                                      next_robot_cfg=next_robot_cfg,
                                                      next_obj_pose=next_pose
                                                      )
        terms.append(term_side)
        term_infos[side] = info

    terms = np.logical_or.reduce(terms)

    return bool(terms), term_infos



