import os

import numpy as np
import tqdm
from stable_baselines3 import PPO, SAC
import json
from robosuite.controllers import load_composite_controller_config
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
import torch
import matplotlib.pyplot as plt
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def make_tracking_env(args, use_cam_obs):
    env_name = "CustomTask"

    sides = args.sides
    if len(sides) == 1:
        if sides[0] == "right":
            robot_name = "RBY1RightArmsOnly"
        else:
            raise NotImplementedError
    elif len(sides) == 2:
        robot_name = "RBY1ArmsOnly"
    else:
        raise NotImplementedError

    controller_config = load_composite_controller_config(controller="RBY1_TORSO_AND_ARMS_ABSOLUTE")
    print("controller: ", controller_config, )

    use_early_term = False
    residual_policy_action = True
    cam_name = args.cam_name
    set_obj_state = False
    track_budget = 1

    policy_obs_keys = ["robot0_joint_pos",
                       "robot0_right_gripper_qpos",
                       "robot0_left_gripper_qpos",
                       "target_quat",
                       "target_pos",
                       "tool_quat",
                       "tool_pos",
                       "control_target"]

    retarget_file = args.data_file
    retarget_data = json.load(open(retarget_file, "r"))
    tool_id = retarget_data["info"]["tool_id"] + "_cm"
    target_id = retarget_data["info"]["target_id"] + "_cm"

    tool_pose = np.array(retarget_data["init_state"]["tool"])
    target_pose = np.array(retarget_data["init_state"]["target"])

    obj_info = {"target": {"xml_id": target_id, "pose": target_pose},
                "tool": {"xml_id": tool_id, "pose": tool_pose}}

    from .envs.rl_env_wrapper import RobosuiteFuncRetarWrapper
    def _init():
        return RobosuiteFuncRetarWrapper(env_name, robot_name,
                                         control_cfg=controller_config,
                                         obj_info=obj_info,
                                         obs_keys=policy_obs_keys,
                                         track_budget=track_budget,
                                         retarget_data=retarget_data,
                                         use_camera_obs=use_cam_obs,
                                         use_early_term=use_early_term,
                                         res_action_for_policy=residual_policy_action,
                                         res_scale=1.,
                                         camera_names=cam_name,
                                         set_obj_state=set_obj_state,
                                         debug=args.debug,
                                         sides=sides,
                                         )
    return _init


def get_robot_cfg(ori_cfg, sides):
    mod_cfg = ori_cfg.copy()
    for discard_side in ["left", "right"]:
        if discard_side not in sides:
            for link_name in [f"{discard_side}_arm_{_}" for _ in range(7)]:
                mod_cfg[link_name] = 0.
    return mod_cfg

def generate_play_data(args):
    use_cam_obs = True

    eval_env = make_tracking_env(args, use_cam_obs=use_cam_obs)()
    track_data = eval_env.track_data
    sides = args.sides
    save_dir = f"{args.save_dir}/data-{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    cam_name = args.cam_name
    num_trials = 1000
    max_var = 0.5

    dataset = {}

    for i in tqdm.tqdm(range(num_trials)):
        demo_traj = {"obs": [], "action": []}
        eval_env.reset()

        for t in range(1, len(track_data)):

            eval_env.env._set_robot_config(get_robot_cfg(track_data[t-1]["robot_cfg"], sides=sides))
            eval_env._set_object_poses(t-1)

            # eval_env.env._set_pose([], [])


            cur_robot_cfg = eval_env.env._get_robot_config()
            cur_jpos = eval_env._make_action_from_config(cur_robot_cfg)

            ref_tar_cfg = get_robot_cfg(track_data[t]["robot_cfg"], sides=sides)
            ref_tar_jpos = eval_env._make_action_from_config(ref_tar_cfg)
            residual_action = ref_tar_jpos - cur_jpos

            applied_action = residual_action + np.random.uniform(-max_var, max_var,
                                                                 size=residual_action.shape)

            cur_obs = eval_env.env._get_current_observation()

            next_obs, _, _, _, _ = eval_env.step(applied_action, filter_obs=False)

            # _, axs = plt.subplots(1, 2)
            # axs[0].imshow(cur_obs[f"{cam_name}_image"])
            # axs[1].imshow(next_obs[f"{cam_name}_image"])
            # plt.show()

            reached_robot_cfg = eval_env.env._get_robot_config()
            reached_jpos = eval_env._make_action_from_config(reached_robot_cfg)
            cur_obs["control_target"] = reached_jpos

            print(reached_jpos-cur_jpos, applied_action.min(), applied_action.max(), applied_action.mean())

            filt_obs = eval_env._filter_observation(cur_obs)
            demo_traj["obs"].append(filt_obs)
            demo_traj["action"].append(applied_action)

        dataset[f"demo_{i}"] = demo_traj

    np.save(f"{save_dir}/offline_data.npy", dataset, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    # parser.add_argument("--data_file", type=str,
    #                     default="retarget_2025-08-03-15-03-20.json")
    parser.add_argument("--data_file", type=str,
                        default="/mnt/data2/dexmimic/workspace/egoengine/rl_tracking_dev/data/Retarget/(brush, brush, helmet)/20231027_013/45/retarget_refined_2025-09-04-17-52-48.json")

    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--sides", nargs="+", type=str, default=["right", ])
    parser.add_argument("--cam_name", type=str, default="frontview")
    parser.add_argument("--debug", default=False)

    args = parser.parse_args()

    generate_play_data(args)