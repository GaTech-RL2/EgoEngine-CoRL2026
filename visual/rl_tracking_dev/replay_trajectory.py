import os

import numpy as np
import json
from robosuite.controllers import load_composite_controller_config
import argparse
import matplotlib.pyplot as plt

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_tracking_env(args, use_cam_obs):
    env_name = "CustomTask"
    robot_name = "RBY1ArmsOnly"
    controller_config = load_composite_controller_config(controller="RBY1_TORSO_AND_ARMS_ABSOLUTE")

    print("controller: ", controller_config,)

    use_early_term = False
    cam_name = args.cam_name

    #### for debug ####
    # use_cam_obs = True
    # cam_name = "frontview"
    # # cam_name = "birdview"
    # # cam_name = "sideview"
    # # cam_name = "agentview"
    # use_early_term = False
    #### for debug ####

    policy_obs_keys = ["robot0_joint_pos",
                       "robot0_right_gripper_qpos",
                       "robot0_left_gripper_qpos",
                       "target_quat",
                       "target_pos",
                       "tool_quat",
                       "tool_pos"]

    track_budget = 3
    retarget_file = args.data_file
    retarget_data = json.load(open("/mnt/data2/dexmimic/datasets/egoengine/TACO/Hand_Poses/(skim off, bowl, plate)/20230927_016/retarget.json", "r"))
    tool_id = retarget_data["info"]["tool_id"] + "_cm"
    target_id = retarget_data["info"]["target_id"] + "_cm"

    desk_pose = np.array(retarget_data["init_state"]["desk"])

    tool_pose = np.array(retarget_data["init_state"]["tool"])
    target_pose = np.array(retarget_data["init_state"]["target"])

    tool2desk = np.linalg.inv(desk_pose) @ tool_pose
    target2desk = np.linalg.inv(desk_pose) @ target_pose

    obj_info = {"target": {"xml_id": target_id, "pose": target2desk},
                "tool": {"xml_id": tool_id, "pose": tool2desk}}

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
                                         camera_names=cam_name)
    return _init

import time
import cv2
def replaying(args):
    # 1) Load data & create env
    retarget_data = json.load(open(
        "/mnt/data2/dexmimic/datasets/egoengine/TACO/Hand_Poses/(brush, eraser, bowl)/20231027_037/retarget.json", "r"))
    traj = retarget_data["traj"]
    env = make_tracking_env(args, use_cam_obs=True)()

    warm_up_steps = retarget_data["warm_up_steps"]

    target2cam = [np.array(m) for m in retarget_data["camera_traj"]["target2camera"]]
    K     = retarget_data["camera_traj"]["intrinsic"]

    # 2) Reset once to get frame size
    obs = env.reset()
    # env.initialize_cam(cam_pose_seq[0], cam_K_seq[0])
    frame = env.render()  # assume H×W×3 uint8 RGB
    H, W = frame.shape[:2]

    # 3) Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('/mnt/data2/dexmimic/workspace/egoengine/rl_tracking_dev/video.mp4', fourcc, 10, (W, H))

    # 4) Replay
    actor_map = env._get_actuator_mapping()
    idx2name = {v: k for k, v in actor_map.items()}
    avg_error = []

    for i, dp in enumerate(traj):
        # env.update_cam(cam_pose_seq[t], cam_K_seq[t])
        time.sleep(0.1)  # simulate some delay for each step
        # 1) Build your action the same as before
        
        robot_cfg = dp["robot_cfg"]
        T_tw = np.array(dp["target"]) # target pose in world frame
        T_tc = target2cam[i] # target→camera
        T_tw[:3, 3] += np.array([-0.7, 0, 0.]) # raise target a bit
        T_cw = T_tw @ np.linalg.inv(T_tc) # camera→world

        env.env.update_cam(T_cw, K)
        
        action = np.array([robot_cfg[idx2name[idx]] 
                            for idx in range(len(actor_map))])

        # 1) Unpack step outputs
        obs_array, *_ = env.step(action)

        # 2) The first len(action) entries of obs_array are the executed joint values
        actual = obs_array[:44]

        # 3) Compute error against your command
        err = np.linalg.norm(actual - action)

        print(f"step={i:3d}  action error: {err:.6f}")
        
        '''            
        obs_dict = env._set_full_config(robot_cfg)
        # obs_dict has keys like 'robot_config', etc.

        # 3) Pull out the actual executed joints from robot_config
        actual_cfg_dict = obs_dict["robot_config"]
        actual = np.array([
            actual_cfg_dict[idx2name[idx]]
            for idx in range(len(actor_map))
        ])
        # actual_cfg is a dict: { joint_name: qpos, … }

        # 3) Turn it into the same vector order


        # 4) Compute & print the error (should be zero)
        err = np.linalg.norm(actual - action)
        print(f"step={i:3d}  command↔actual error: {err:.6f}")
        '''
        
        avg_error.append(err)
        
        

        # 5) Render & write video as before
        frame = env.render()
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    print("mean error: ", np.mean(avg_error))

    # 5) Clean up
    writer.release()
    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--data_file", type=str,
                        default="/mnt/data2/dexmimic/datasets/egoengine/TACO/Hand_Poses/(brush, eraser, bowl)/20231027_037/retarget.json")
    parser.add_argument("--num_envs", type=int, default=1,)
    parser.add_argument("--total_timesteps", type=int, default=6e6)
    parser.add_argument("--ckpt_freq", type=int, default=1e5)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--rl_algm", type=str, default="sac")

    parser.add_argument("--log_video_freq", type=int, default=1e5)
    parser.add_argument("--cam_name", type=str, default="cameraview")
    parser.add_argument("--video_len", type=int, default=600)

    args = parser.parse_args()

    # debug #
    args.total_timesteps = 10
    args.log_video_freq = 5
    args.ckpt_freq = 5
    # debug #

    save_dir = f"{args.save_dir}/{args.rl_algm}-{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/config.txt", "w+") as f:
        f.writelines(str(args))

    replaying(args,)
