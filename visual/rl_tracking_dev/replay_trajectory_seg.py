import os

import numpy as np
import json
from robosuite.controllers import load_composite_controller_config
import argparse
import matplotlib.pyplot as plt

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
np.random.seed(42) 

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
    retarget_data = json.load(open(args.data_file, "r"))
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
                                         camera_names=cam_name,
                                         )
    return _init

import time
import cv2
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
def replaying(args):
    # 1) Load data & create env
    retarget_data = json.load(open(args.data_file, "r"))
    traj = retarget_data["traj"]
    env = make_tracking_env(args, use_cam_obs=True)()

    warm_up_steps = retarget_data["warm_up_steps"]

    target2cam = [np.array(m) for m in retarget_data["camera_traj"]["target2camera"]]
    K     = retarget_data["camera_traj"]["intrinsic"]

    # 2) Reset once to get frame size
    obs = env.reset()

    right_body_ids = list(range(13, 20)) + [20, 21] + list(range(22, 36))
    left_body_ids  = list(range(36, 43)) + [43, 44] + list(range(45, 59)) + [59, 60, 61]

    both_arm_body_ids = right_body_ids + left_body_ids

    m = env.env.sim.model
    arm_geom_ids = [
        gid for gid in range(m.ngeom)
        if m.geom_bodyid[gid] in both_arm_body_ids
    ]
    object_body_ids = set()
    for obj_name in ("target", "tool"):
        obj = getattr(env.env, obj_name, None)
        if obj is None:
            continue
        for body_name in getattr(obj, "bodies", []):
            try:
                object_body_ids.add(m.body_name2id(body_name))
            except ValueError:
                continue
    object_geom_ids = np.asarray(
        [gid for gid in range(m.ngeom) if m.geom_bodyid[gid] in object_body_ids],
        dtype=np.int32,
    )


    # env.initialize_cam(cam_pose_seq[0], cam_K_seq[0])
    frame = env.render()  # assume H×W×3 uint8 RGB
    H, W = frame.shape[:2]

    # 3) Setup VideoWriter
    # Always write directly to the episode RGB directory (no occlusion_on/off subfolders).
    video_dir = os.path.dirname(args.data_file).replace("Hand_Poses", "Egocentric_RGB_Videos")
    os.makedirs(video_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(video_dir, "video.mp4"), fourcc, 10, (W, H))

    # 4) Replay (pure reset-style, no controller action stepping)

    mask_writer = None
    mask_bin_writer = None

    replay_indices = range(warm_up_steps, len(traj))
    iterable = tqdm(replay_indices, desc="Replaying", unit="frame") if tqdm else replay_indices
    for i in iterable:
        dp = traj[i]


        # env.update_cam(cam_pose_seq[t], cam_K_seq[t])
        time.sleep(0.1)  # simulate some delay for each step
        # 1) Set robot config directly (reset-style replay, no controller tracking)
        
        robot_cfg = dp["robot_cfg"]
        T_tw = np.array(dp["target"]) # target pose in world frame
        T_tc = target2cam[i] # target→camera
        T_tw[:3, 3] += np.array([-0.7, 0, 0.]) # raise target a bit
        T_cw = T_tw @ np.linalg.inv(T_tc) # camera→world

        env.env.update_cam(T_cw, K)
        
        # [BUGFIX] Ensure objects follow their trajectory
        if hasattr(env.env, '_set_object_poses'):
            tool_pose = np.array(dp["tool"])
            tool_pose[:3, 3] += np.array([-0.7, 0.0, 0.0])
            env.env._set_object_poses(target_pose=T_tw, tool_pose=tool_pose)

        env._set_full_config(robot_cfg)
        
        # 1) Keep the render path identical across modes, like Aria
        frame_full = env.render()
        writer.write(cv2.cvtColor(np.ascontiguousarray(frame_full), cv2.COLOR_RGB2BGR))

        # 2) lazy init writers
        if mask_writer is None:
            fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
            mask_writer = cv2.VideoWriter(
                os.path.join(video_dir, "mask.mp4"),
                fourcc2, 10, (W, H)
            )
        if mask_bin_writer is None:
            fourcc3 = cv2.VideoWriter_fourcc(*'mp4v')
            mask_bin_writer = cv2.VideoWriter(
                os.path.join(video_dir, "mask_bin.mp4"),
                fourcc3, 10, (W, H)
            )

        # 3) render background-only and arms-only for mask diff
        orig_rgba_step = m.geom_rgba.copy()
        orig_matid_step = m.geom_matid.copy()
        m.geom_matid[:] = -1

        tmp = orig_rgba_step.copy()
        idx = np.asarray(arm_geom_ids, dtype=np.int32)

        # Background (arms hidden)
        tmp[idx, 3] = 0.0
        if not args.occlusion and object_geom_ids.size > 0:
            # Difference from Aria: only hide objects, not everything
            tmp[object_geom_ids, 3] = 0.0
            
        m.geom_rgba[:] = tmp
        env.env.sim.forward()
        bg_rgb = env.render()               # background frame

        # Arms visible
        tmp[idx, 3] = 1.0              # arms opaque
        m.geom_rgba[:] = tmp
        env.env.sim.forward()
        arms_rgb = env.render()             # arms-only frame

        # Restore
        m.geom_rgba[:]  = orig_rgba_step
        m.geom_matid[:] = orig_matid_step
        env.env.sim.forward()

        # 4) binary mask = pixels that changed from bg when arms became visible
        diff = np.abs(arms_rgb.astype(np.int16) - bg_rgb.astype(np.int16))
        mask_bool = (diff.max(axis=2) > 2)
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        # 5) Keep colors from the same scene render used for video output.
        masked_rgb = (frame_full * (mask_u8[..., None] // 255)).astype(np.uint8)

        # 7) color-preserving masked video
        mask_writer.write(cv2.cvtColor(np.ascontiguousarray(masked_rgb), cv2.COLOR_RGB2BGR))

        # 8) 0/1 mask video
        mask_bin_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        mask_bin_writer.write(np.ascontiguousarray(mask_bin_rgb))########

        '''

        # 5) Render & write video as before
        frame = env.render()
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        '''
    print("replay mode: reset-style (_set_full_config), controller stepping disabled")

    # 5) Clean up
    writer.release()
    env.close()

    if mask_writer is not None:
        mask_writer.release()
    if mask_bin_writer is not None:
        mask_bin_writer.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--data_file", type=str,
                        default="/mnt/data2/dexmimic/datasets/egoengine/TACO/Hand_Poses/(empty, bowl, plate)/20231027_123/retarget.json")
    parser.add_argument("--num_envs", type=int, default=1,)
    parser.add_argument("--total_timesteps", type=int, default=6e6)
    parser.add_argument("--ckpt_freq", type=int, default=1e5)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--rl_algm", type=str, default="sac")

    parser.add_argument("--log_video_freq", type=int, default=1e5)
    parser.add_argument("--cam_name", type=str, default="cameraview")
    parser.add_argument("--video_len", type=int, default=600)
    parser.add_argument("--occlusion", action="store_true", default=True, help="Consider object occlusion in masks")
    parser.add_argument("--no-occlusion", action="store_false", dest="occlusion", help="Disable object occlusion in masks")

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


###################################################
'''
This is to create the segmentation masks.
'''
###################################################






'''
0 world
1 table
2 left_eef_target
3 right_eef_target
4 robot0_base
5 robot0_wheel_l
6 robot0_link_torso_0
7 robot0_link_torso_1
8 robot0_link_torso_2
9 robot0_link_torso_3
10 robot0_link_torso_4
11 robot0_link_torso_5
12 robot0_link_head_0
13 robot0_link_right_arm_0
14 robot0_link_right_arm_1
15 robot0_link_right_arm_2
16 robot0_link_right_arm_3
17 robot0_link_right_arm_4
18 robot0_link_right_arm_5
19 robot0_link_right_arm_6
20 robot0_right_mount
21 robot0_right_eef
22 gripper0_right_x_right_hand_root
23 gripper0_right_right_hand_thumb_bend_link
24 gripper0_right_right_hand_thumb_rota_link1
25 gripper0_right_right_hand_thumb_rota_link2
26 gripper0_right_right_hand_index_bend_link
27 gripper0_right_right_hand_index_rota_link1
28 gripper0_right_right_hand_index_rota_link2
29 gripper0_right_right_hand_mid_link1
30 gripper0_right_right_hand_mid_link2
31 gripper0_right_right_hand_ring_link1
32 gripper0_right_right_hand_ring_link2
33 gripper0_right_right_hand_pinky_link1
34 gripper0_right_right_hand_pinky_link2
35 gripper0_right_eef
36 robot0_link_left_arm_0
37 robot0_link_left_arm_1
38 robot0_link_left_arm_2
39 robot0_link_left_arm_3
40 robot0_link_left_arm_4
41 robot0_link_left_arm_5
42 robot0_link_left_arm_6
43 robot0_left
44 robot0_left_eef
45 gripper0_left_x_left_hand_root
46 gripper0_left_left_hand_thumb_bend_link
47 gripper0_left_left_hand_thumb_rota_link1
48 gripper0_left_left_hand_thumb_rota_link2
49 gripper0_left_left_hand_index_bend_link
50 gripper0_left_left_hand_index_rota_link1
51 gripper0_left_left_hand_index_rota_link2
52 gripper0_left_left_hand_mid_link1
53 gripper0_left_left_hand_mid_link2
54 gripper0_left_left_hand_ring_link1
55 gripper0_left_left_hand_ring_link2
56 gripper0_left_left_hand_pinky_link1
57 gripper0_left_left_hand_pinky_link2
58 gripper0_left_eef
59 robot0_link_head_1
60 robot0_link_head_2
61 mobilebase0_support
62 robot0_wheel_r
63 target_main
64 tool_main
'''
