import os
import json
import time
import cv2
import argparse
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from robosuite.controllers import load_composite_controller_config
# If you run this inside a package, keep the relative import below inside the function (unchanged behavior).
# from .envs.rl_env_wrapper import RobosuiteFuncRetarWrapper

# ---------- constants ----------
HAND_POSES_ROOT = "/mnt/data2/dexmimic/datasets/egoengine/Aria/data/Hand_Poses"
SEGMENTATION_ROOT = "/mnt/data2/dexmimic/datasets/egoengine/Aria/data/Segmentation"

# ---------- discovery ----------
def find_retarget_jsons(base_dir):
    """Yield absolute paths to all 'retarget.json' under base_dir."""
    for root, _dirs, files in os.walk(base_dir):
        if "retarget.json" in files:
            yield os.path.join(root, "retarget.json")

def triplet_and_serial_from_path(retarget_path):
    """
    Expect: /srv/.../Hand_Poses/(triplet)/serial/retarget.json
    Return (triplet_str, serial_str)
    """
    # .../Hand_Poses/<triplet>/<serial>/retarget.json
    parts = os.path.normpath(retarget_path).split(os.sep)
    try:
        idx = parts.index("Hand_Poses")
    except ValueError:
        raise RuntimeError(f"'Hand_Poses' not found in path: {retarget_path}")
    triplet = parts[idx + 1]
    serial = parts[idx + 2]
    return triplet, serial

def outdir_for_retarget(retarget_path):
    triplet, serial = triplet_and_serial_from_path(retarget_path)
    out_dir = os.path.join(SEGMENTATION_ROOT, triplet, serial)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ---------- env setup ----------
def make_tracking_env(args, use_cam_obs, retarget_path):
    print(retarget_path)
    env_name = "CustomTask"
    robot_name = "RBY1ArmsOnly"
    controller_config = load_composite_controller_config(controller="RBY1_TORSO_AND_ARMS_ABSOLUTE")
    print("controller: ", controller_config)

    use_early_term = False
    cam_name = args.cam_name

    policy_obs_keys = [
        "robot0_joint_pos",
        "robot0_right_gripper_qpos",
        "robot0_left_gripper_qpos",
        "target_quat",
        "target_pos",
        "tool_quat",
        "tool_pos",
    ]

    track_budget = 3
    retarget_data = json.load(open(retarget_path, "r"))
    # tool_id = retarget_data["info"]["tool_id"] + "_cm"
    # target_id = retarget_data["info"]["target_id"] + "_cm"

    desk_pose = np.array(retarget_data["init_state"]["desk"])
    # tool_pose = np.array(retarget_data["init_state"]["tool"])
    # target_pose = np.array(retarget_data["init_state"]["target"])

    # tool2desk = np.linalg.inv(desk_pose) @ tool_pose
    # target2desk = np.linalg.inv(desk_pose) @ target_pose

    obj_info = None

    # keep import local to avoid top-level package import issues in subprocesses
    from .envs.rl_env_wrapper import RobosuiteFuncRetarWrapper

    def _init():
        return RobosuiteFuncRetarWrapper(
            env_name,
            robot_name,
            control_cfg=controller_config,
            obj_info=obj_info,
            obs_keys=policy_obs_keys,
            track_budget=track_budget,
            retarget_data=retarget_data,
            use_camera_obs=use_cam_obs,
            use_early_term=use_early_term,
            camera_names=cam_name,
        )

    return _init, retarget_data

# ---------- one-run worker ----------
def run_single(args, retarget_path):
    """
    Runs a single trajectory and writes:
      - video.mp4
      - mask.mp4
      - mask_bin.mp4
      - metrics.json (mean error, steps)
    into /mnt/data2/dexmimic/datasets/egoengine/TACO/Segmentation/(triplet)/serial_number
    """
    # out_dir = outdir_for_retarget(retarget_path)
    out_dir = "/mnt/data2/dexmimic/datasets/egoengine/Aria/data"
    env_init_fn, retarget_data = make_tracking_env(args, use_cam_obs=True, retarget_path=retarget_path)
    traj = retarget_data["traj"]
    env = env_init_fn()

    warm_up_steps = retarget_data["warm_up_steps"]
    target2cam = [np.array(m) for m in retarget_data["camera_traj"]["target2camera"]]
    K = retarget_data["camera_traj"]["intrinsic"]

    # Reset & get frame size
    _ = env.reset()

    right_body_ids = list(range(13, 20)) + [20, 21] + list(range(22, 36))
    left_body_ids  = list(range(36, 43)) + [43, 44] + list(range(45, 59))
    both_arm_body_ids = right_body_ids + left_body_ids

    m = env.env.sim.model
    arm_geom_ids = [gid for gid in range(m.ngeom) if m.geom_bodyid[gid] in both_arm_body_ids]

    frame0 = env.render()
    H, W = frame0.shape[:2]

    video_path = os.path.join(out_dir, "video.mp4")
    mask_path = os.path.join(out_dir, "mask.mp4")
    mask_bin_path = os.path.join(out_dir, "mask_bin.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 10, (W, H))
    mask_writer = cv2.VideoWriter(mask_path, fourcc, 10, (W, H))
    mask_bin_writer = cv2.VideoWriter(mask_bin_path, fourcc, 10, (W, H))

    actor_map = env._get_actuator_mapping()
    idx2name = {v: k for k, v in actor_map.items()}
    avg_error = []

    for i, dp in enumerate(traj):
        if i < warm_up_steps:
            continue

        time.sleep(0.1)

        robot_cfg = dp["robot_cfg"]
        T_tw = np.array(dp["target"])
        T_tc = target2cam[i]
        T_tw[:3, 3] += np.array([-0.7, 0, 0.0])
        T_cw = T_tw @ np.linalg.inv(T_tc)
        env.env.update_cam(T_cw, K)

        action = np.array([robot_cfg[idx2name[idx]] for idx in range(len(actor_map))])
        obs_array, *_ = env.step(action)
        actual = obs_array[:44]
        err = np.linalg.norm(actual - action)
        avg_error.append(err)

        frame_full = env.render()
        writer.write(cv2.cvtColor(frame_full, cv2.COLOR_RGB2BGR))

        # Background-only & arms-only to build masks
        orig_rgba_step = m.geom_rgba.copy()
        orig_matid_step = m.geom_matid.copy()
        m.geom_matid[:] = -1

        tmp = orig_rgba_step.copy()
        tmp[:, 3] = 0.0
        m.geom_rgba[:] = tmp
        env.env.sim.forward()
        bg_rgb = env.render()

        idx = np.asarray(arm_geom_ids, dtype=np.int32)
        tmp[idx, 3] = 1.0
        m.geom_rgba[:] = tmp
        env.env.sim.forward()
        arms_rgb = env.render()

        # restore
        m.geom_rgba[:] = orig_rgba_step
        m.geom_matid[:] = orig_matid_step
        env.env.sim.forward()

        diff = np.abs(arms_rgb.astype(np.int16) - bg_rgb.astype(np.int16))
        mask_bool = (diff.max(axis=2) > 2)
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        masked_rgb = (frame_full * (mask_u8[..., None] // 255)).astype(np.uint8)
        mask_writer.write(cv2.cvtColor(np.ascontiguousarray(masked_rgb), cv2.COLOR_RGB2BGR))

        mask_bin_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        mask_bin_writer.write(np.ascontiguousarray(mask_bin_rgb))

    # finish
    writer.release()
    mask_writer.release()
    mask_bin_writer.release()
    env.close()

    metrics = {
        "retarget_path": retarget_path,
        "out_dir": out_dir,
        "steps": len(traj) - int(warm_up_steps),
        "mean_error": float(np.mean(avg_error)) if avg_error else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Parallel replay over all TACO retarget.json files.")
    parser.add_argument("--base_dir", type=str, default=HAND_POSES_ROOT,
                        help="Root folder to search for retarget.json")
    parser.add_argument("--data_file", type=str, default=None,
                        help="If set, run ONLY this retarget.json (no discovery)")
    parser.add_argument("--cam_name", type=str, default="cameraview")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="Number of parallel processes")
    # unused but kept to match your signature
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=10)
    parser.add_argument("--ckpt_freq", type=int, default=5)
    parser.add_argument("--rl_algm", type=str, default="sac")
    parser.add_argument("--log_video_freq", type=int, default=5)
    parser.add_argument("--video_len", type=int, default=600)
    args = parser.parse_args()

    # Collect work list
    if args.data_file:
        retarget_paths = args.data_file
    else:
        retarget_paths = list(find_retarget_jsons(args.base_dir))

    print(f"Found {len(retarget_paths)} retarget.json file(s). Writing outputs under {SEGMENTATION_ROOT}")

    # Parallel execution
    # results = []
    # with ProcessPoolExecutor(max_workers=args.workers) as ex:
    #     futures = {ex.submit(run_single, args, p): p for p in retarget_paths}
    #     for fut in as_completed(futures):
    #         p = futures[fut]
    #         try:
    #             res = fut.result()
    #             results.append(res)
    #             print(f"[DONE] {p} -> {res['out_dir']} (mean_err={res['mean_error']})")
    #         except Exception as e:
    #             print(f"[ERROR] {p}: {e}")
    run_single(args, retarget_paths)

if __name__ == "__main__":
    main()
