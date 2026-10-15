# Prepared-input contract

The default `examples/prepared/manifest.json` lists six cases. A separate prepared
package can be selected with `egoengine --data-root /path/to/package ...`.
Perception is outside this interface.

Each manifest entry specifies `task`, `dataset` (`Aria` or `TACO`), `dataset_name`,
`robot_type`, `embodiment_type`, `processed_dir`, `visual_dir`, and completeness.
The tested robot is XHand, with right-only or bimanual prepared scenes.

```text
package/
  manifest.json
  assets/                         # mesh dependencies
  processed/<dataset>/xhand/<embodiment>/<task>/
    scene.xml                     # relative asset paths
    scene_eq.xml
    task_info.json                # ref_dt, sim_dt
    0/trajectory_kinematic.npz
  visual_inputs/<dataset>/<task>/
    retarget.json                 # camera trajectory and original RB-Y1 configurations
    egocentric_intrinsic.txt
    cropped_video.mp4             # Aria source frame size; TACO uses color.mp4
    removed_w_mask_5.mp4           # prepared demonstrator-free background
    mask.mp4                      # optional prepared robot foreground for CPU demo
    mask_bin.mp4                  # matching prepared visibility mask
```

`trajectory_kinematic.npz` contains finite `qpos[T,nq]`, `qvel[T,nv]` and
`ctrl[T,nu]`. Controls are MuJoCo actuator targets. Reference timestep comes from
`task_info.json`. Reference/simulator timesteps must have an integer ratio.
Both XML variants and the arrays must agree on dimensions.

`retarget.json` contains the original full RB-Y1 joint commands in `traj[*].robot_cfg`,
`warm_up_steps`, and `camera_traj.target2camera`. Set manifest
`camera_convention` to `camera_to_world` for the Aria producer, or
`target_to_camera` for the TACO producer. TACO normalizes with
`world_from_camera = world_from_target @ inverse(camera_from_target)`. The visual branch uses the same-frame `traj[*].target` from the original
retarget world. Robot, tool, target and camera are replayed together, as in the
retained `replay_trajectory_seg*.py` scripts. The original robosuite replay
translates the scene by -0.7 m in X. The adapter retains that shared offset for
the scene and camera, preserving relative camera/robot/object poses.
For optional `--renderer xhand` debugging, `camera_anchor_joint` selects the
prepared action scene's reference target frame. This debug path is independent
of normal RB-Y1 visual generation. Both formats yield OpenCV camera-to-world
4x4 poses. Intrinsics are `cam_K` or `intrinsic`;
`egocentric_intrinsic.txt` is a fallback. The recorded warmup prefix is removed
only when its length makes the camera and reference frame counts agree.

All compositing inputs must have the reference frame count and represent the same
frame ordering. Source and background must already be rectified to the same calibrated pinhole view. Rendering
may resize it using the scaled intrinsics; HDF5 intrinsics are scaled again to the
stored image resolution. Display-only historical videos with different sampling
are never silently paired with newly optimized actions.

The default visual path calls the retained replay scripts, using the original
robosuite model, gripper and arena XMLs under `visual/robosuite/models/assets`.
Their authored mesh/material/texture values are preserved. The portable wrapper
only stages paths, invokes the existing function and adapts output filenames,
size and display clock. Native replay outputs are retained in
`render/original_pipeline/`. No new full-robot assembly or IK fit is used.
The original `robot_cfg` commands are also saved in `rby1_commands.npz` and
`obs/rby1_joint_commands`; these are commands, rather than newly simulated
whole-robot state observations. EEF controls and simulated states from the action
branch are paired independently by case and reference index.
