"""Path/output adapter for the retained visual replay; no model or material synthesis."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .data import PROJECT_ROOT
from .video import retime, video_info


def source_files(dataset):
    robot = 'robot_patched_torso_x_aria.xml' if dataset == 'Aria' else 'robot_patched_torso_x.xml'
    hand = 'x_right_hand_aria.xml' if dataset == 'Aria' else 'x_right_hand.xml'
    replay = 'replay_trajectory_seg_aria.py' if dataset == 'Aria' else 'replay_trajectory_seg.py'
    return [
        PROJECT_ROOT / 'visual/rl_tracking_dev' / replay,
        PROJECT_ROOT / 'visual/rl_tracking_dev/envs/rl_env_wrapper.py',
        PROJECT_ROOT / 'visual/rl_tracking_dev/envs/custom_tasks.py',
        PROJECT_ROOT / 'visual/robosuite/models/robots/manipulators/rby1_robot.py',
        PROJECT_ROOT / 'visual/robosuite/models/assets/robots/rby1' / robot,
        PROJECT_ROOT / 'visual/robosuite/models/assets/grippers' / hand,
        PROJECT_ROOT / 'visual/robosuite/models/assets/grippers/x_left_hand.xml',
        PROJECT_ROOT / 'visual/robosuite/models/assets/arenas/table_arena.xml',
    ]


def render_original(case, output, width=None, height=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged = output / 'original_pipeline'
    hand_dir = staged / 'Hand_Poses' / case.name / '1'
    rgb_dir = staged / 'Egocentric_RGB_Videos' / case.name / '1'
    hand_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    # Only copies live in the replay's writable dataset tree.
    shutil.copy2(case.visual / 'retarget.json', hand_dir / 'retarget.json')
    rgb_name = 'cropped_video.mp4' if case.info['dataset'] == 'Aria' else 'color.mp4'
    rgb_link = rgb_dir / rgb_name
    if not rgb_link.exists():
        rgb_link.symlink_to(case.visual / rgb_name)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(PROJECT_ROOT / 'visual'), str(PROJECT_ROOT), env.get('PYTHONPATH', '')]
    )
    env.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache/matplotlib'))
    env.setdefault('NUMBA_CACHE_DIR', str(PROJECT_ROOT / '.cache/numba'))
    command = [sys.executable, '-m', 'egoengine.legacy_visual',
               '--data-file', str(hand_dir / 'retarget.json'),
               '--dataset', case.info['dataset']]
    log = output / 'original_replay.log'
    print(f'[render] {case.name}: original {case.info["dataset"]} replay; log: {log}', flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, cwd=PROJECT_ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log.read_text(errors='replace')[-5000:]
        raise RuntimeError(
            f'Original visual replay failed (exit {result.returncode}). See {log}\n{tail}'
        )
    source_size = video_info(rgb_dir / 'video.mp4')
    width = width or source_size['width']
    if height is None:
        height = max(2, int(round(width * source_size['height'] / source_size['width'] / 2)) * 2)
    count = len(case.reference()['qpos'])
    paths = {}
    for key, original, name in [('scene', 'video.mp4', 'scene.mp4'),
                                ('foreground', 'mask.mp4', 'foreground.mp4'),
                                ('mask', 'mask_bin.mp4', 'visibility_mask.mp4')]:
        source = rgb_dir / original
        if video_info(source, decode=True)['frames'] != count:
            raise ValueError(f'Original replay frame count mismatch: {source}')
        # Retain untouched native videos; public copies only change display size/clock.
        retime(source, output / name, 1 / case.ref_dt, count, size=(width, height))
        paths[key] = output / name
    frames = case.retarget_frames()
    joint_names = sorted(frames[0]['robot_cfg'])
    np.savez(output / 'rby1_commands.npz', joint_names=np.asarray(joint_names),
             joint_commands=np.asarray([[f['robot_cfg'][n] for n in joint_names] for f in frames]))
    # Remove incompatible cached render artifacts before recording this replay.
    (output / 'rby1_states.npz').unlink(missing_ok=True)
    (output / 'ik.json').unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(paths['mask']))
    pixels = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pixels.append(int((frame.max(axis=2) > 127).sum()))
    cap.release()
    metadata = dict(
        case=case.name, renderer='rby1', backend='original_robosuite_replay',
        frames=count, fps=1 / case.ref_dt, action_source='original_retarget',
        retarget_sha256=hashlib.sha256((case.visual / 'retarget.json').read_bytes()).hexdigest(),
        action_sha256=None, render_collision_mesh=False, visible_geom_groups=[1],
        frames_with_visible_robot=sum(p > 0 for p in pixels),
        mask_pixels_min=min(pixels), mask_pixels_max=max(pixels),
        original_options=dict(cam_name='cameraview', occlusion=True,
                              **(dict(reset_replay=True, flip=False, visualize=False)
                                 if case.info['dataset'] == 'Aria' else {})),
        native_videos=str(rgb_dir.relative_to(output)), native_video_info=source_size,
        source_sha256={str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_files(case.info['dataset'])},
        world_frame='original robosuite replay world (saved camera/object poses shifted by -0.7m X)',
        render_paths={key: path.name for key, path in paths.items()},
    )
    (output / 'render.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return paths


def main():
    """Call the existing replay function with its existing CLI options."""
    import argparse
    import importlib
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--dataset', choices=['Aria', 'TACO'], required=True)
    args = parser.parse_args()
    name = 'replay_trajectory_seg_aria' if args.dataset == 'Aria' else 'replay_trajectory_seg'
    original = importlib.import_module('rl_tracking_dev.' + name)
    original.replaying(SimpleNamespace(data_file=args.data_file, cam_name='cameraview',
                                      occlusion=True, reset_replay=True,
                                      flip=False, visualize=False))


if __name__ == '__main__':
    main()
