"""Validate portable prepared references and run a short CPU physics check."""

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    results = []
    for case in manifest['cases']:
        directory = root / case['processed_dir']
        models = [mujoco.MjModel.from_xml_path(str(directory / file))
                  for file in ('scene.xml', 'scene_eq.xml')]
        model = models[0]
        with np.load(directory / '0/trajectory_kinematic.npz', allow_pickle=False) as reference:
            shapes = {key: list(reference[key].shape) for key in reference.files}
            for key in reference.files:
                array = reference[key]
                if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                    raise ValueError(f'{case["task"]}: non-finite {key}')
            qpos = reference['qpos']
            if qpos.ndim != 2 or qpos.shape[1] != model.nq:
                raise ValueError(f'{case["task"]}: qpos {qpos.shape} does not match nq={model.nq}')
            data = mujoco.MjData(model)
            data.qpos[:] = qpos[0]
            if 'qvel' in reference.files:
                data.qvel[:] = reference['qvel'][0]
            if 'ctrl' in reference.files:
                data.ctrl[:] = reference['ctrl'][0]
            mujoco.mj_forward(model, data)
            for _ in range(5):
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise ValueError(f'{case["task"]}: non-finite CPU rollout')
            frames = int(qpos.shape[0])

        videos = []
        for video in sorted((root / case['visual_dir']).glob('*.mp4')):
            cap = cv2.VideoCapture(str(video))
            count = 0
            try:
                if not cap.isOpened():
                    raise ValueError(f'Cannot open {video}')
                expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    count += 1
                if count == 0 or (expected > 0 and count != expected):
                    raise ValueError(f'{video}: decoded {count} of {expected} frames')
            finally:
                cap.release()
            if case['dataset'] == 'Aria' and count != frames:
                raise ValueError(f'{video}: {count} video frames versus {frames} reference frames')
            if video.name == 'historical_inpainted_video.mp4' and count != frames:
                raise ValueError(f'{video}: historical visual output does not match reference length')
            videos.append({'file': str(video.relative_to(root)), 'decoded_frames': count})
        result = {'task': case['task'], 'dataset': case['dataset'], 'reference_frames': frames,
                  'nq': model.nq, 'nu': model.nu, 'npz_shapes': shapes,
                  'both_xmls_load': True, 'cpu_physics_steps': 5, 'videos': videos,
                  'visual_inputs_complete': case['visual_inputs_complete'],
                  'visual_compositor_verified': (root / case['visual_dir'] / 'visual_regenerated.mp4').exists(),
                  'validation_scope': 'Asset loading, finite reference arrays, video decode, and five CPU steps; no full task success or GPU optimization claim.'}
        results.append(result)
        print(f'{case["dataset"]}: {case["task"]} — {frames} reference frames, both scenes load, {len(videos)} videos decode')
    (root / 'VALIDATION.json').write_text(json.dumps({'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
