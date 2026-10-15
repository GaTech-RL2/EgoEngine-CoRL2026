"""Replay retargeted trajectory in MuJoCo with reset support.

Keys:
- `R`: reset to first frame
- `Space`: pause / resume
- `N`: single-step when paused
- `Esc`: close viewer
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

import spider
from spider.io import get_processed_data_dir


def _infer_ctrl(qpos: np.ndarray, nu: int, embodiment_type: str, nq_obj: int) -> np.ndarray:
    if nu <= 0:
        return np.zeros((qpos.shape[0], 0), dtype=np.float32)
    if embodiment_type in ["bimanual", "right", "left"]:
        return qpos[:, : qpos.shape[1] - nq_obj]
    return qpos[:, :nu]


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "gigahand",
    robot_type: str = "xhand",
    embodiment_type: str = "right",
    task: str = "mustard_901_1",
    data_id: int = 0,
    replay_speed: float = 1.0,
    sim_dt: float = 0.01,
    loop: bool = True,
    hold_last: bool = False,
):
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    scene_path = Path(processed_dir).parent / "scene.xml"
    traj_path = Path(processed_dir) / "trajectory_kinematic.npz"

    if not scene_path.exists():
        raise FileNotFoundError(f"scene.xml not found: {scene_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory_kinematic.npz not found: {traj_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(sim_dt)

    traj = np.load(str(traj_path))
    qpos = np.asarray(traj["qpos"], dtype=np.float32).reshape(-1, model.nq)
    qvel = np.asarray(traj["qvel"], dtype=np.float32).reshape(-1, model.nv)
    if "ctrl" in traj:
        ctrl = np.asarray(traj["ctrl"], dtype=np.float32).reshape(-1, model.nu)
    else:
        nq_obj = 14 if embodiment_type == "bimanual" else 7
        ctrl = _infer_ctrl(qpos, model.nu, embodiment_type, nq_obj)

    T = min(qpos.shape[0], qvel.shape[0], ctrl.shape[0] if model.nu > 0 else qpos.shape[0])
    qpos = qpos[:T]
    qvel = qvel[:T]
    if model.nu > 0:
        ctrl = ctrl[:T]

    paused = False
    single_step = False
    reset_requested = False
    idx = 0

    def _reset_to(i: int = 0):
        nonlocal idx
        idx = max(0, min(int(i), T - 1))
        data.qpos[:] = qpos[idx]
        data.qvel[:] = qvel[idx]
        if model.nu > 0:
            data.ctrl[:] = ctrl[idx]
        mujoco.mj_forward(model, data)

    def _on_key(keycode: int):
        nonlocal paused, single_step, reset_requested
        ch = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if ch == "r":
            reset_requested = True
        elif ch == " ":
            paused = not paused
        elif ch == "n":
            single_step = True

    _reset_to(0)
    rate = RateLimiter(frequency=1.0 / (sim_dt / max(replay_speed, 1e-6)))

    with mujoco.viewer.launch_passive(model, data, key_callback=_on_key) as viewer:
        while viewer.is_running():
            if reset_requested:
                _reset_to(0)
                reset_requested = False

            do_step = (not paused) or single_step
            if do_step:
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                if model.nu > 0:
                    data.ctrl[:] = ctrl[idx]
                mujoco.mj_forward(model, data)
                viewer.sync()
                single_step = False
                idx += 1
                if idx >= T:
                    if loop:
                        _reset_to(0)
                    elif hold_last:
                        idx = T - 1
                        paused = True
                    else:
                        break
            else:
                viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    tyro.cli(main)

