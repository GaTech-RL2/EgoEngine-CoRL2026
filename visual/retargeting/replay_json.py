#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import time

from .pb_visualizer import PybulletVisualizer, Entity  # same import as your repo

ROBOT_URDF = "retargeting/assets/rbyxhand/model.urdf"
DESK_URDF  = "retargeting/assets/desk/desk.urdf"

def replay(json_path, dt=0.05):
    # 1. Load saved JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    traj = data["traj"]

    # 2. Build environment, exactly like in retargeting code
    desk_pose = np.eye(4, dtype=np.float32)
    desk_pose[:3, 3] = [0.9, 0.0, 0.724]

    entities = [
        Entity(name="robot",
               urdf=ROBOT_URDF,
               pose=np.eye(4, dtype=np.float32),
               color=None,
               fix_base=True),
        Entity(name="desk",
               urdf=DESK_URDF,
               pose=desk_pose,
               color=[0.5, 0.5, 0.5, 0.5],
               fix_base=True),
    ]
    viz = PybulletVisualizer(entities)
    viz.run(0.5)  # short warmup

    # 3. Step through trajectory, set joints by name
    for frame in traj:
        qdict = {k: float(v) for k, v in frame["robot_cfg"].items()}
        viz.set_joint_configuration(qdict)
        viz.run(dt)

    print(f"[REPLAY] Finished {len(traj)} frames from {json_path}")

if __name__ == "__main__":
    replay("/mnt/data2/dexmimic/datasets/egoengine/Aria/data/Hand_Poses/201/1/retarget_viz_world_start_t0.json")
