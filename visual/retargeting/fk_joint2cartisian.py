from .rby1_kinematics import MinkSolver
import numpy as np
import json

# --- load your recorded trajectory JSON ---
with open("/mnt/data2/dexmimic/datasets/egoengine/Aria/data/Hand_Poses/202/1/retarget.json", "r") as f:
    data = json.load(f)

# Init the solver
cur_dir = ""
kin_model = MinkSolver("retargeting/assets/rbyxhand/model.xml")

# Pick the link names you want FK for
arm_link_names = ["link_right_arm_6"]

# Collect T (4x4) for each frame
all_T = []
for frame in data["traj"]:
    q_dict = frame["robot_cfg"]        # dict of {joint_name: value}
    fk_poses = kin_model.compute_fk(q_dict, arm_link_names)
    # fk_poses is usually {link_name: 4x4 ndarray}
    T_list = [np.array(fk_poses[name]) for name in arm_link_names]
    all_T.append(T_list)

# Now all_T is a list: len = #frames, each element is list of 4x4 matrices
print("FK results for first frame:")
for T in all_T:
    print(T)
