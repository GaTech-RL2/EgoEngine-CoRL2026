#!/usr/bin/env bash
set -euo pipefail

# ====== USER CONFIG ======
DATASET_DIR="/home/ycl/projects/workspace/spider/example_datasets"
DATASET_NAME="gigahand"
ROBOT_TYPE="xhand"
EMBODIMENT_TYPE="right"   # right / left / bimanual

# Your Aria sample
ARIA_HANDPOSE_DIR="/home/ycl/projects/datasets/egoengine/Aria/mustard/data/Hand_Poses/901/1"
RETARGET_JSON="${ARIA_HANDPOSE_DIR}/retarget.json"
ENV_XML="${ARIA_HANDPOSE_DIR}/env.xml"
ALL_POSE_JSON="${ARIA_HANDPOSE_DIR}/all_pose.json"
W2CAM_NPY="/home/ycl/projects/datasets/egoengine/Aria/mustard/data/Egocentric_Camera_Parameters/901/1/upd_egocentric_frame_extrinsic.npy"

# Output id in SPIDER processed tree
TASK="mustard_901_1"
DATA_ID=0
OBJECT_NAME="mustard_901_1"
BASE_OFFSET_XYZ="-0.7 0.0 0.0"
ARENA_DIFF_XYZ="0.0 0.0 0.0"
HAND_Z_ROT_DEG="0.0"
UP_SAMPLE="1"   # trajectory temporal upsample factor (integer >= 1)
START_STEP="0" # "64" # only for MPC input trajectory, start from timestep K

# Optional flags
RUN_CONTACT_DETECT=0   # 1 to enable
SHOW_VIEWER=1

# MPC tuning: CTRL_DT must match reference trajectory rate (20 Hz => 0.05s)
MPC_SIM_DT="0.01"
MPC_CTRL_DT="0.05"
MPC_HORIZON="1.6" # 1.6
MPC_KNOT_DT="0.4"
MPC_NUM_SAMPLES="1024" # 1024
MPC_MAX_ITERS="32" # 32
MPC_IMPROVE_THRESH="0.01" # 0.01
MPC_TEMPERATURE="0.1"

# Reward tuning: default.yaml defaults
OBJ_POS_REW_SCALE="1.0" # 1.0
OBJ_ROT_REW_SCALE="0.3" # 0.3
JOINT_REW_SCALE="0.003" # 0.003
BASE_POS_REW_SCALE="1.0" # 1.0
BASE_ROT_REW_SCALE="0.3" # 0.3
VEL_REW_SCALE="0.000" # 0.0001
CONTACT_REW_SCALE="0.0" # 0.0

# ====== 1) Bridge Aria retarget -> SPIDER processed/mano ======
# uv run spider/process_datasets/aria_retarget_to_spider.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}" \
#   --embodiment-type "${EMBODIMENT_TYPE}" \
#   --retarget-json "${RETARGET_JSON}" \
#   --env-xml "${ENV_XML}" \
#   --all-pose-json "${ALL_POSE_JSON}" \
#  --w2cam-npy "${W2CAM_NPY}" \
#   --object-name "${OBJECT_NAME}" \
#   --ref-dt 0.05 \
#   --base-offset-xyz ${BASE_OFFSET_XYZ} \
#   --arena-diff-xyz ${ARENA_DIFF_XYZ} \
#   --hand-z-rot-deg ${HAND_Z_ROT_DEG}

# ====== 2) Object convex decomposition ======
# uv run spider/preprocess/decompose_fast.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --robot-type "${ROBOT_TYPE}" \
#  --embodiment-type "${EMBODIMENT_TYPE}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}"

# ====== 3) Contact detection (forced path for this Aria bridge) ======
if [[ "${RUN_CONTACT_DETECT}" == "1" ]]; then
  MANO_KIN_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_kinematic.npz"
  MANO_KEY_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_keypoints.npz"
  if [[ ! -f "${MANO_KIN_PATH}" && -f "${MANO_KEY_PATH}" ]]; then
    cp -f "${MANO_KEY_PATH}" "${MANO_KIN_PATH}"
    echo "[INFO] created MANO trajectory_kinematic from trajectory_keypoints for contact detection"
  fi
  if [[ ! -f "${MANO_KIN_PATH}" ]]; then
    echo "[ERROR] contact mode enabled, but MANO trajectory_kinematic is missing: ${MANO_KIN_PATH}"
    exit 1
  fi
  uv run spider/preprocess/detect_contact.py \
    --dataset-dir "${DATASET_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --embodiment-type "${EMBODIMENT_TYPE}" \
    --task "${TASK}" \
    --data-id "${DATA_ID}" \
    --show-viewer

  # hard check: contact labels must exist after detect_contact
  uv run python - "${MANO_KEY_PATH}" <<'PY'
import numpy as np, sys
path = sys.argv[1]
data = np.load(path)
required = ("contact_left", "contact_right", "contact_pos_left", "contact_pos_right")
missing = [k for k in required if k not in data]
if missing:
    print(f"[ERROR] contact fields missing in {path}: {missing}")
    sys.exit(1)
print("[OK] contact fields found in trajectory_keypoints.npz")
PY
fi

# ====== 4) Generate scene.xml for xhand ======
# uv run spider/preprocess/generate_xml.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --robot-type "${ROBOT_TYPE}" \
#   --embodiment-type "${EMBODIMENT_TYPE}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}" \
#   --no-show-viewer

# ====== 5) Kinematic IK retargeting ======
uv run spider/preprocess/ik.py \
  --dataset-dir "${DATASET_DIR}" \
 --dataset-name "${DATASET_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --embodiment-type "${EMBODIMENT_TYPE}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}" \
  --no-show-viewer

# ====== 5.5) Optional START_STEP + UP_SAMPLE for MPC input trajectory only ======
if [[ "${UP_SAMPLE}" != "1" || "${START_STEP}" != "0" ]]; then
  ROBOT_KIN_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_kinematic.npz"
  uv run python - "${ROBOT_KIN_PATH}" "${UP_SAMPLE}" "${START_STEP}" <<'PY'
import sys
import numpy as np

kin_path = sys.argv[1]
up = int(sys.argv[2])
start_step = int(sys.argv[3])
if up < 1:
    raise SystemExit("UP_SAMPLE must be >= 1")
if start_step < 0:
    raise SystemExit("START_STEP must be >= 0")

with np.load(kin_path) as d:
    keys = list(d.files)
    src = {k: d[k].copy() for k in keys}  # load to memory first (avoid CRC issue)

if not keys:
    raise SystemExit(f"Empty npz: {kin_path}")

T0 = src[keys[0]].shape[0]
if start_step >= T0:
    raise SystemExit(f"START_STEP={start_step} out of range for length={T0}")

def upsample_time(arr: np.ndarray, factor: int) -> np.ndarray:
    n = arr.shape[0]
    if n <= 1:
        return arr.copy()
    n_up = (n - 1) * factor + 1
    t_old = np.arange(n, dtype=np.float64)
    t_new = np.linspace(0.0, n - 1, n_up, dtype=np.float64)
    flat = arr.reshape(n, -1)
    out = np.empty((n_up, flat.shape[1]), dtype=flat.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_new, t_old, flat[:, j])
    return out.reshape((n_up,) + arr.shape[1:])

out = {}
for k, v in src.items():
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
        v2 = v[start_step:]
        if up > 1 and v2.shape[0] > 1:
            out[k] = upsample_time(v2, up)
        else:
            out[k] = v2
    else:
        out[k] = v

np.savez_compressed(kin_path, **out)
print(f"[OK] prepared trajectory_kinematic for MPC: {kin_path}")
print(f"[INFO] START_STEP={start_step}, UP_SAMPLE={up}")
print(f"[INFO] frames: {src[keys[0]].shape[0]} -> {out[keys[0]].shape[0]}")
PY
fi


# python -m spider.preprocess.replay \
# --dataset-dir /home/ycl/projects/workspace/spider/example_datasets \
# --dataset-name gigahand \
# --robot-type xhand \
# --embodiment-type right \
# --task mustard_901_1 \
# --data-id 0 \
# --sim-dt 0.01 \
# --replay-speed 0.2


# ====== 6) Dynamics optimization (MJWP) ======
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

uv run examples/run_mjwp.py \
  +override=${DATASET_NAME} \
  task=${TASK} \
  data_id=${DATA_ID} \
  robot_type=${ROBOT_TYPE} \
  embodiment_type=${EMBODIMENT_TYPE} \
  show_viewer=${SHOW_VIEWER} \
  viewer=mujoco-rerun \
  sim_dt=${MPC_SIM_DT} \
  ctrl_dt=${MPC_CTRL_DT} \
  horizon=${MPC_HORIZON} \
  knot_dt=${MPC_KNOT_DT} \
  num_samples=${MPC_NUM_SAMPLES} \
  max_num_iterations=${MPC_MAX_ITERS} \
  improvement_threshold=${MPC_IMPROVE_THRESH} \
  temperature=${MPC_TEMPERATURE} \
  first_ctrl_noise_scale=0.5 \
  last_ctrl_noise_scale=1.0 \
  final_noise_scale=0.1 \
  exploit_ratio=0.01 \
  exploit_noise_scale=0.01 \
  joint_noise_scale=0.15 \
  pos_noise_scale=0.03 \
  rot_noise_scale=0.03 \
  pos_rew_scale=${OBJ_POS_REW_SCALE} \
  rot_rew_scale=${OBJ_ROT_REW_SCALE} \
  joint_rew_scale=${JOINT_REW_SCALE} \
  base_pos_rew_scale=${BASE_POS_REW_SCALE} \
  base_rot_rew_scale=${BASE_ROT_REW_SCALE} \
  vel_rew_scale=${VEL_REW_SCALE} \
  +contact_rew_scale=${CONTACT_REW_SCALE}
