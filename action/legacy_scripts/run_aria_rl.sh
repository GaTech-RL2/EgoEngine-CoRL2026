#!/usr/bin/env bash
set -u
set -o pipefail

ARIA_ROOT_BASE="/home/ycl/projects/datasets"
CODE_ROOT="/home/ycl/projects/workspace/spider"
TASK_NAME="${TASK_NAME:-}"
HZ="${HZ:-20}"

MUSTARD_ROOT="${MUSTARD_ROOT:-${ARIA_ROOT_BASE}/mustard/data}"
DRAWER_ROOT="${DRAWER_ROOT:-${ARIA_ROOT_BASE}/drawer/data}"
FLOWER_ROOT="${FLOWER_ROOT:-${ARIA_ROOT_BASE}/flower/data}"
HAMMER_ROOT="${HAMMER_ROOT:-${ARIA_ROOT_BASE}/hammer/data}"

# RL config (same as run_taco_rl.sh style)
DATASET_DIR="/home/ycl/projects/workspace/spider/example_datasets"
DATASET_NAME="aria"
ROBOT_TYPE="xhand"
EMBODIMENT_TYPE="right"
DATA_ID=0

SOURCE_EMBODIMENT="right"
HAND_MODES=("right")

RL_DEVICE="cuda:0"
RL_NUM_ENVS="4096"
RL_EVAL_NUM_ENVS="256"
RL_FUTURE_CTRL_STEPS="0"
RL_EPISODE_CTRL_STEPS="64"
RL_NUM_STEPS_PER_ENV="16"
RL_LEARNING_ITERS="${RL_LEARNING_ITERS:-300}"
RL_EARLY_STOP_ON_SUCCESS="${RL_EARLY_STOP_ON_SUCCESS:-1}"
RL_CKPT_INTERVAL="500"
RL_VIZ_INTERVAL="40"
RL_ROLLOUT_VIDEO_CTRL_STEPS="0"
RL_TRAIN_SUCCESS_TRIGGER_EVAL="${RL_TRAIN_SUCCESS_TRIGGER_EVAL:-0}"
RL_TRAIN_SUCCESS_TRIGGER_COUNT="${RL_TRAIN_SUCCESS_TRIGGER_COUNT:-0}"
RL_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX="${RL_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX:-8}"

WANDB_MEDIA_ONLY="1"
WANDB_PROJECT="egoengine-spider-rl"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
RL_EXPERIMENT_SUFFIX="${RL_EXPERIMENT_SUFFIX:-}"
LOAD_CKPT="${LOAD_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_MODE=0

SIM_DT="0.01"
CTRL_DT="0.05"
HORIZON="1.6"
KNOT_DT="0.4"

OBJ_POS_REW_SCALE="1.0"
OBJ_ROT_REW_SCALE="0.3"
JOINT_REW_SCALE="0.0001"
BASE_POS_REW_SCALE="0.001"
BASE_ROT_REW_SCALE="0.001"
VEL_REW_SCALE="0.000"
CONTACT_REW_SCALE="0.0"
LIFTING_REW_SCALE="0.0"
OBJECT_POS_THRESHOLD="0.05"
OBJECT_ROT_THRESHOLD="1"
OBJECT_MASS_SCALE="0.1"
ACTION_SMOOTHNESS_REW_SCALE="0.005"
DR_PAIR_MARGIN_MIN="0.0"
DR_PAIR_MARGIN_MAX="0.0"
DR_XY_OFFSET_MIN="0.0"
DR_XY_OFFSET_MAX="0.0"
NUM_DR="0"
DR_OBJECT_MASS_SCALE_MIN="1.0"
DR_OBJECT_MASS_SCALE_MAX="1.0"

FIRST_CTRL_NOISE="0.5"
LAST_CTRL_NOISE="0.65"
FINAL_NOISE="0.1"
JOINT_NOISE_SCALE="0.3"
POS_NOISE_SCALE="0.08"
ROT_NOISE_SCALE="0.05"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

RL_PYTHON_BIN="${RL_PYTHON_BIN:-python}"
RL_LOG_DIR="${CODE_ROOT}/outputs/rl_rsl"

EPISODES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --episodes)
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        EPISODES+=("$1")
        shift
      done
      ;;
    --load-ckpt|--load_ckpt|--load)
      LOAD_CKPT="$2"
      shift 2
      ;;
    --resume-ckpt|--resume_ckpt)
      RESUME_MODE=1
      RESUME_CKPT="$2"
      shift 2
      ;;
    --resume)
      RESUME_MODE=1
      if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
        RESUME_CKPT="$2"
        shift 2
      else
        shift
      fi
      ;;
    -h|--help)
      printf '%s\n' "Usage: bash run_aria_rl.sh --task-name {mustard|drawer|flower|hammer|all} [--episodes EP ...] [--load-ckpt PATH] [--resume [PATH]]"
      printf '%s\n' "Train-trigger env vars: RL_TRAIN_SUCCESS_TRIGGER_EVAL=0 RL_TRAIN_SUCCESS_TRIGGER_COUNT=0 RL_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX=8"
      exit 0
      ;;
    *)
      printf '%s\n' "[ERROR] unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "${TASK_NAME}" ]; then
  printf '%s\n' "[ERROR] please run task by task: --task-name mustard|drawer|flower|hammer"
  exit 1
fi

LOAD_CKPT_FLAG=()
if [[ -n "${LOAD_CKPT}" ]]; then
  LOAD_CKPT_FLAG+=(--load-ckpt "${LOAD_CKPT}")
fi

if [[ -n "${LOAD_CKPT}" && "${RESUME_MODE}" == "1" ]]; then
  printf '%s\n' "[ERROR] use only one of --load-ckpt or --resume"
  exit 1
fi

load_aria_rl_config() {
  local task="$1"
  eval "$(python3 - "${CODE_ROOT}/configs" "${task}" "${CODE_ROOT}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_dir = Path(sys.argv[1])
task = sys.argv[2]
code_root = Path(sys.argv[3])

task_path = cfg_dir / f"aria_{task}_rl.json"
base_name = "aria_rl.json"
if task_path.exists():
    task_json = json.loads(task_path.read_text())
    base_name = task_json.get("base_config", base_name)
else:
    task_json = {}

base = json.loads((cfg_dir / base_name).read_text())

def emit(name, value):
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{name}={shlex.quote(str(value))}")

dataset = base.get("dataset", {})
if "aria_root_base" in dataset:
    emit("ARIA_ROOT_BASE", dataset["aria_root_base"])
if "dataset_dir" in dataset:
    emit("DATASET_DIR", dataset["dataset_dir"])
if "dataset_name" in dataset:
    emit("DATASET_NAME", dataset["dataset_name"])
if "robot_type" in dataset:
    emit("ROBOT_TYPE", dataset["robot_type"])
if "source_embodiment" in dataset:
    emit("SOURCE_EMBODIMENT", dataset["source_embodiment"])
if "hz" in dataset:
    emit("HZ", dataset["hz"])
if "hand_modes" in dataset:
    modes = " ".join(shlex.quote(str(x)) for x in dataset["hand_modes"])
    print(f"HAND_MODES=({modes})")
roots = dataset.get("roots", {})
for name, var in {
    "mustard": "MUSTARD_ROOT",
    "drawer": "DRAWER_ROOT",
    "flower": "FLOWER_ROOT",
    "hammer": "HAMMER_ROOT",
}.items():
    if name in roots:
        emit(var, str(Path(dataset.get("aria_root_base", "")) / roots[name]) if dataset.get("aria_root_base") else roots[name])

run = base.get("run", {})
if "python_bin" in run:
    emit("RL_PYTHON_BIN", run["python_bin"])
if "device" in run:
    emit("RL_DEVICE", run["device"])
if "log_dir" in run:
    log_dir = Path(run["log_dir"])
    emit("RL_LOG_DIR", log_dir if log_dir.is_absolute() else code_root / log_dir)

section_maps = [
    (base.get("rl", {}), {
        "num_envs": "RL_NUM_ENVS",
        "eval_num_envs": "RL_EVAL_NUM_ENVS",
        "future_ctrl_steps": "RL_FUTURE_CTRL_STEPS",
        "max_episode_ctrl_steps": "RL_EPISODE_CTRL_STEPS",
        "num_steps_per_env": "RL_NUM_STEPS_PER_ENV",
        "learning_iterations": "RL_LEARNING_ITERS",
        "early_stop_on_success": "RL_EARLY_STOP_ON_SUCCESS",
        "ckpt_interval": "RL_CKPT_INTERVAL",
        "viz_interval": "RL_VIZ_INTERVAL",
        "rollout_video_ctrl_steps": "RL_ROLLOUT_VIDEO_CTRL_STEPS",
    }),
    (base.get("wandb", {}), {
        "media_only": "WANDB_MEDIA_ONLY",
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "run_name": "WANDB_RUN_NAME",
    }),
    (base.get("timing", {}), {
        "sim_dt": "SIM_DT",
        "ctrl_dt": "CTRL_DT",
        "horizon": "HORIZON",
        "knot_dt": "KNOT_DT",
    }),
    (base.get("reward", {}), {
        "pos_rew_scale": "OBJ_POS_REW_SCALE",
        "rot_rew_scale": "OBJ_ROT_REW_SCALE",
        "joint_rew_scale": "JOINT_REW_SCALE",
        "base_pos_rew_scale": "BASE_POS_REW_SCALE",
        "base_rot_rew_scale": "BASE_ROT_REW_SCALE",
        "vel_rew_scale": "VEL_REW_SCALE",
        "contact_rew_scale": "CONTACT_REW_SCALE",
        "lifting_rew_scale": "LIFTING_REW_SCALE",
        "object_pos_threshold": "OBJECT_POS_THRESHOLD",
        "object_rot_threshold": "OBJECT_ROT_THRESHOLD",
        "object_mass_scale": "OBJECT_MASS_SCALE",
        "action_smoothness_rew_scale": "ACTION_SMOOTHNESS_REW_SCALE",
    }),
    (base.get("domain_randomization", {}), {
        "num_dr": "NUM_DR",
        "pair_margin_range": ("DR_PAIR_MARGIN_MIN", "DR_PAIR_MARGIN_MAX"),
        "xy_offset_range": ("DR_XY_OFFSET_MIN", "DR_XY_OFFSET_MAX"),
        "object_mass_scale_range": ("DR_OBJECT_MASS_SCALE_MIN", "DR_OBJECT_MASS_SCALE_MAX"),
    }),
    (base.get("noise", {}), {
        "first_ctrl_noise_scale": "FIRST_CTRL_NOISE",
        "last_ctrl_noise_scale": "LAST_CTRL_NOISE",
        "final_noise_scale": "FINAL_NOISE",
        "joint_noise_scale": "JOINT_NOISE_SCALE",
        "pos_noise_scale": "POS_NOISE_SCALE",
        "rot_noise_scale": "ROT_NOISE_SCALE",
    }),
]

flat_map = {}
for section, mapping in section_maps:
    for key, var in mapping.items():
        if key not in section:
            continue
        value = section[key]
        if isinstance(var, tuple):
            emit(var[0], value[0])
            emit(var[1], value[1])
        else:
            emit(var, value)
        flat_map[key] = var

overrides = task_json.get("overrides", {})
for key, value in overrides.items():
    var = flat_map.get(key)
    if var is None:
        continue
    if isinstance(var, tuple):
        emit(var[0], value[0])
        emit(var[1], value[1])
    else:
        emit(var, value)
PY
)"
}

task_root() {
  case "$1" in
    mustard) printf '%s\n' "${MUSTARD_ROOT}" ;;
    drawer) printf '%s\n' "${DRAWER_ROOT}" ;;
    flower) printf '%s\n' "${FLOWER_ROOT}" ;;
    hammer) printf '%s\n' "${HAMMER_ROOT}" ;;
    *) return 1 ;;
  esac
}

task_id_from_episode() {
  local task="$1"
  local episode="$2"
  printf '%s\n' "${episode#${task}_}"
}

collect_episodes() {
  local task="$1"
  local root="$2"
  if [ ${#EPISODES[@]} -gt 0 ]; then
    printf '%s\n' "${EPISODES[@]}"
  else
    find "${root}/Hand_Poses" -mindepth 1 -maxdepth 1 -type d -name "${task}_*" -printf '%f\n' | sort -V
  fi
}

trajectory_frame_count() {
  local traj_path="$1"
  "${RL_PYTHON_BIN}" - "${traj_path}" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1]) as data:
    print(int(data["qpos"].shape[0]))
PY
}

TASKS=()
case "${TASK_NAME}" in
  all) TASKS=(mustard drawer flower hammer) ;;
  mustard|drawer|flower|hammer) TASKS=("${TASK_NAME}") ;;
  *)
    printf '%s\n' "[ERROR] unsupported --task-name: ${TASK_NAME}"
    exit 1
    ;;
esac

load_aria_rl_config "${TASKS[0]}"
"${RL_PYTHON_BIN}" -c "import rsl_rl" >/dev/null 2>&1 || {
  printf '%s\n' "[ERROR] rsl_rl not found in ${RL_PYTHON_BIN}. Try: pip install rsl-rl-lib"
  exit 1
}

WANDB_MEDIA_FLAG=()
if [[ "${WANDB_MEDIA_ONLY}" == "1" ]]; then
  WANDB_MEDIA_FLAG+=(--wandb-media-only)
fi

tmp_summary="$(mktemp)"
trap 'rm -f "${tmp_summary}"' EXIT

attempted_count=0
skipped_count=0
failed_count=0
completed_count=0

# ========== RL one by one ==========
for task_name in "${TASKS[@]}"; do
  load_aria_rl_config "${task_name}"
  DATA_ROOT="$(task_root "${task_name}")" || exit 1
  if [ ! -d "${DATA_ROOT}/Hand_Poses" ]; then
    printf '%s\n' "[ERROR] Hand_Poses not found: ${DATA_ROOT}/Hand_Poses"
    failed_count=$((failed_count + 1))
    continue
  fi

  while IFS= read -r episode; do
    [ -z "$episode" ] && continue

    TASK="${task_name}"
    DATA_ID="$(task_id_from_episode "${task_name}" "${episode}")"

    TASK_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}/${DATA_ID}"
    TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}"
    MANO_TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${SOURCE_EMBODIMENT}/${TASK}"
    KIN_PATH="${TASK_DIR}/trajectory_kinematic.npz"
    SCENE_EQ_PATH="${TASK_ROOT_DIR}/scene_eq.xml"
    SRC_DIR="${DATA_ROOT}/Hand_Poses/${episode}/1"

    if [ -f "${SRC_DIR}/trajectory_kinematic.npz" ] && [ -f "${SRC_DIR}/scene_eq.xml" ]; then
      mkdir -p "${TASK_DIR}" "${TASK_ROOT_DIR}" "${MANO_TASK_ROOT_DIR}"
      cp -f "${SRC_DIR}/trajectory_kinematic.npz" "${KIN_PATH}"
      cp -f "${SRC_DIR}/scene_eq.xml" "${TASK_ROOT_DIR}/scene_eq.xml"
      [ -f "${SRC_DIR}/scene.xml" ] && cp -f "${SRC_DIR}/scene.xml" "${TASK_ROOT_DIR}/scene.xml"
      if [ -f "${SRC_DIR}/task_info.json" ]; then
        cp -f "${SRC_DIR}/task_info.json" "${TASK_ROOT_DIR}/task_info.json"
        cp -f "${SRC_DIR}/task_info.json" "${MANO_TASK_ROOT_DIR}/task_info.json"
      fi
    fi

    if [ ! -f "${KIN_PATH}" ] || [ ! -f "${SCENE_EQ_PATH}" ]; then
      printf '%s\n' "[SKIP] missing processed files for ${TASK}_${DATA_ID}"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    TOTAL_FRAMES="$(trajectory_frame_count "${KIN_PATH}")"
    if [ "${TOTAL_FRAMES}" -le 1 ]; then
      printf '%s\n' "[SKIP] trajectory too short for ${TASK}_${DATA_ID}: frames=${TOTAL_FRAMES}"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    for hand_mode in "${HAND_MODES[@]}"; do
      attempted_count=$((attempted_count + 1))
      printf '%s\n' "=== RL Aria ${TASK} - ${episode} (${hand_mode}) ==="

      CUR_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
      if [ -z "${CUR_WANDB_RUN_NAME}" ]; then
        CUR_WANDB_RUN_NAME="rl_${TASK}_${DATA_ID}_${hand_mode}"
      fi

      EXPERIMENT_NAME="${TASK}_${DATA_ID}_rl"
      if [ -n "${RL_EXPERIMENT_SUFFIX}" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RL_EXPERIMENT_SUFFIX}"
        CUR_WANDB_RUN_NAME="${CUR_WANDB_RUN_NAME}_${RL_EXPERIMENT_SUFFIX}"
      fi
      TASK_LOG_DIR="${RL_LOG_DIR}/${EXPERIMENT_NAME}"
      mkdir -p "${TASK_LOG_DIR}"
      RUN_LOG_PATH="${TASK_LOG_DIR}/train.log"

      RESUME_CKPT_FLAG=()
      if [[ "${RESUME_MODE}" == "1" ]]; then
        CUR_RESUME_CKPT="${RESUME_CKPT}"
        if [[ -z "${CUR_RESUME_CKPT}" ]]; then
          CUR_RESUME_CKPT="${TASK_LOG_DIR}/checkpoints/eval_latest.pt"
        fi
        RESUME_CKPT_FLAG+=(--resume-ckpt "${CUR_RESUME_CKPT}")
      fi

      if [[ "${RESUME_MODE}" == "1" ]]; then
        printf '%s\n' "[INFO] resuming from ${RESUME_CKPT_FLAG[1]}"
        TEE_ARGS=(-a "${RUN_LOG_PATH}")
      else
        : > "${RUN_LOG_PATH}"
        TEE_ARGS=("${RUN_LOG_PATH}")
      fi

      if "${RL_PYTHON_BIN}" examples/run_mjwp_rl_rsl.py \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --robot-type "${ROBOT_TYPE}" \
        --embodiment-type "${SOURCE_EMBODIMENT}" \
        --active-hand "${hand_mode}" \
        --task "${TASK}" \
        --data-id "${DATA_ID}" \
        --device "${RL_DEVICE}" \
        --num-envs "${RL_NUM_ENVS}" \
        --eval-num-envs "${RL_EVAL_NUM_ENVS}" \
        --future-ctrl-steps "${RL_FUTURE_CTRL_STEPS}" \
        --max-episode-ctrl-steps "${RL_EPISODE_CTRL_STEPS}" \
        --num-steps-per-env "${RL_NUM_STEPS_PER_ENV}" \
        --learning-iterations "${RL_LEARNING_ITERS}" \
        --early-stop-on-success "${RL_EARLY_STOP_ON_SUCCESS}" \
        --train-success-trigger-eval "${RL_TRAIN_SUCCESS_TRIGGER_EVAL}" \
        --train-success-trigger-count "${RL_TRAIN_SUCCESS_TRIGGER_COUNT}" \
        --train-success-trigger-ctrl-steps "${TOTAL_FRAMES}" \
        --train-success-trigger-backoff-max "${RL_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX}" \
        --ckpt-interval "${RL_CKPT_INTERVAL}" \
        --viz-interval "${RL_VIZ_INTERVAL}" \
        --rollout-video-ctrl-steps "${RL_ROLLOUT_VIDEO_CTRL_STEPS}" \
        --log-dir "${RL_LOG_DIR}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-run-name "${CUR_WANDB_RUN_NAME}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        "${WANDB_MEDIA_FLAG[@]}" \
        "${LOAD_CKPT_FLAG[@]}" \
        "${RESUME_CKPT_FLAG[@]}" \
        --sim-dt "${SIM_DT}" \
        --ctrl-dt "${CTRL_DT}" \
        --horizon "${HORIZON}" \
        --knot-dt "${KNOT_DT}" \
        --pos-rew-scale "${OBJ_POS_REW_SCALE}" \
        --rot-rew-scale "${OBJ_ROT_REW_SCALE}" \
        --joint-rew-scale "${JOINT_REW_SCALE}" \
        --base-pos-rew-scale "${BASE_POS_REW_SCALE}" \
        --base-rot-rew-scale "${BASE_ROT_REW_SCALE}" \
        --vel-rew-scale "${VEL_REW_SCALE}" \
        --contact-rew-scale "${CONTACT_REW_SCALE}" \
        --lifting-rew-scale "${LIFTING_REW_SCALE}" \
        --object-pos-threshold "${OBJECT_POS_THRESHOLD}" \
        --object-rot-threshold "${OBJECT_ROT_THRESHOLD}" \
        --object-mass-scale "${OBJECT_MASS_SCALE}" \
        --action-smoothness-rew-scale "${ACTION_SMOOTHNESS_REW_SCALE}" \
        --num-dr "${NUM_DR}" \
        --pair-margin-range-min "${DR_PAIR_MARGIN_MIN}" \
        --pair-margin-range-max "${DR_PAIR_MARGIN_MAX}" \
        --xy-offset-range-min "${DR_XY_OFFSET_MIN}" \
        --xy-offset-range-max "${DR_XY_OFFSET_MAX}" \
        --object-mass-scale-range-min "${DR_OBJECT_MASS_SCALE_MIN}" \
        --object-mass-scale-range-max "${DR_OBJECT_MASS_SCALE_MAX}" \
        --first-ctrl-noise-scale "${FIRST_CTRL_NOISE}" \
        --last-ctrl-noise-scale "${LAST_CTRL_NOISE}" \
        --final-noise-scale "${FINAL_NOISE}" \
        --joint-noise-scale "${JOINT_NOISE_SCALE}" \
        --pos-noise-scale "${POS_NOISE_SCALE}" \
        --rot-noise-scale "${ROT_NOISE_SCALE}" 2>&1 | tee "${TEE_ARGS[@]}"
      then
        rl_rc=0
      else
        rl_rc=$?
      fi

      rl_line="$(grep -a '^[[]RL[]] task=' "${RUN_LOG_PATH}" | tail -n 1)"
      if [ ${rl_rc} -ne 0 ] || [ -z "${rl_line}" ]; then
        failed_count=$((failed_count + 1))
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode} log=${RUN_LOG_PATH}"
        continue
      fi

      completed_count=$((completed_count + 1))
      fps="$(printf '%s\n' "${rl_line}" | sed -En 's/.*fps=([^ ]*).*/\1/p')"
      reward_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*reward_actual=([^ ]*).*/\1/p')"
      reward_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*reward_max=([^ ]*).*/\1/p')"
      steps_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*steps_actual=([^ ]*).*/\1/p')"
      steps_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*steps_max=([^ ]*).*/\1/p')"
      success="$(printf '%s\n' "${rl_line}" | sed -En 's/.*success=([^ ]*).*/\1/p')"

      printf '%s %s %s %s %s %s %s\n' "${TASK}_${DATA_ID}:${hand_mode}" "${fps}" "${reward_actual}" "${reward_max}" "${steps_actual}" "${steps_max}" "${success}" >> "${tmp_summary}"

      reward_pct="$(awk -v a="${reward_actual}" -v b="${reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
      steps_pct="$(awk -v a="${steps_actual}" -v b="${steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
      printf '%s\n' "[SUMMARY] task=${TASK} episode=${episode} hand=${hand_mode} fps=${fps} reward=${reward_actual}/${reward_max} (${reward_pct}) steps=${steps_actual}/${steps_max} (${steps_pct}) success=${success}"
    done
  done < <(collect_episodes "${task_name}" "${DATA_ROOT}")
done

if [ ${completed_count} -gt 0 ]; then
  read -r avg_fps mean_reward_actual mean_reward_max mean_steps_actual mean_steps_max success_rate <<EOF_SUMMARY
$(awk '{fps+=$2; reward_actual+=$3; reward_max+=$4; steps_actual+=$5; steps_max+=$6; success+=$7; n+=1} END {if (n>0) printf "%.6f %.6f %.6f %.6f %.6f %.6f", fps/n, reward_actual/n, reward_max/n, steps_actual/n, steps_max/n, success/n; else printf "nan nan nan nan nan nan"}' "${tmp_summary}")
EOF_SUMMARY
else
  avg_fps="nan"
  mean_reward_actual="nan"
  mean_reward_max="nan"
  mean_steps_actual="nan"
  mean_steps_max="nan"
  success_rate="nan"
fi

if [ "${mean_reward_actual}" != "nan" ]; then
  avg_reward_pct="$(awk -v a="${mean_reward_actual}" -v b="${mean_reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
  avg_steps_pct="$(awk -v a="${mean_steps_actual}" -v b="${mean_steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
  success_rate_pct="$(awk -v x="${success_rate}" 'BEGIN { printf "%.2f%%", 100.0 * x }')"
else
  avg_reward_pct="nan"
  avg_steps_pct="nan"
  success_rate_pct="nan"
fi

printf '%s\n' "[FINAL] demos=${completed_count} fps=${avg_fps} reward=${mean_reward_actual}/${mean_reward_max} (${avg_reward_pct}) steps=${mean_steps_actual}/${mean_steps_max} (${avg_steps_pct}) success_rate=${success_rate_pct}"
if [ ${skipped_count} -gt 0 ] || [ ${failed_count} -gt 0 ]; then
  printf '%s\n' "[FINAL] skipped=${skipped_count} failed=${failed_count} attempted=${attempted_count}"
fi
exit $?
