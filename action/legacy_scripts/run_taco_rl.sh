#!/usr/bin/env bash
""":"
set -u
set -o pipefail

# WARNING: plaintext secret in script (user requested). Prefer exporting in shell instead.
export WANDB_API_KEY="${WANDB_API_KEY:-}"

DATASET_ROOT="/home/ycl/projects/datasets/egoengine"
CODE_ROOT="/home/ycl/projects/workspace/spider"
LIST_FILE="${CODE_ROOT}/vis_taco_usable_demos.txt"
HZ="${HZ:-20}"

# RL config (same as run_aria_rl.sh style)
DATASET_DIR="/home/ycl/projects/workspace/spider/example_datasets"
DATASET_NAME="gigahand"
ROBOT_TYPE="xhand"
EMBODIMENT_TYPE="right"
DATA_ID=0

SOURCE_EMBODIMENT="bimanual"
HAND_MODES=("left" "right")

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

WANDB_MEDIA_ONLY="1"
WANDB_PROJECT="egoengine-spider-rl"
WANDB_ENTITY=""
WANDB_RUN_NAME=""

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
OBJECT_POS_THRESHOLD="0.05"
OBJECT_ROT_THRESHOLD="1"
OBJECT_MASS_SCALE="0.1"
ACTION_SMOOTHNESS_REW_SCALE="0.000"
DR_PAIR_MARGIN_MIN="0.0"
DR_PAIR_MARGIN_MAX="0.0"
DR_XY_OFFSET_MIN="0.0"
DR_XY_OFFSET_MAX="0.0"
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

if [ ! -f "${LIST_FILE}" ]; then
  printf '%s\n' "[ERROR] list file not found: ${LIST_FILE}"
  exit 1
fi

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
while IFS=$'\t' read -r triplet sequence; do
  [ -z "$triplet" ] && continue
  [ -z "$sequence" ] && continue

  task_triplet="$(printf '%s' "$triplet" | sed -e 's/[()]//g' -e 's/, /_/g' -e 's/,/_/g' -e 's/ /_/g')"
  TASK="${task_triplet}_${sequence}"

  TASK_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}/${DATA_ID}"
  TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}"
  KIN_PATH="${TASK_DIR}/trajectory_kinematic.npz"
  SCENE_EQ_PATH="${TASK_ROOT_DIR}/scene_eq.xml"

  if [ ! -f "${KIN_PATH}" ] || [ ! -f "${SCENE_EQ_PATH}" ]; then
    printf '%s\n' "[SKIP] missing processed files for ${TASK}"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  for hand_mode in "${HAND_MODES[@]}"; do
    attempted_count=$((attempted_count + 1))
    printf '%s\n' "=== RL TACO ${triplet} - ${sequence} (${hand_mode}) ==="

    CUR_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
    if [ -z "${CUR_WANDB_RUN_NAME}" ]; then
      CUR_WANDB_RUN_NAME="rl_${TASK}_${hand_mode}"
    fi

    EXPERIMENT_NAME="${TASK}_rl"
    TASK_LOG_DIR="${RL_LOG_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "${TASK_LOG_DIR}"
    RUN_LOG_PATH="${TASK_LOG_DIR}/train.log"
    : > "${RUN_LOG_PATH}"

    CUR_OBJECT_ROT_THRESHOLD="1.5"
    CUR_CONTACT_REW_SCALE="2.0"
    CUR_JOINT_REW_SCALE="0.005"
    CUR_JOINT_NOISE_SCALE="0.4"
    CUR_ROT_NOISE_SCALE="0.7"

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
      --ckpt-interval "${RL_CKPT_INTERVAL}" \
      --viz-interval "${RL_VIZ_INTERVAL}" \
      --rollout-video-ctrl-steps "${RL_ROLLOUT_VIDEO_CTRL_STEPS}" \
      --log-dir "${RL_LOG_DIR}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-run-name "${CUR_WANDB_RUN_NAME}" \
      --experiment-name "${EXPERIMENT_NAME}" \
      "${WANDB_MEDIA_FLAG[@]}" \
      --sim-dt "${SIM_DT}" \
      --ctrl-dt "${CTRL_DT}" \
      --horizon "${HORIZON}" \
      --knot-dt "${KNOT_DT}" \
      --pos-rew-scale "${OBJ_POS_REW_SCALE}" \
      --rot-rew-scale "${OBJ_ROT_REW_SCALE}" \
      --joint-rew-scale "${CUR_JOINT_REW_SCALE}" \
      --base-pos-rew-scale "${BASE_POS_REW_SCALE}" \
      --base-rot-rew-scale "${BASE_ROT_REW_SCALE}" \
      --vel-rew-scale "${VEL_REW_SCALE}" \
      --contact-rew-scale "${CUR_CONTACT_REW_SCALE}" \
      --object-pos-threshold "${OBJECT_POS_THRESHOLD}" \
      --object-rot-threshold "${CUR_OBJECT_ROT_THRESHOLD}" \
      --object-mass-scale "${OBJECT_MASS_SCALE}" \
      --action-smoothness-rew-scale "${ACTION_SMOOTHNESS_REW_SCALE}" \
      --num-dr "0" \
      --pair-margin-range-min "${DR_PAIR_MARGIN_MIN}" \
      --pair-margin-range-max "${DR_PAIR_MARGIN_MAX}" \
      --xy-offset-range-min "${DR_XY_OFFSET_MIN}" \
      --xy-offset-range-max "${DR_XY_OFFSET_MAX}" \
      --object-mass-scale-range-min "${DR_OBJECT_MASS_SCALE_MIN}" \
      --object-mass-scale-range-max "${DR_OBJECT_MASS_SCALE_MAX}" \
      --first-ctrl-noise-scale "${FIRST_CTRL_NOISE}" \
      --last-ctrl-noise-scale "${LAST_CTRL_NOISE}" \
      --final-noise-scale "${FINAL_NOISE}" \
      --joint-noise-scale "${CUR_JOINT_NOISE_SCALE}" \
      --pos-noise-scale "${POS_NOISE_SCALE}" \
      --rot-noise-scale "${CUR_ROT_NOISE_SCALE}" 2>&1 | tee "${RUN_LOG_PATH}"
    then
      rl_rc=0
    else
      rl_rc=$?
    fi

    rl_line="$(grep '^[[]RL[]] task=' "${RUN_LOG_PATH}" | tail -n 1)"
    if [ ${rl_rc} -ne 0 ] || [ -z "${rl_line}" ]; then
      failed_count=$((failed_count + 1))
      printf '%s\n' "[FAIL] task=${TASK} hand=${hand_mode} log=${RUN_LOG_PATH}"
      continue
    fi

    completed_count=$((completed_count + 1))
    fps="$(printf '%s\n' "${rl_line}" | sed -En 's/.*fps=([^ ]*).*/\1/p')"
    reward_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*reward_actual=([^ ]*).*/\1/p')"
    reward_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*reward_max=([^ ]*).*/\1/p')"
    steps_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*steps_actual=([^ ]*).*/\1/p')"
    steps_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*steps_max=([^ ]*).*/\1/p')"
    success="$(printf '%s\n' "${rl_line}" | sed -En 's/.*success=([^ ]*).*/\1/p')"

    printf '%s %s %s %s %s %s %s\n' "${TASK}:${hand_mode}" "${fps}" "${reward_actual}" "${reward_max}" "${steps_actual}" "${steps_max}" "${success}" >> "${tmp_summary}"

    reward_pct="$(awk -v a="${reward_actual}" -v b="${reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
    steps_pct="$(awk -v a="${steps_actual}" -v b="${steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
    printf '%s\n' "[SUMMARY] task=${TASK} hand=${hand_mode} fps=${fps} reward=${reward_actual}/${reward_max} (${reward_pct}) steps=${steps_actual}/${steps_max} (${steps_pct}) success=${success}"
  done
done < "${LIST_FILE}"

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
":"""

import os
import pathlib
import sys


source = pathlib.Path(__file__).read_text(encoding="utf-8")
start_marker = '""":"\n'
end_marker = '\n":"""'
start = source.index(start_marker) + len(start_marker)
end = source.index(end_marker, start)
os.execvp("bash", ["bash", "-c", source[start:end], __file__, *sys.argv[1:]])
