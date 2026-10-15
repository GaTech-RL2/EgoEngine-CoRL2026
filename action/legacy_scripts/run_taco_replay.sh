#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${SCRIPT_DIR}"
LIST_FILE="${CODE_ROOT}/vis_taco_usable_demos.txt"
REPLAY_RUN_PY="${CODE_ROOT}/examples/run_mjwp_replay.py"
DATASET_ROOT="/home/ycl/projects/datasets/egoengine"
DATASET_ROOT="/nethome/scheng77/flash7/scheng77/spider"

DATASET_DIR="example_datasets"
DATASET_NAME="gigahand"
ROBOT_TYPE="xhand"
HAND_MODES=("left" "right")
SOURCE_EMBODIMENT="bimanual"
DATA_ID=0

SIM_DT="0.01"
CTRL_DT="0.05"
HORIZON="1.6"
KNOT_DT="0.4"

OBJ_POS_REW_SCALE="1.0"
OBJ_ROT_REW_SCALE="0.3"
JOINT_REW_SCALE="0"
BASE_POS_REW_SCALE="0"
BASE_ROT_REW_SCALE="0"
VEL_REW_SCALE="0.000"
CONTACT_REW_SCALE="0.0"
OBJECT_POS_THRESHOLD="0.05"
OBJECT_ROT_THRESHOLD="1"
OBJECT_MASS_SCALE="0.1"

REPLAY_DEVICE="${REPLAY_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHOW_VIEWER="${SHOW_VIEWER:-0}"
VIEWER_BACKEND="${VIEWER_BACKEND:-mujoco}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_DIR="${VIDEO_DIR:-${CODE_ROOT}/outputs/taco_replay_videos}"
EARLY_TERMINATION="${EARLY_TERMINATION:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/spider_warp_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/spider_xdg_cache}"
mkdir -p "${WARP_CACHE_PATH}" "${XDG_CACHE_HOME}"

if [ ! -f "${LIST_FILE}" ]; then
  printf '%s\n' "[ERROR] list file not found: ${LIST_FILE}"
  exit 1
fi

if [ ! -f "${REPLAY_RUN_PY}" ]; then
  printf '%s\n' "[ERROR] replay entry not found: ${REPLAY_RUN_PY}"
  exit 1
fi

VIEWER_ARGS=()
if [ "${SHOW_VIEWER}" = "1" ]; then
  VIEWER_ARGS+=(--show-viewer --viewer "${VIEWER_BACKEND}")
fi

VIDEO_ARGS=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  mkdir -p "${VIDEO_DIR}"
  VIDEO_ARGS+=(--save-video)
fi

tmp_summary="$(mktemp)"
trap 'rm -f "${tmp_summary}"' EXIT

attempted_count=0
skipped_count=0
failed_count=0
completed_count=0
success_count=0

while IFS=$'\t' read -r triplet sequence; do
  [ -z "$triplet" ] && continue
  [ -z "$sequence" ] && continue

  task_triplet="$(printf '%s' "$triplet" | sed -e 's/[()]//g' -e 's/, /_/g' -e 's/,/_/g' -e 's/ /_/g')"
  TASK="${task_triplet}_${sequence}"

  TASK_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}/${DATA_ID}"
  TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}"
  MANO_TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${SOURCE_EMBODIMENT}/${TASK}"
  KIN_PATH="${TASK_DIR}/trajectory_kinematic.npz"
  SCENE_DIR="$(dirname "${TASK_DIR}")"
  SRC_DIR="${DATASET_ROOT}/TACO/Hand_Poses/${triplet}/${sequence}"

  # Sync already-generated digital twin files from TACO source layout.
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

  if [ ! -f "${KIN_PATH}" ]; then
    printf '%s\n' "[SKIP] task=${TASK} reason=missing_trajectory"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  if [ ! -f "${SCENE_DIR}/scene.xml" ] && [ ! -f "${SCENE_DIR}/scene_eq.xml" ]; then
    printf '%s\n' "[SKIP] task=${TASK} reason=missing_scene"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  for hand_mode in "${HAND_MODES[@]}"; do
    attempted_count=$((attempted_count + 1))
    cur_video_args=("${VIDEO_ARGS[@]}")
    if [ "${SAVE_VIDEO}" = "1" ]; then
      if [ "${EARLY_TERMINATION}" = "0" ]; then
        video_suffix="no_early_termination"
      else
        video_suffix="early_termination"
      fi
      cur_video_args+=(--video-path "${VIDEO_DIR}/${TASK}_${DATA_ID}_${hand_mode}_replay_${video_suffix}.mp4")
    fi
    replay_output="$({
      cd "${CODE_ROOT}" || exit 1
      PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" "examples/run_mjwp_replay.py" \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --robot-type "${ROBOT_TYPE}" \
        --embodiment-type "${SOURCE_EMBODIMENT}" \
        --active-hand "${hand_mode}" \
        --task "${TASK}" \
        --data-id "${DATA_ID}" \
        --device "${REPLAY_DEVICE}" \
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
        --object-pos-threshold "${OBJECT_POS_THRESHOLD}" \
        --object-rot-threshold "${OBJECT_ROT_THRESHOLD}" \
        --object-mass-scale "${OBJECT_MASS_SCALE}" \
        --early-termination "${EARLY_TERMINATION}" \
        "${cur_video_args[@]}" \
        "${VIEWER_ARGS[@]}"
    } 2>&1)"
    replay_rc=$?

    replay_line="$(printf '%s\n' "${replay_output}" | grep '^\[REPLAY\]' | tail -n 1)"
    if [ ${replay_rc} -ne 0 ] || [ -z "${replay_line}" ]; then
      failed_count=$((failed_count + 1))
      printf '%s\n' "[FAIL] task=${TASK} hand=${hand_mode}"
      if [ -n "${replay_output}" ]; then
        printf '%s\n' "${replay_output}"
      fi
      continue
    fi

    completed_count=$((completed_count + 1))
    status="$(printf '%s\n' "${replay_line}" | sed -n 's/.*status=\([^ ]*\).*/\1/p')"
    fps="$(printf '%s\n' "${replay_line}" | sed -n 's/.*fps=\([^ ]*\).*/\1/p')"
    reward_actual="$(printf '%s\n' "${replay_line}" | sed -n 's/.*reward_actual=\([^ ]*\).*/\1/p')"
    reward_max="$(printf '%s\n' "${replay_line}" | sed -n 's/.*reward_max=\([^ ]*\).*/\1/p')"
    steps_actual="$(printf '%s\n' "${replay_line}" | sed -n 's/.*steps_actual=\([^ ]*\).*/\1/p')"
    steps_max="$(printf '%s\n' "${replay_line}" | sed -n 's/.*steps_max=\([^ ]*\).*/\1/p')"
    success="$(printf '%s\n' "${replay_line}" | sed -n 's/.*success=\([^ ]*\).*/\1/p')"

    if [ "${success}" = "1" ]; then
      success_count=$((success_count + 1))
    fi

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
