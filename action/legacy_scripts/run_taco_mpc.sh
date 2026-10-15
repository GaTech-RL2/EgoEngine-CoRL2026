#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${SCRIPT_DIR}"
LIST_FILE="${CODE_ROOT}/vis_taco_usable_demos.txt"
MPC_RUN_PY="${CODE_ROOT}/examples/run_mjwp_mpc.py"
DATASET_ROOT="/home/ycl/projects/datasets/egoengine"

DATASET_DIR="example_datasets"
DATASET_NAME="gigahand"
ROBOT_TYPE="xhand"
HAND_MODES=("left" "right")
SOURCE_EMBODIMENT="bimanual"
DATA_ID=0

SIM_DT="0.01"
CTRL_DT="0.05"
HORIZON="0.5"
KNOT_DT="0.05"
MPC_NUM_SAMPLES="2048"
MPC_MAX_ITERS="32"
MPC_IMPROVE_THRESH="0.005"
MPC_TEMPERATURE="0.1"

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

MPC_DEVICE="${MPC_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHOW_VIEWER="${SHOW_VIEWER:-0}"
VIEWER_BACKEND="${VIEWER_BACKEND:-mujoco}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

if [ ! -f "${LIST_FILE}" ]; then
  printf '%s\n' "[ERROR] list file not found: ${LIST_FILE}"
  exit 1
fi

if [ ! -f "${MPC_RUN_PY}" ]; then
  printf '%s\n' "[ERROR] mpc entry not found: ${MPC_RUN_PY}"
  exit 1
fi

tmp_summary="$(mktemp)"
trap 'rm -f "${tmp_summary}"' EXIT

attempted_count=0
skipped_count=0
failed_count=0
completed_count=0

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
    mpc_output="$({
      cd "${CODE_ROOT}" || exit 1
      viewer_value="none"
      show_viewer_value="false"
      if [ "${SHOW_VIEWER}" = "1" ]; then
        viewer_value="${VIEWER_BACKEND}"
        show_viewer_value="true"
      fi
      PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" "examples/run_mjwp_mpc.py" \
        dataset_dir="${DATASET_DIR}" \
        dataset_name="${DATASET_NAME}" \
        robot_type="${ROBOT_TYPE}" \
        embodiment_type="${SOURCE_EMBODIMENT}" \
        +active_hand="${hand_mode}" \
        task="${TASK}" \
        data_id="${DATA_ID}" \
        device="${MPC_DEVICE}" \
        sim_dt="${SIM_DT}" \
        ctrl_dt="${CTRL_DT}" \
        horizon="${HORIZON}" \
        knot_dt="${KNOT_DT}" \
        num_samples="${MPC_NUM_SAMPLES}" \
        max_num_iterations="${MPC_MAX_ITERS}" \
        improvement_threshold="${MPC_IMPROVE_THRESH}" \
        temperature="${MPC_TEMPERATURE}" \
        pos_rew_scale="${OBJ_POS_REW_SCALE}" \
        rot_rew_scale="${OBJ_ROT_REW_SCALE}" \
        joint_rew_scale="${JOINT_REW_SCALE}" \
        base_pos_rew_scale="${BASE_POS_REW_SCALE}" \
        base_rot_rew_scale="${BASE_ROT_REW_SCALE}" \
        vel_rew_scale="${VEL_REW_SCALE}" \
        +contact_rew_scale="${CONTACT_REW_SCALE}" \
        object_pos_threshold="${OBJECT_POS_THRESHOLD}" \
        object_rot_threshold="${OBJECT_ROT_THRESHOLD}" \
        +object_mass_scale="${OBJECT_MASS_SCALE}" \
        viewer="${viewer_value}" \
        show_viewer="${show_viewer_value}" \
        rerun_spawn=false \
        save_video=false \
        save_rerun=false \
        save_metrics=false \
        save_info=false \
        save_config=false
    } 2>&1)"
    mpc_rc=$?

    mpc_line="$(printf '%s\n' "${mpc_output}" | grep '^\[MPC\]' | tail -n 1)"
    if [ ${mpc_rc} -ne 0 ] || [ -z "${mpc_line}" ]; then
      failed_count=$((failed_count + 1))
      printf '%s\n' "[FAIL] task=${TASK} hand=${hand_mode}"
      if [ -n "${mpc_output}" ]; then
        printf '%s\n' "${mpc_output}"
      fi
      continue
    fi

    completed_count=$((completed_count + 1))
    fps="$(printf '%s\n' "${mpc_line}" | sed -n 's/.*fps=\([^ ]*\).*/\1/p')"
    reward_actual="$(printf '%s\n' "${mpc_line}" | sed -n 's/.*reward_actual=\([^ ]*\).*/\1/p')"
    reward_max="$(printf '%s\n' "${mpc_line}" | sed -n 's/.*reward_max=\([^ ]*\).*/\1/p')"
    steps_actual="$(printf '%s\n' "${mpc_line}" | sed -n 's/.*steps_actual=\([^ ]*\).*/\1/p')"
    steps_max="$(printf '%s\n' "${mpc_line}" | sed -n 's/.*steps_max=\([^ ]*\).*/\1/p')"
    success="$(printf '%s\n' "${mpc_line}" | sed -n 's/.*success=\([^ ]*\).*/\1/p')"

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
