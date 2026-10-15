#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${SCRIPT_DIR}"
MPC_RUN_PY="${CODE_ROOT}/examples/run_mjwp_mpc.py"

TASK_NAME="${TASK_NAME:-}"
ARIA_ROOT_BASE="/home/ycl/projects/datasets"
MUSTARD_ROOT="${MUSTARD_ROOT:-${ARIA_ROOT_BASE}/mustard/data}"
DRAWER_ROOT="${DRAWER_ROOT:-${ARIA_ROOT_BASE}/drawer/data}"
FLOWER_ROOT="${FLOWER_ROOT:-${ARIA_ROOT_BASE}/flower/data}"
HAMMER_ROOT="${HAMMER_ROOT:-${ARIA_ROOT_BASE}/hammer/data}"

DATASET_DIR="${DATASET_DIR:-${CODE_ROOT}/example_datasets}"
DATASET_NAME="aria"
ROBOT_TYPE="xhand"
HAND_MODES=("right")
SOURCE_EMBODIMENT="right"
DATA_ID=0

SIM_DT="0.01"
CTRL_DT="0.05"
HORIZON="0.2"
KNOT_DT="0.05"
MPC_NUM_SAMPLES="2048"
MPC_MAX_ITERS="32"
MPC_IMPROVE_THRESH="0.005"
MPC_TEMPERATURE="0.1"

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
EARLY_TERMINATION="${EARLY_TERMINATION:-1}"
FIRST_CTRL_NOISE="0.5"
LAST_CTRL_NOISE="0.65"
FINAL_NOISE="0.1"
JOINT_NOISE_SCALE="0.3"
POS_NOISE_SCALE="0.08"
ROT_NOISE_SCALE="0.05"

MPC_DEVICE="${MPC_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHOW_VIEWER="${SHOW_VIEWER:-0}"
VIEWER_BACKEND="${VIEWER_BACKEND:-mujoco}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_DIR="${VIDEO_DIR:-${CODE_ROOT}/outputs/aria_mpc_videos}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

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
    --viewer)
      SHOW_VIEWER=1
      shift
      ;;
    -h|--help)
      printf '%s\n' "Usage: bash run_aria_mpc.sh --task-name {mustard|drawer|flower|hammer|all} [--episodes EP ...] [--viewer]"
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

if [ ! -f "${MPC_RUN_PY}" ]; then
  printf '%s\n' "[ERROR] mpc entry not found: ${MPC_RUN_PY}"
  exit 1
fi

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

TASKS=()
case "${TASK_NAME}" in
  all) TASKS=(mustard drawer flower hammer) ;;
  mustard|drawer|flower|hammer) TASKS=("${TASK_NAME}") ;;
  *)
    printf '%s\n' "[ERROR] unsupported --task-name: ${TASK_NAME}"
    exit 1
    ;;
esac

tmp_summary="$(mktemp)"
trap 'rm -f "${tmp_summary}"' EXIT

attempted_count=0
skipped_count=0
failed_count=0
completed_count=0

for task_name in "${TASKS[@]}"; do
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
    SCENE_DIR="$(dirname "${TASK_DIR}")"
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

    if [ ! -f "${KIN_PATH}" ]; then
      printf '%s\n' "[SKIP] task=${TASK} episode=${episode} reason=missing_trajectory"
      skipped_count=$((skipped_count + 1))
      continue
    fi
    if [ ! -f "${SCENE_DIR}/scene.xml" ] && [ ! -f "${SCENE_DIR}/scene_eq.xml" ]; then
      printf '%s\n' "[SKIP] task=${TASK} episode=${episode} reason=missing_scene"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    for hand_mode in "${HAND_MODES[@]}"; do
      attempted_count=$((attempted_count + 1))
      CUR_EARLY_TERMINATION="${EARLY_TERMINATION}"
      CUR_OBJ_POS_REW_SCALE="${OBJ_POS_REW_SCALE}"
      CUR_OBJ_ROT_REW_SCALE="${OBJ_ROT_REW_SCALE}"
      CUR_OBJECT_ROT_THRESHOLD="${OBJECT_ROT_THRESHOLD}"
      CUR_CONTACT_REW_SCALE="${CONTACT_REW_SCALE}"
      CUR_JOINT_REW_SCALE="${JOINT_REW_SCALE}"
      CUR_JOINT_NOISE_SCALE="${JOINT_NOISE_SCALE}"
      CUR_ROT_NOISE_SCALE="${ROT_NOISE_SCALE}"
      if [ "${TASK}" = "mustard" ]; then
        CUR_OBJECT_ROT_THRESHOLD="1.5"
        CUR_CONTACT_REW_SCALE="2.0"
        CUR_JOINT_REW_SCALE="0.005"
        CUR_JOINT_NOISE_SCALE="0.4"
        CUR_ROT_NOISE_SCALE="0.7"
      elif [ "${TASK}" = "drawer" ]; then
        CUR_OBJECT_ROT_THRESHOLD="1.5"
        CUR_CONTACT_REW_SCALE="2.0"
        CUR_JOINT_REW_SCALE="0.005"
        CUR_JOINT_NOISE_SCALE="0.4"
        CUR_ROT_NOISE_SCALE="0.7"
      elif [ "${TASK}" = "flower" ]; then
        CUR_OBJECT_ROT_THRESHOLD="1.5"
        CUR_CONTACT_REW_SCALE="2.0"
        CUR_JOINT_REW_SCALE="0.005"
        CUR_JOINT_NOISE_SCALE="0.4"
        CUR_ROT_NOISE_SCALE="0.7"
      elif [ "${TASK}" = "hammer" ]; then
        CUR_EARLY_TERMINATION="0"
        CUR_OBJECT_ROT_THRESHOLD="1.5"
        CUR_CONTACT_REW_SCALE="2.0"
        CUR_JOINT_REW_SCALE="0.005"
        CUR_JOINT_NOISE_SCALE="0.4"
        CUR_ROT_NOISE_SCALE="1.0"
      fi
      mpc_output="$({
        cd "${CODE_ROOT}" || exit 1
        viewer_value="none"
        show_viewer_value="false"
        save_video_value="false"
        if [ "${SHOW_VIEWER}" = "1" ]; then
          viewer_value="${VIEWER_BACKEND}"
          show_viewer_value="true"
        fi
        if [ "${SAVE_VIDEO}" = "1" ]; then
          save_video_value="true"
          mkdir -p "${VIDEO_DIR}/${TASK}_${DATA_ID}_${hand_mode}"
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
          pos_rew_scale="${CUR_OBJ_POS_REW_SCALE}" \
          rot_rew_scale="${CUR_OBJ_ROT_REW_SCALE}" \
          joint_rew_scale="${CUR_JOINT_REW_SCALE}" \
          base_pos_rew_scale="${BASE_POS_REW_SCALE}" \
          base_rot_rew_scale="${BASE_ROT_REW_SCALE}" \
          vel_rew_scale="${VEL_REW_SCALE}" \
          +contact_rew_scale="${CUR_CONTACT_REW_SCALE}" \
          object_pos_threshold="${OBJECT_POS_THRESHOLD}" \
          object_rot_threshold="${CUR_OBJECT_ROT_THRESHOLD}" \
          +object_mass_scale="${OBJECT_MASS_SCALE}" \
          +early_termination="${CUR_EARLY_TERMINATION}" \
          first_ctrl_noise_scale="${FIRST_CTRL_NOISE}" \
          last_ctrl_noise_scale="${LAST_CTRL_NOISE}" \
          final_noise_scale="${FINAL_NOISE}" \
          joint_noise_scale="${CUR_JOINT_NOISE_SCALE}" \
          pos_noise_scale="${POS_NOISE_SCALE}" \
          rot_noise_scale="${CUR_ROT_NOISE_SCALE}" \
          viewer="${viewer_value}" \
          show_viewer="${show_viewer_value}" \
          rerun_spawn=false \
          save_video="${save_video_value}" \
          output_dir="${VIDEO_DIR}/${TASK}_${DATA_ID}_${hand_mode}" \
          save_rerun=false \
          save_metrics=false \
          save_info=false \
          save_config=false
      } 2>&1)"
      mpc_rc=$?

      mpc_line="$(printf '%s\n' "${mpc_output}" | grep '^\[MPC\]' | tail -n 1)"
      if [ ${mpc_rc} -ne 0 ] || [ -z "${mpc_line}" ]; then
        failed_count=$((failed_count + 1))
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode}"
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
