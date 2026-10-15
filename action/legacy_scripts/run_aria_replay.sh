#!/usr/bin/env bash
set -u

# Replay already-generated Aria digital twins and report reward, like run_taco_replay.sh.
# It does not regenerate scene.xml / trajectory_kinematic.npz / ik.mp4.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${SCRIPT_DIR}"
REPLAY_RUN_PY="${CODE_ROOT}/examples/run_mjwp_replay.py"

DATASET_DIR="${DATASET_DIR:-${CODE_ROOT}/example_datasets}"
DATASET_NAME="${DATASET_NAME:-aria}"
ROBOT_TYPE="${ROBOT_TYPE:-xhand}"
EMBODIMENT_TYPE="${EMBODIMENT_TYPE:-right}"
ACTIVE_HAND="${ACTIVE_HAND:-right}"
TASK_NAME="${TASK_NAME:-}"
DATA_ID=0

MUSTARD_ROOT="${MUSTARD_ROOT:-/home/ycl/projects/datasets/mustard/data}"
DRAWER_ROOT="${DRAWER_ROOT:-/home/ycl/projects/datasets/drawer/data}"
FLOWER_ROOT="${FLOWER_ROOT:-/home/ycl/projects/datasets/flower/data}"
HAMMER_ROOT="${HAMMER_ROOT:-/home/ycl/projects/datasets/hammer/data}"

SIM_DT="${SIM_DT:-0.01}"
CTRL_DT="${CTRL_DT:-0.05}"
HORIZON="${HORIZON:-1.6}"
KNOT_DT="${KNOT_DT:-0.4}"

OBJ_POS_REW_SCALE="${OBJ_POS_REW_SCALE:-1.0}"
OBJ_ROT_REW_SCALE="${OBJ_ROT_REW_SCALE:-0.3}"
JOINT_REW_SCALE="${JOINT_REW_SCALE:-0.0001}"
BASE_POS_REW_SCALE="${BASE_POS_REW_SCALE:-0.001}"
BASE_ROT_REW_SCALE="${BASE_ROT_REW_SCALE:-0.001}"
VEL_REW_SCALE="${VEL_REW_SCALE:-0.000}"
CONTACT_REW_SCALE="${CONTACT_REW_SCALE:-0.0}"
OBJECT_POS_THRESHOLD="${OBJECT_POS_THRESHOLD:-0.05}"
OBJECT_ROT_THRESHOLD="${OBJECT_ROT_THRESHOLD:-1}"
OBJECT_MASS_SCALE="${OBJECT_MASS_SCALE:-0.1}"

REPLAY_DEVICE="${REPLAY_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHOW_VIEWER="${SHOW_VIEWER:-0}"
VIEWER_BACKEND="${VIEWER_BACKEND:-mujoco}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_DIR="${VIDEO_DIR:-${CODE_ROOT}/outputs/aria_replay_videos}"
EARLY_TERMINATION="${EARLY_TERMINATION:-1}"
EPISODES=()

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

usage() {
  cat <<'EOF'
Usage:
  bash run_aria_replay.sh --task-name TASK [options]

Options:
  --task-name {all|mustard|drawer|flower|hammer}
  --episodes EPISODE [EPISODE ...]
  --viewer
  --save-video
  --video-dir DIR
  --early-termination {0|1}

Environment overrides:
  PYTHON_BIN, DATASET_DIR, DATASET_NAME, REPLAY_DEVICE,
  MUSTARD_ROOT, DRAWER_ROOT, FLOWER_ROOT, HAMMER_ROOT
EOF
}

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
    --save-video)
      SAVE_VIDEO=1
      shift
      ;;
    --video-dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --early-termination)
      EARLY_TERMINATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '%s\n' "[ERROR] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${TASK_NAME}" ]; then
  printf '%s\n' "[ERROR] please run task by task: --task-name mustard|drawer|flower|hammer"
  printf '%s\n' "        use --task-name all only when you explicitly want every task."
  usage
  exit 1
fi

if [ ! -f "${REPLAY_RUN_PY}" ]; then
  printf '%s\n' "[ERROR] replay entry not found: ${REPLAY_RUN_PY}"
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

sync_demo() {
  local task="$1"
  local src_dir="$2"
  local data_id="$3"
  local task_root_dir="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${EMBODIMENT_TYPE}/${task}"
  local data_dir="${task_root_dir}/${data_id}"
  local mano_task_root_dir="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${EMBODIMENT_TYPE}/${task}"

  if [ ! -f "${src_dir}/trajectory_kinematic.npz" ] || [ ! -f "${src_dir}/scene_eq.xml" ]; then
    return 1
  fi

  mkdir -p "${data_dir}" "${task_root_dir}" "${mano_task_root_dir}"
  cp -f "${src_dir}/trajectory_kinematic.npz" "${data_dir}/trajectory_kinematic.npz"
  cp -f "${src_dir}/scene_eq.xml" "${task_root_dir}/scene_eq.xml"
  [ -f "${src_dir}/scene.xml" ] && cp -f "${src_dir}/scene.xml" "${task_root_dir}/scene.xml"
  if [ -f "${src_dir}/task_info.json" ]; then
    cp -f "${src_dir}/task_info.json" "${task_root_dir}/task_info.json"
    cp -f "${src_dir}/task_info.json" "${mano_task_root_dir}/task_info.json"
  fi
  return 0
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
    usage
    exit 1
    ;;
esac

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

for task in "${TASKS[@]}"; do
  root="$(task_root "${task}")" || exit 1
  if [ ! -d "${root}/Hand_Poses" ]; then
    printf '%s\n' "[FAIL] task=${task} missing Hand_Poses: ${root}/Hand_Poses"
    failed_count=$((failed_count + 1))
    continue
  fi

  while IFS= read -r episode; do
    [ -z "${episode}" ] && continue
    data_id="$(task_id_from_episode "${task}" "${episode}")"
    src_dir="${root}/Hand_Poses/${episode}/1"

    if ! sync_demo "${task}" "${src_dir}" "${data_id}"; then
      printf '%s\n' "[SKIP] task=${task} episode=${episode} reason=missing_replay_outputs"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    attempted_count=$((attempted_count + 1))
    CUR_EARLY_TERMINATION="${EARLY_TERMINATION}"
    CUR_OBJ_POS_REW_SCALE="${OBJ_POS_REW_SCALE}"
    CUR_OBJ_ROT_REW_SCALE="${OBJ_ROT_REW_SCALE}"
    CUR_OBJECT_ROT_THRESHOLD="${OBJECT_ROT_THRESHOLD}"
    CUR_CONTACT_REW_SCALE="${CONTACT_REW_SCALE}"
    CUR_JOINT_REW_SCALE="${JOINT_REW_SCALE}"
    if [ "${task}" = "mustard" ]; then
      CUR_OBJECT_ROT_THRESHOLD="1.5"
      CUR_CONTACT_REW_SCALE="2.0"
      CUR_JOINT_REW_SCALE="0.005"
    elif [ "${task}" = "drawer" ]; then
      CUR_OBJECT_ROT_THRESHOLD="1.5"
      CUR_CONTACT_REW_SCALE="2.0"
      CUR_JOINT_REW_SCALE="0.005"
    elif [ "${task}" = "flower" ]; then
      CUR_OBJECT_ROT_THRESHOLD="1.5"
      CUR_CONTACT_REW_SCALE="2.0"
      CUR_JOINT_REW_SCALE="0.005"
    elif [ "${task}" = "hammer" ]; then
      CUR_EARLY_TERMINATION="0"
      CUR_OBJECT_ROT_THRESHOLD="1.5"
      CUR_CONTACT_REW_SCALE="2.0"
      CUR_JOINT_REW_SCALE="0.005"
    fi
    cur_video_args=("${VIDEO_ARGS[@]}")
    if [ "${SAVE_VIDEO}" = "1" ]; then
      if [ "${CUR_EARLY_TERMINATION}" = "0" ]; then
        video_suffix="no_early_termination"
      else
        video_suffix="early_termination"
      fi
      cur_video_args+=(--video-path "${VIDEO_DIR}/${task}_${data_id}_replay_${video_suffix}.mp4")
    fi
    replay_output="$({
      cd "${CODE_ROOT}" || exit 1
      PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" "examples/run_mjwp_replay.py" \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --robot-type "${ROBOT_TYPE}" \
        --embodiment-type "${EMBODIMENT_TYPE}" \
        --active-hand "${ACTIVE_HAND}" \
        --task "${task}" \
        --data-id "${data_id}" \
        --device "${REPLAY_DEVICE}" \
        --sim-dt "${SIM_DT}" \
        --ctrl-dt "${CTRL_DT}" \
        --horizon "${HORIZON}" \
        --knot-dt "${KNOT_DT}" \
        --pos-rew-scale "${CUR_OBJ_POS_REW_SCALE}" \
        --rot-rew-scale "${CUR_OBJ_ROT_REW_SCALE}" \
        --joint-rew-scale "${CUR_JOINT_REW_SCALE}" \
        --base-pos-rew-scale "${BASE_POS_REW_SCALE}" \
        --base-rot-rew-scale "${BASE_ROT_REW_SCALE}" \
        --vel-rew-scale "${VEL_REW_SCALE}" \
        --contact-rew-scale "${CUR_CONTACT_REW_SCALE}" \
        --object-pos-threshold "${OBJECT_POS_THRESHOLD}" \
        --object-rot-threshold "${CUR_OBJECT_ROT_THRESHOLD}" \
        --object-mass-scale "${OBJECT_MASS_SCALE}" \
        --early-termination "${CUR_EARLY_TERMINATION}" \
        "${cur_video_args[@]}" \
        "${VIEWER_ARGS[@]}"
    } 2>&1)"
    replay_rc=$?

    replay_line="$(printf '%s\n' "${replay_output}" | grep '^\[REPLAY\]' | tail -n 1)"
    if [ ${replay_rc} -ne 0 ] || [ -z "${replay_line}" ]; then
      failed_count=$((failed_count + 1))
      printf '%s\n' "[FAIL] task=${task} episode=${episode}"
      [ -n "${replay_output}" ] && printf '%s\n' "${replay_output}"
      continue
    fi

    completed_count=$((completed_count + 1))
    fps="$(printf '%s\n' "${replay_line}" | sed -n 's/.*fps=\([^ ]*\).*/\1/p')"
    reward_actual="$(printf '%s\n' "${replay_line}" | sed -n 's/.*reward_actual=\([^ ]*\).*/\1/p')"
    reward_max="$(printf '%s\n' "${replay_line}" | sed -n 's/.*reward_max=\([^ ]*\).*/\1/p')"
    steps_actual="$(printf '%s\n' "${replay_line}" | sed -n 's/.*steps_actual=\([^ ]*\).*/\1/p')"
    steps_max="$(printf '%s\n' "${replay_line}" | sed -n 's/.*steps_max=\([^ ]*\).*/\1/p')"
    success="$(printf '%s\n' "${replay_line}" | sed -n 's/.*success=\([^ ]*\).*/\1/p')"

    if [ "${success}" = "1" ]; then
      success_count=$((success_count + 1))
    fi

    printf '%s %s %s %s %s %s %s\n' "${task}_${data_id}" "${fps}" "${reward_actual}" "${reward_max}" "${steps_actual}" "${steps_max}" "${success}" >> "${tmp_summary}"
    reward_pct="$(awk -v a="${reward_actual}" -v b="${reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
    steps_pct="$(awk -v a="${steps_actual}" -v b="${steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
    printf '%s\n' "[SUMMARY] task=${task} episode=${episode} fps=${fps} reward=${reward_actual}/${reward_max} (${reward_pct}) steps=${steps_actual}/${steps_max} (${steps_pct}) success=${success}"
  done < <(collect_episodes "${task}" "${root}")
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

avg_reward_pct="$(awk -v a="${mean_reward_actual}" -v b="${mean_reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
avg_steps_pct="$(awk -v a="${mean_steps_actual}" -v b="${mean_steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
success_rate_pct="$(awk -v x="${success_rate}" 'BEGIN { if (x == x) printf "%.2f%%", 100.0 * x; else printf "nan" }')"

printf '%s\n' "[FINAL] demos=${completed_count} fps=${avg_fps} reward=${mean_reward_actual}/${mean_reward_max} (${avg_reward_pct}) steps=${mean_steps_actual}/${mean_steps_max} (${avg_steps_pct}) success_rate=${success_rate_pct}"
if [ ${skipped_count} -gt 0 ] || [ ${failed_count} -gt 0 ]; then
  printf '%s\n' "[FINAL] skipped=${skipped_count} failed=${failed_count} attempted=${attempted_count}"
fi
if [ ${failed_count} -gt 0 ]; then
  exit 1
fi
