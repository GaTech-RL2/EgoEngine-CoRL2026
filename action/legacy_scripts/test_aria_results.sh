#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS_CSV="${METHODS:-replay,mpc}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-5}"
TASKS_CSV="${TASKS_CSV:-mustard,drawer,flower,hammer}"

ARIA_ROOT_BASE="${ARIA_ROOT_BASE:-/home/ycl/projects/datasets}"
MUSTARD_ROOT="${MUSTARD_ROOT:-${ARIA_ROOT_BASE}/mustard/data}"
DRAWER_ROOT="${DRAWER_ROOT:-${ARIA_ROOT_BASE}/drawer/data}"
FLOWER_ROOT="${FLOWER_ROOT:-${ARIA_ROOT_BASE}/flower/data}"
HAMMER_ROOT="${HAMMER_ROOT:-${ARIA_ROOT_BASE}/hammer/data}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/outputs/aria_result_tests/${RUN_STAMP}}"
DETAIL_TSV="${RESULT_DIR}/details.tsv"
SUMMARY_TSV="${RESULT_DIR}/summary.tsv"
RUNTIME_TSV="${RESULT_DIR}/runtime.tsv"

mkdir -p "${RESULT_DIR}"
printf 'method\ttask\tepisode\tfps\tTPS\tReward\tSTR\tSR\trunner_success\treward_actual\treward_max\tsteps_actual\tsteps_max\tlog\n' > "${DETAIL_TSV}"
printf 'method\ttask\telapsed_sec\n' > "${RUNTIME_TSV}"

usage() {
  cat <<'EOF'
Usage:
  bash test_aria_results.sh [options]

Options:
  --methods replay,mpc,rl,egoengine
  --samples-per-task N
  --tasks mustard,drawer,flower,hammer

Environment:
  METHODS, TASKS_CSV, SAMPLES_PER_TASK, RESULT_DIR, PYTHON_BIN, REPLAY_DEVICE, MPC_DEVICE,
  MUSTARD_ROOT, DRAWER_ROOT, FLOWER_ROOT, HAMMER_ROOT

Default: METHODS=replay,mpc and SAMPLES_PER_TASK=5.
EgoEngine is intentionally marked skipped because it is not completed yet.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --methods)
      METHODS_CSV="$2"
      shift 2
      ;;
    --samples-per-task)
      SAMPLES_PER_TASK="$2"
      shift 2
      ;;
    --tasks)
      TASKS_CSV="$2"
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

task_root() {
  case "$1" in
    mustard) printf '%s\n' "${MUSTARD_ROOT}" ;;
    drawer) printf '%s\n' "${DRAWER_ROOT}" ;;
    flower) printf '%s\n' "${FLOWER_ROOT}" ;;
    hammer) printf '%s\n' "${HAMMER_ROOT}" ;;
    *) return 1 ;;
  esac
}

script_for_method() {
  case "$1" in
    replay) prepare_replay_script ;;
    mpc) prepare_mpc_script ;;
    rl) printf '%s\n' "${SCRIPT_DIR}/run_aria_rl.sh" ;;
    egoengine) return 1 ;;
    *) return 2 ;;
  esac
}

prepare_replay_script() {
  local src="${SCRIPT_DIR}/run_aria_replay.sh"
  local tmp="${RESULT_DIR}/run_aria_replay_object_metric.sh"
  if [ ! -f "${tmp}" ]; then
    sed \
      -e "s|^SCRIPT_DIR=.*|SCRIPT_DIR=\"${SCRIPT_DIR}\"|" \
      -e 's/^JOINT_REW_SCALE=.*/JOINT_REW_SCALE="${JOINT_REW_SCALE:-0.0}"/' \
      -e 's/^BASE_POS_REW_SCALE=.*/BASE_POS_REW_SCALE="${BASE_POS_REW_SCALE:-0.0}"/' \
      -e 's/^BASE_ROT_REW_SCALE=.*/BASE_ROT_REW_SCALE="${BASE_ROT_REW_SCALE:-0.0}"/' \
      -e 's/^VEL_REW_SCALE=.*/VEL_REW_SCALE="${VEL_REW_SCALE:-0.0}"/' \
      -e 's/^CONTACT_REW_SCALE=.*/CONTACT_REW_SCALE="${CONTACT_REW_SCALE:-0.0}"/' \
      -e 's/CUR_EARLY_TERMINATION="0"/CUR_EARLY_TERMINATION="${EARLY_TERMINATION}"/' \
      -e 's/CUR_CONTACT_REW_SCALE="2.0"/CUR_CONTACT_REW_SCALE="0.0"/g' \
      -e 's/CUR_JOINT_REW_SCALE="0.005"/CUR_JOINT_REW_SCALE="0.0"/g' \
      "${src}" > "${tmp}"
    chmod +x "${tmp}"
  fi
  printf '%s\n' "${tmp}"
}

prepare_mpc_script() {
  local src="${SCRIPT_DIR}/run_aria_mpc.sh"
  local tmp="${RESULT_DIR}/run_aria_mpc_same_early_termination.sh"
  if [ ! -f "${tmp}" ]; then
    sed \
      -e "s|^SCRIPT_DIR=.*|SCRIPT_DIR=\"${SCRIPT_DIR}\"|" \
      -e 's/^CTRL_DT=.*/CTRL_DT="${CTRL_DT:-0.05}"/' \
      -e 's/^HORIZON=.*/HORIZON="${HORIZON:-0.5}"/' \
      -e 's/^JOINT_REW_SCALE=.*/JOINT_REW_SCALE="0.0"/' \
      -e 's/^BASE_POS_REW_SCALE=.*/BASE_POS_REW_SCALE="0.0"/' \
      -e 's/^BASE_ROT_REW_SCALE=.*/BASE_ROT_REW_SCALE="0.0"/' \
      -e 's/^VEL_REW_SCALE=.*/VEL_REW_SCALE="0.0"/' \
      -e 's/^CONTACT_REW_SCALE=.*/CONTACT_REW_SCALE="0.0"/' \
      -e 's/CUR_EARLY_TERMINATION="0"/CUR_EARLY_TERMINATION="${EARLY_TERMINATION}"/' \
      -e 's/CUR_CONTACT_REW_SCALE="2.0"/CUR_CONTACT_REW_SCALE="0.0"/g' \
      -e 's/CUR_JOINT_REW_SCALE="0.005"/CUR_JOINT_REW_SCALE="0.0"/g' \
      "${src}" > "${tmp}"
    chmod +x "${tmp}"
  fi
  printf '%s\n' "${tmp}"
}

collect_episodes() {
  local task="$1"
  local root="$2"
  find "${root}/Hand_Poses" -mindepth 1 -maxdepth 1 -type d -name "${task}_*" -printf '%f\n' \
    | sort -V \
    | awk -v n="${SAMPLES_PER_TASK}" 'NR <= n { print }'
}

parse_summary_lines() {
  local method="$1"
  local task="$2"
  local log_path="$3"
  local line episode fps reward_actual reward_max steps_actual steps_max runner_success success tps reward_norm str

  while IFS= read -r line; do
    episode="$(printf '%s\n' "${line}" | sed -En 's/.*episode=([^ ]*).*/\1/p')"
    [ -n "${episode}" ] || episode="unknown"
    fps="$(printf '%s\n' "${line}" | sed -En 's/.*fps=([^ ]*).*/\1/p')"
    reward_actual="$(printf '%s\n' "${line}" | sed -En 's/.*reward=([^/ ]*)\/([^ ]*).*/\1/p')"
    reward_max="$(printf '%s\n' "${line}" | sed -En 's/.*reward=([^/ ]*)\/([^ ]*).*/\2/p')"
    steps_actual="$(printf '%s\n' "${line}" | sed -En 's/.*steps=([^/ ]*)\/([^ ]*).*/\1/p')"
    steps_max="$(printf '%s\n' "${line}" | sed -En 's/.*steps=([^/ ]*)\/([^ ]*).*/\2/p')"
    runner_success="$(printf '%s\n' "${line}" | sed -En 's/.*success=([^ ]*).*/\1/p')"

    tps="$(awk -v fps="${fps}" 'BEGIN { if (fps > 0) printf "%.6f", 1.0 / fps; else printf "nan" }')"
    reward_norm="$(awk -v a="${reward_actual}" -v b="${reward_max}" 'BEGIN { if (b > 0) printf "%.6f", a / b; else printf "nan" }')"
    str="$(awk -v a="${steps_actual}" -v b="${steps_max}" 'BEGIN { if (b > 0) printf "%.6f", a / b; else printf "nan" }')"
    success="${runner_success}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${method}" "${task}" "${episode}" "${fps}" "${tps}" "${reward_norm}" "${str}" "${success}" "${runner_success}" \
      "${reward_actual}" "${reward_max}" "${steps_actual}" "${steps_max}" "${log_path}" >> "${DETAIL_TSV}"
  done < <(grep '^\[SUMMARY\]' "${log_path}" || true)
}

run_method_task() {
  local method="$1"
  local task="$2"
  local root script_path log_path rc
  local start_time end_time elapsed_sec
  local episodes=()

  root="$(task_root "${task}")" || return 1
  if [ ! -d "${root}/Hand_Poses" ]; then
    printf '%s\n' "[SKIP] method=${method} task=${task} reason=missing ${root}/Hand_Poses"
    return 0
  fi

  while IFS= read -r episode; do
    [ -n "${episode}" ] && episodes+=("${episode}")
  done < <(collect_episodes "${task}" "${root}")

  if [ ${#episodes[@]} -eq 0 ]; then
    printf '%s\n' "[SKIP] method=${method} task=${task} reason=no_episodes"
    return 0
  fi

  script_path="$(script_for_method "${method}")"
  if [ ! -f "${script_path}" ]; then
    printf '%s\n' "[SKIP] method=${method} task=${task} reason=missing_script"
    return 0
  fi

  log_path="${RESULT_DIR}/${method}_${task}.log"
  printf '%s\n' "[RUN] method=${method} task=${task} samples=${#episodes[@]} episodes=${episodes[*]}"

  start_time="$(date +%s.%N)"
  (
    cd "${SCRIPT_DIR}" || exit 1
    SAVE_VIDEO=0 \
    SHOW_VIEWER=0 \
    RL_EARLY_STOP_ON_SUCCESS=1 \
    bash "${script_path}" --task-name "${task}" --episodes "${episodes[@]}"
  ) 2>&1 | tee "${log_path}"
  rc=${PIPESTATUS[0]}
  end_time="$(date +%s.%N)"
  elapsed_sec="$(awk -v a="${start_time}" -v b="${end_time}" 'BEGIN { printf "%.6f", b - a }')"
  printf '%s\t%s\t%s\n' "${method}" "${task}" "${elapsed_sec}" >> "${RUNTIME_TSV}"

  parse_summary_lines "${method}" "${task}" "${log_path}"

  if [ ${rc} -ne 0 ]; then
    printf '%s\n' "[WARN] method=${method} task=${task} exited=${rc}; parsed any completed summaries from ${log_path}"
  fi
  return 0
}

print_summary() {
  printf 'method\tdemos\tSR\tSTR\tReward\tTPS\tDPH\tavg_fps\n' > "${SUMMARY_TSV}"
  awk -F '\t' '
    FILENAME == ARGV[1] && FNR > 1 {
      elapsed[$1] += $3
      next
    }
    FILENAME == ARGV[2] && FNR > 1 {
      method = $1
      n[method] += 1
      fps[method] += $4
      tps[method] += $5
      reward[method] += $6
      str[method] += $7
      sr[method] += $8
      next
    }
    END {
      for (method in n) {
        if (n[method] > 0) {
          dph = "nan"
          if (elapsed[method] > 0) {
            dph = sprintf("%.6f", 3600.0 * n[method] / elapsed[method])
          }
          printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%s\t%.6f\n",
            method, n[method], sr[method] / n[method], str[method] / n[method],
            reward[method] / n[method], tps[method] / n[method], dph, fps[method] / n[method]
        }
      }
    }
  ' "${RUNTIME_TSV}" "${DETAIL_TSV}" | sort >> "${SUMMARY_TSV}"

  printf '%s\n' ""
  printf '%s\n' "=== Aria aggregate metrics ==="
  column -t -s $'\t' "${SUMMARY_TSV}"
  printf '%s\n' ""
  printf '%s\n' "=== Per-sample metrics ==="
  column -t -s $'\t' "${DETAIL_TSV}"
  printf '%s\n' ""
  printf '%s\n' "[RESULT_DIR] ${RESULT_DIR}"
}

IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

for method in "${METHODS[@]}"; do
  case "${method}" in
    replay|mpc|rl)
      for task in "${TASKS[@]}"; do
        run_method_task "${method}" "${task}"
      done
      ;;
    egoengine)
      printf '%s\n' "[SKIP] method=egoengine reason=not_completed_yet"
      ;;
    *)
      printf '%s\n' "[WARN] unknown method=${method}; skipping"
      ;;
  esac
done

print_summary
