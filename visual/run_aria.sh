#!/usr/bin/env bash
set -Eeuo pipefail

# Aria task full pipeline:
# 1) extrinsic preprocess (optional)
# 2) retarget
# 3) replay segmentation
# 4) inpainting


# ========== 0) Setup ==========
DATA_ROOT="${DATA_ROOT:-/home/ycl/projects/datasets/mustard/data}"
CODE_ROOT="${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_NAME="${TASK_NAME:-}"
TARGET_POSE="${TARGET_POSE:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash run_aria.sh [--task-name {mustard|drawer|flower|hammer}] [DATA_ROOT]"
  echo ""
  echo "DATA_ROOT can be the task directory or its data/ directory."
  echo "TASK_NAME is optional; if omitted, retargeting infers it from DATA_ROOT."
  echo "Examples:"
  echo "  bash run_aria.sh /home/ycl/projects/datasets/drawer"
  echo "  bash run_aria.sh --task-name drawer /home/ycl/projects/datasets/drawer"
  echo "  bash run_aria.sh /home/ycl/projects/datasets/flower"
  echo "  TASK_NAME=drawer DATA_ROOT=/home/ycl/projects/datasets/drawer PYTHON_BIN=python bash run_aria.sh"
  exit 0
fi

DATA_ROOT_ARG_SET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name|--task_name)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires one of: mustard drawer flower hammer"; exit 1; }
      TASK_NAME="$2"
      shift 2
      ;;
    --task-name=*|--task_name=*)
      TASK_NAME="${1#*=}"
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        if [[ "${DATA_ROOT_ARG_SET}" == "1" ]]; then
          echo "[ERROR] Unexpected extra argument: $1"
          exit 1
        fi
        DATA_ROOT="$1"
        DATA_ROOT_ARG_SET=1
        shift
      done
      ;;
    -*)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
    *)
      if [[ "${DATA_ROOT_ARG_SET}" == "1" ]]; then
        echo "[ERROR] Unexpected extra argument: $1"
        exit 1
      fi
      DATA_ROOT="$1"
      DATA_ROOT_ARG_SET=1
      shift
      ;;
  esac
done

case "${TASK_NAME}" in
  ""|mustard|drawer|flower|hammer) ;;
  *) echo "[ERROR] TASK_NAME must be one of: mustard drawer flower hammer"; exit 1 ;;
esac

retarget_task_args=()
if [[ -n "${TARGET_POSE}" ]]; then
  retarget_task_args+=(--target-pose "${TARGET_POSE}")
fi
if [[ -n "${TASK_NAME}" ]]; then
  retarget_task_args+=(--task-name "${TASK_NAME}")
fi

if [[ ! -d "${DATA_ROOT}/Egocentric_RGB_Videos" && -d "${DATA_ROOT}/data/Egocentric_RGB_Videos" ]]; then
  DATA_ROOT="${DATA_ROOT}/data"
fi

HAND_ROOT="${DATA_ROOT}/Hand_Poses"
if [[ -d "${DATA_ROOT}/Egocentric_Camera_Parameters/Egocentric_Camera_Parameters" ]]; then
  CAM_ROOT="${DATA_ROOT}/Egocentric_Camera_Parameters/Egocentric_Camera_Parameters"
else
  CAM_ROOT="${DATA_ROOT}/Egocentric_Camera_Parameters"
fi
RGB_ROOT="${DATA_ROOT}/Egocentric_RGB_Videos"
REM_ROOT="${DATA_ROOT}/Removed"

TAG_REF="${CODE_ROOT}/retargeting/tag_in_world.txt"
CAM_NAME="cameraview"
RUN_EXTRINSIC_PREPROCESS="${RUN_EXTRINSIC_PREPROCESS:-0}"
OBJECT_CENTRIC="${OBJECT_CENTRIC:-0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

mapfile -t episodes < <(find "${HAND_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V)
[[ -d "${HAND_ROOT}" && -d "${CAM_ROOT}" && -d "${RGB_ROOT}" && -d "${REM_ROOT}" ]] || {
  echo "[ERROR] Aria task layout not found under: ${DATA_ROOT}"
  exit 1
}
if [[ "${RUN_EXTRINSIC_PREPROCESS}" == "1" ]]; then
  [[ -f "${TAG_REF}" ]] || { echo "[ERROR] tag reference file missing: ${TAG_REF}"; exit 1; }
fi
[[ ${#episodes[@]} -gt 0 ]] || { echo "[ERROR] No episodes found in ${HAND_ROOT}"; exit 1; }

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] CAM_ROOT=${CAM_ROOT}"
echo "[INFO] RUN_EXTRINSIC_PREPROCESS=${RUN_EXTRINSIC_PREPROCESS}"
echo "[INFO] OBJECT_CENTRIC=${OBJECT_CENTRIC}"
echo "[INFO] TASK_NAME=${TASK_NAME:-<inferred>}"
echo "[INFO] episodes=${#episodes[@]}"

# ========== 1) Extrinsic preprocess ==========
if [[ "${RUN_EXTRINSIC_PREPROCESS}" == "1" ]]; then
  echo ""
  echo "[STEP] extrinsic_pre_process_batch_single_tag (all episodes, session 1, single shared detector process)"
  echo "[PREPROCESS:ALL]"
  if ! "${PYTHON_BIN}" -m retargeting.extrinsic_pre_process_batch_single_tag \
      --data-root "${DATA_ROOT}" \
      --tag-ref "${TAG_REF}" \
      --episodes "${episodes[@]}" \
      --sessions 1; then
    echo "[WARN] preprocess batch failed; continuing to full pipeline"
  fi
fi

# ========== 2) Retarget + Replay + Inpainting ==========
total=${#episodes[@]}
success=0
retarget_fail=0
replay_fail=0
inpaint_fail=0

for ep in "${episodes[@]}"; do
  hand_dir="${HAND_ROOT}/${ep}/1"
  cam_dir="${CAM_ROOT}/${ep}/1"
  rgb_dir="${RGB_ROOT}/${ep}/1"
  rem_dir="${REM_ROOT}/${ep}/1"

  all_pose="${hand_dir}/all_pose.json"
  retarget_json="${hand_dir}/retarget.json"
  w2cam="${cam_dir}/upd_egocentric_frame_extrinsic.npy"

  src_video="${rgb_dir}/mask.mp4"
  mask_video="${rgb_dir}/mask_bin.mp4"
  bg_video="${rem_dir}/removed_w_mask_5.mp4"

  echo ""
  echo "=== [${ep}] ==="

  [[ -f "${all_pose}" ]] || { echo "[FAIL:${ep}] retarget input missing: ${all_pose}"; retarget_fail=$((retarget_fail + 1)); continue; }
  [[ -f "${w2cam}" ]] || { echo "[FAIL:${ep}] retarget input missing: ${w2cam}"; retarget_fail=$((retarget_fail + 1)); continue; }

  echo "[RETARGET:${ep}]"
  if ! "${PYTHON_BIN}" -m retargeting.show_aria_new retarget \
      --w2cam "${w2cam}" \
      --hand-json "${all_pose}" \
      --tag-world "${TAG_REF}" \
      --object-centric "${OBJECT_CENTRIC}" \
      "${retarget_task_args[@]}"; then
    echo "[FAIL:${ep}] retarget"
    retarget_fail=$((retarget_fail + 1))
    continue
  fi

  [[ -f "${retarget_json}" ]] || { echo "[FAIL:${ep}] retarget output missing: ${retarget_json}"; retarget_fail=$((retarget_fail + 1)); continue; }

  echo "[REPLAY:${ep}]"
  if ! "${PYTHON_BIN}" -m rl_tracking_dev.replay_trajectory_seg_aria \
      --data_file "${retarget_json}" \
      --occlusion \
      --cam_name "${CAM_NAME}"; then
    echo "[FAIL:${ep}] replay"
    replay_fail=$((replay_fail + 1))
    continue
  fi

  [[ -f "${src_video}" && -f "${mask_video}" && -f "${bg_video}" ]] || {
    echo "[FAIL:${ep}] missing inputs for inpainting"
    echo "  src:  ${src_video}"
    echo "  mask: ${mask_video}"
    echo "  bg:   ${bg_video}"
    inpaint_fail=$((inpaint_fail + 1))
    continue
  }

  echo "[INPAINT:${ep}]"
  if ! "${PYTHON_BIN}" -m tests.inpainting \
      --rgb-dir "${rgb_dir}" \
      --rem-dir "${rem_dir}" \
      --src-name "mask.mp4" \
      --mask-name "mask_bin.mp4" \
      --bg-name "removed_w_mask_5.mp4" \
      --out-name "inpainted_video_egoengine.mp4" \
      --edge-blur 1 \
      --edge-erode 1 \
      --rim-blend 0.05 \
      --rim-noise 0.5 \
      --fg-noise 0.1 \
      --clone-mode mixed; then
    echo "[FAIL:${ep}] inpainting"
    inpaint_fail=$((inpaint_fail + 1))
    continue
  fi
  success=$((success + 1))
  echo "[DONE:${ep}]"
done


# ========== Summary ==========
echo ""
echo "===== Aria task pipeline summary ====="
echo "total episodes:          ${total}"
echo "all steps successful:    ${success}"
echo "retarget failed:         ${retarget_fail}"
echo "replay failed:           ${replay_fail}"
echo "inpainting failed:       ${inpaint_fail}"
