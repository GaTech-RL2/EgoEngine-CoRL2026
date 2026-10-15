#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="/home/ycl/projects/workspace/egoengine"
DATA_ROOT="${DATA_ROOT:-/home/ycl/projects/datasets/mustard/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/ycl/miniconda3/envs/egoengine/bin/python}"

VIDEO_SOURCE="${VIDEO_SOURCE:-inpainted}"
OUTPUT="${OUTPUT:-${DATA_ROOT}/mustard_${VIDEO_SOURCE}_replay_dataset.hdf5}"
COMPRESSION="${COMPRESSION:-gzip}"

args=(
  "${CODE_ROOT}/postprocess/create_dataset.py"
  --data-root "${DATA_ROOT}"
  --video-source "${VIDEO_SOURCE}"
  --output "${OUTPUT}"
  --compression "${COMPRESSION}"
  --overwrite
)

if [[ -n "${VIDEO_ROOT:-}" ]]; then
  args+=(--video-root "${VIDEO_ROOT}")
fi

if [[ -n "${VIDEO_NAME:-}" ]]; then
  args+=(--video-name "${VIDEO_NAME}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi

if [[ -n "${EPISODES:-}" ]]; then
  # shellcheck disable=SC2206
  episode_args=(${EPISODES})
  args+=(--episodes "${episode_args[@]}")
fi

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] VIDEO_SOURCE=${VIDEO_SOURCE}"
echo "[INFO] OUTPUT=${OUTPUT}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" "${args[@]}"
