#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Prepared TACO inputs use the historical gigahand directory name.
# --window-size counts control steps. Two chunks are planned, one is committed.
exec "${PYTHON_BIN:-python}" "${CODE_ROOT}/examples/run_mjwp_mixed.py" \
  --device "${MIXED_DEVICE:-cuda:0}" \
  --methods replay,mpc,rl \
  --window-size 20 \
  --lookahead-chunks 2 \
  --task "${TASK_NAME:-brush_brush_bowl_20230927_027}" \
  --dataset-dir "${DATASET_DIR:-${CODE_ROOT}/example_datasets}" \
  --dataset-name gigahand \
  --robot-type xhand \
  --embodiment-type bimanual \
  --active-hand right \
  --data-id 0 \
  --output-name trajectory_mjwp_mixed.npz \
  "$@"
