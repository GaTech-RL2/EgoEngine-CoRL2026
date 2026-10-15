#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/franka_xhand}"
cd "${PROJECT_DIR}"
for CASE_NAME in mustard_1 mustard_2 mustard_3; do
  "${PYTHON_BIN}" -m egoengine render --case "${CASE_NAME}" \
    --robot-config examples/robots/franka_xhand.json --width 480 --output "${OUTPUT_DIR}"
  "${PYTHON_BIN}" -m egoengine composite --case "${CASE_NAME}" --output "${OUTPUT_DIR}" \
    --render-dir "${OUTPUT_DIR}/${CASE_NAME}/render"
  "${PYTHON_BIN}" -m egoengine export --case "${CASE_NAME}" --output "${OUTPUT_DIR}" \
    --video "${OUTPUT_DIR}/${CASE_NAME}/visual.mp4"
done
