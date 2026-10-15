#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m egoengine list
"${PYTHON_BIN}" -m egoengine validate
"${PYTHON_BIN}" -m unittest discover -s tests -v
for CASE_NAME in mustard_1 mustard_2 mustard_3; do
  "${PYTHON_BIN}" -m egoengine demo --case "${CASE_NAME}"
done
"${PYTHON_BIN}" -m egoengine media --output outputs/cpu_media
