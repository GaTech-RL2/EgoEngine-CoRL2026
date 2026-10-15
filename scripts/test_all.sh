#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -m unittest discover -s action/tests -v
PYTHONPATH="${PROJECT_DIR}/visual:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m unittest discover -s visual/tests -p 'test_aria_reference_preservation.py' -v
