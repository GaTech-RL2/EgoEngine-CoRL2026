#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST_FILE="${SCRIPT_DIR}/vis_taco_usable_demos.txt"

SRC_ROOT="/home/ycl/projects/datasets/egoengine/TACO"
DST_ROOT="/home/ycl/projects/workspace/temp/TACO"

REQUIRED_FILES=(
  "trajectory_kinematic.npz"
  "scene_eq.xml"
)

OPTIONAL_FILES=(
  "scene.xml"
  "task_info.json"
)

[[ -f "${LIST_FILE}" ]] || { echo "[ERROR] list file not found: ${LIST_FILE}"; exit 1; }
[[ -d "${SRC_ROOT}/Hand_Poses" ]] || { echo "[ERROR] source TACO Hand_Poses not found: ${SRC_ROOT}/Hand_Poses"; exit 1; }

mkdir -p "${DST_ROOT}"

copied_demos=0
skipped_demos=0
copied_files=0
copied_meshes=0

declare -A copied_mesh_paths

copy_meshes_from_scene() {
  local scene_file="$1"
  [[ -f "${scene_file}" ]] || return 0

  while IFS= read -r mesh_path; do
    [[ -n "${mesh_path}" ]] || continue
    [[ -f "${mesh_path}" ]] || continue
    [[ "${mesh_path}" == "${SRC_ROOT}/"* ]] || continue
    [[ -z "${copied_mesh_paths["${mesh_path}"]+x}" ]] || continue

    local rel_path="${mesh_path#${SRC_ROOT}/}"
    local dst_mesh="${DST_ROOT}/${rel_path}"
    mkdir -p "$(dirname "${dst_mesh}")"
    cp -f "${mesh_path}" "${dst_mesh}"

    copied_mesh_paths["${mesh_path}"]=1
    copied_meshes=$((copied_meshes + 1))
    copied_files=$((copied_files + 1))
    echo "[MESH] ${rel_path}"
  done < <(sed -n 's/.*file="\([^"]*\)".*/\1/p' "${scene_file}")
}

while IFS=$'\t' read -r triplet sequence; do
  [[ -n "${triplet}" ]] || continue
  [[ -n "${sequence}" ]] || continue

  src_dir="${SRC_ROOT}/Hand_Poses/${triplet}/${sequence}"
  dst_dir="${DST_ROOT}/Hand_Poses/${triplet}/${sequence}"

  missing_required=0
  for name in "${REQUIRED_FILES[@]}"; do
    [[ -f "${src_dir}/${name}" ]] || {
      echo "[SKIP] ${triplet}/${sequence}: missing ${src_dir}/${name}"
      skipped_demos=$((skipped_demos + 1))
      missing_required=1
      break
    }
  done
  [[ ${missing_required} -eq 0 ]] || continue

  mkdir -p "${dst_dir}"

  for name in "${REQUIRED_FILES[@]}"; do
    cp -f "${src_dir}/${name}" "${dst_dir}/${name}"
    copied_files=$((copied_files + 1))
  done

  for name in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "${src_dir}/${name}" ]]; then
      cp -f "${src_dir}/${name}" "${dst_dir}/${name}"
      copied_files=$((copied_files + 1))
    fi
  done

  copy_meshes_from_scene "${src_dir}/scene_eq.xml"
  copy_meshes_from_scene "${src_dir}/scene.xml"

  copied_demos=$((copied_demos + 1))
  echo "[COPIED] ${triplet}/${sequence}"
done < "${LIST_FILE}"

echo ""
echo "===== TACO Digital Twin Copy Summary ====="
echo "source root:   ${SRC_ROOT}"
echo "dest root:     ${DST_ROOT}"
echo "copied demos:  ${copied_demos}"
echo "skipped demos: ${skipped_demos}"
echo "copied meshes: ${copied_meshes}"
echo "copied files:  ${copied_files}"
