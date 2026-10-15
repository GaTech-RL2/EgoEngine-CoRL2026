#!/usr/bin/env bash

DATASET_ROOT="/home/ycl/projects/datasets/egoengine"
CODE_ROOT="/home/ycl/projects/workspace/egoengine"

run_cassie_batch() {
  export MUJOCO_GL="${MUJOCO_GL:-egl}"

  local py_bin="${PYTHON_BIN:-python}"
  local data_root="${CASSIE_DATA_ROOT:-${DATASET_ROOT}/Aria/human/data}"
  if [[ ! -d "${data_root}/Hand_Poses" && -d "${data_root}/data/Hand_Poses" ]]; then
    data_root="${data_root}/data"
  fi

  local hand_root="${data_root}/Hand_Poses"
  local cam_root="${data_root}/Egocentric_Camera_Parameters"
  local rgb_root="${data_root}/Egocentric_RGB_Videos"
  local rem_root="${data_root}/Removed"
  local bg_name="${CASSIE_BG_NAME:-removed_w_mask_5.mp4}"

  [[ -d "${hand_root}" && -d "${cam_root}" && -d "${rgb_root}" && -d "${rem_root}" ]] || {
    echo "[ERROR] Cassie layout not found under: ${data_root}"
    return 1
  }

  local episodes=()
  if [[ $# -gt 0 ]]; then
    episodes=("$@")
  else
    mapfile -t episodes < <(find "${hand_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V)
  fi
  [[ ${#episodes[@]} -gt 0 ]] || { echo "[ERROR] No Cassie episodes found in ${hand_root}"; return 1; }

  echo "[INFO] CASSIE_DATA_ROOT=${data_root}"
  echo "[INFO] episodes=${#episodes[@]}"

  local total=${#episodes[@]}
  local success=0
  local retarget_fail=0
  local replay_fail=0
  local inpaint_fail=0

  for ep in "${episodes[@]}"; do
    local hand_dir="${hand_root}/${ep}/1"
    local cam_dir="${cam_root}/${ep}/1"
    local rgb_dir="${rgb_root}/${ep}/1"
    local rem_dir="${rem_root}/${ep}/1"
    if [[ ! -f "${rem_dir}/${bg_name}" && "${ep}" == *_ ]]; then
      local ep_trimmed="${ep%_}"
      if [[ -f "${rem_root}/${ep_trimmed}/1/${bg_name}" ]]; then
        rem_dir="${rem_root}/${ep_trimmed}/1"
      fi
    fi

    local all_pose="${hand_dir}/all_pose.json"
    local retarget_json="${hand_dir}/retarget.json"
    local w2cam="${cam_dir}/egocentric_frame_extrinsic.npy"
    local intrinsic="${cam_dir}/egocentric_intrinsic.txt"

    echo ""
    echo "=== [${ep}] ==="

    [[ -f "${all_pose}" ]] || { echo "[FAIL:${ep}] missing ${all_pose}"; retarget_fail=$((retarget_fail + 1)); continue; }
    [[ -f "${w2cam}" ]] || { echo "[FAIL:${ep}] missing ${w2cam}"; retarget_fail=$((retarget_fail + 1)); continue; }
    [[ -f "${intrinsic}" ]] || { echo "[FAIL:${ep}] missing ${intrinsic}"; retarget_fail=$((retarget_fail + 1)); continue; }

    echo "[RETARGET:${ep}]"
    if ! "${py_bin}" -m retargeting.show_aria_new_wo_tag retarget \
        --w2cam "${w2cam}" \
        --hand-json "${all_pose}" \
        --out-json "${retarget_json}"; then
      echo "[FAIL:${ep}] retarget"
      retarget_fail=$((retarget_fail + 1))
      continue
    fi

    [[ -f "${retarget_json}" ]] || { echo "[FAIL:${ep}] missing ${retarget_json}"; retarget_fail=$((retarget_fail + 1)); continue; }

    echo "[REPLAY:${ep}]"
    if ! "${py_bin}" -m rl_tracking_dev.replay_trajectory_seg_aria \
        --data_file "${retarget_json}" \
        --cam_name cameraview \
        --flip; then
      echo "[FAIL:${ep}] replay"
      replay_fail=$((replay_fail + 1))
      continue
    fi

    [[ -f "${rgb_dir}/mask.mp4" && -f "${rgb_dir}/mask_bin.mp4" && -f "${rem_dir}/${bg_name}" ]] || {
      echo "[FAIL:${ep}] missing inpainting inputs"
      echo "  src:  ${rgb_dir}/mask.mp4"
      echo "  mask: ${rgb_dir}/mask_bin.mp4"
      echo "  bg:   ${rem_dir}/${bg_name}"
      inpaint_fail=$((inpaint_fail + 1))
      continue
    }

    echo "[INPAINT:${ep}]"
    if ! "${py_bin}" -m tests.inpainting \
        --rgb-dir "${rgb_dir}" \
        --rem-dir "${rem_dir}" \
        --src-name mask.mp4 \
        --mask-name mask_bin.mp4 \
        --bg-name "${bg_name}" \
        --out-name inpainted_video.mp4 \
        --clone-mode mixed; then
      echo "[FAIL:${ep}] inpainting"
      inpaint_fail=$((inpaint_fail + 1))
      continue
    fi

    success=$((success + 1))
    echo "[DONE:${ep}]"
  done

  echo ""
  echo "----- Cassie pipeline summary -----"
  echo "total episodes:       ${total}"
  echo "all steps successful: ${success}"
  echo "retarget failed:      ${retarget_fail}"
  echo "replay failed:        ${replay_fail}"
  echo "inpainting failed:    ${inpaint_fail}"
}

if [[ "${1:-}" == "cassie" ]]; then
  shift
  run_cassie_batch "$@"
  exit $?
fi

# TACO dataset retargeting
export MUJOCO_GL="egl"
TACO_USABLE_LIST="${CODE_ROOT}/taco_usable_demos.txt"
CONTACT_ATTACH_THRESH_M="0.03"  # 3 cm fingertip-to-object mesh snap threshold



# ========== TACO: existing retarget -> replay_seg -> inpainting ==========
total=0
ran=0
skip_no_retarget=0
skip_no_removed=0
skip_no_bg=0
skip_replay_failed=0
skip_no_replay_video=0
skip_no_replay_mask=0
skip_inpaint_failed=0

while IFS=$'\t' read -r triplet sequence; do
  [ -z "$triplet" ] && continue
  [ -z "$sequence" ] && continue
  total=$((total+1))

  hand_retarget_file="${DATASET_ROOT}/TACO/Hand_Poses/${triplet}/${sequence}/retarget.json"
  rgb_dir="${DATASET_ROOT}/TACO/Egocentric_RGB_Videos/${triplet}/${sequence}"
  rem_dir="${DATASET_ROOT}/TACO/Removed/${triplet}/${sequence}"
  bg_name="removed_w_mask_15.mp4"

  [ -f "${hand_retarget_file}" ] || { skip_no_retarget=$((skip_no_retarget+1)); echo "SKIP ${triplet}/${sequence}: missing ${hand_retarget_file}"; continue; }
  [ -d "${rem_dir}" ] || { skip_no_removed=$((skip_no_removed+1)); echo "SKIP ${triplet}/${sequence}: missing ${rem_dir}"; continue; }
  [ -f "${rem_dir}/${bg_name}" ] || { skip_no_bg=$((skip_no_bg+1)); echo "SKIP ${triplet}/${sequence}: missing ${rem_dir}/${bg_name}"; continue; }

  echo "=== Replaying TACO ${triplet} - ${sequence} ==="
  python -m rl_tracking_dev.replay_trajectory_seg \
    --data_file "${hand_retarget_file}" \
    --cam_name cameraview || { skip_replay_failed=$((skip_replay_failed+1)); echo "SKIP ${triplet}/${sequence}: replay failed"; continue; }

  [ -f "${rgb_dir}/video.mp4" ] || { skip_no_replay_video=$((skip_no_replay_video+1)); echo "SKIP ${triplet}/${sequence}: missing ${rgb_dir}/video.mp4"; continue; }
  [ -f "${rgb_dir}/mask_bin.mp4" ] || { skip_no_replay_mask=$((skip_no_replay_mask+1)); echo "SKIP ${triplet}/${sequence}: missing ${rgb_dir}/mask_bin.mp4"; continue; }

  echo "=== Inpainting TACO ${triplet} - ${sequence} ==="
  python -m tests.inpainting \
    --rgb-dir "${rgb_dir}" \
    --rem-dir "${rem_dir}" \
    --src-name video.mp4 \
    --mask-name mask_bin.mp4 \
    --bg-name "${bg_name}" \
    --out-name inpainted_video.mp4 \
    --clone-mode mixed || { skip_inpaint_failed=$((skip_inpaint_failed+1)); echo "ERROR: Failed to inpaint ${triplet}/${sequence}"; continue; }

  ran=$((ran+1))
done < "${TACO_USABLE_LIST}"

echo "----- TACO replay + inpainting summary -----"
echo "total demos checked: ${total}"
echo "ran successfully: ${ran}"
echo "skipped missing retarget: ${skip_no_retarget}"
echo "skipped missing removed dir: ${skip_no_removed}"
echo "skipped missing removed_w_mask_15.mp4: ${skip_no_bg}"
echo "skipped replay failure: ${skip_replay_failed}"
echo "skipped missing replay video.mp4: ${skip_no_replay_video}"
echo "skipped missing replay mask_bin.mp4: ${skip_no_replay_mask}"
echo "skipped inpainting failure: ${skip_inpaint_failed}"



while IFS=$'\t' read -r triplet sequence; do
  [ -z "$triplet" ] && continue
  [ -z "$sequence" ] && continue
  echo "=== Retargeting TACO $triplet - $sequence ==="
  python -m retargeting.show_human_new retarget \
    --base-dir "${DATASET_ROOT}/TACO" \
    --triplet "$triplet" \
    --sequence "$sequence" \
    --object-centric "${CONTACT_ATTACH_THRESH_M}" || echo "ERROR: Failed to retarget $triplet - $sequence"
done < "${TACO_USABLE_LIST}"

# ========== 2) TACO Segment + Blending (occlusion_on only) ==========
while IFS=$'\t' read -r triplet sequence; do
  [ -z "$triplet" ] && continue
  [ -z "$sequence" ] && continue
  echo "=== Replaying TACO $triplet - $sequence (occlusion_on) ==="
  python -m rl_tracking_dev.replay_trajectory_seg \
    --data_file "${DATASET_ROOT}/TACO/Hand_Poses/${triplet}/${sequence}/retarget.json" \
    --cam_name cameraview \
    --occlusion || echo "ERROR: Failed to replay (occlusion_on) $triplet - $sequence"

  echo "=== Inpainting TACO $triplet - $sequence (occlusion_on) ==="
  python -m tests.inpainting \
    --rgb-dir "${DATASET_ROOT}/TACO/Egocentric_RGB_Videos/${triplet}/${sequence}/occlusion_on" \
    --rem-dir "${DATASET_ROOT}/TACO/Removed/${triplet}/${sequence}" \
    --src-name video.mp4 \
    --mask-name mask_bin.mp4 \
    --out-name inpainted_video.mp4 \
    --clone-mode mixed || echo "ERROR: Failed to inpaint (occlusion_on) $triplet - $sequence"
done < "${TACO_USABLE_LIST}"
exit


# to get the camera pose for with april tags
python -m retargeting.extrinsic_pre_process_batch \
  --data-root "${DATASET_ROOT}/Aria/mustard/data" \
  --tag-ref "${CODE_ROOT}/retargeting/tag_in_world.txt"


# ========== 1) Retarget ==========
for ep in \
  Cassie-bartending Cassie-bartending2 Cassie-blinds Cassie-boardgame_ Cassie-cat2 Cassie-cat_food \
  Cassie-cooking2 Cassie-door Cassie-earbuds Cassie-eyedrops_ Cassie-flower Cassie-fridge \
  Cassie-laptop Cassie-microwave_ Cassie-oven Cassie-Rowan-Allen Cassie-subway Cassie-yogurt
do
  echo "=== Retargeting human $ep ==="
  python -m retargeting.show_aria_new_wo_tag retarget \
    --w2cam "${DATASET_ROOT}/Aria/human/data/Egocentric_Camera_Parameters/${ep}/1/egocentric_frame_extrinsic.npy" \
    --hand-json "${DATASET_ROOT}/Aria/human/data/Hand_Poses/${ep}/1/all_pose.json" \
    --out-json "${DATASET_ROOT}/Aria/human/data/Hand_Poses/${ep}/1/retarget.json"
done


# ========== 2) Segment (replay & render) — with --flip ==========
for ep in \
  Cassie-bartending Cassie-bartending2 Cassie-blinds Cassie-boardgame_ Cassie-cat2 Cassie-cat_food \
  Cassie-cooking2 Cassie-door Cassie-earbuds Cassie-eyedrops_ Cassie-flower Cassie-fridge \
  Cassie-laptop Cassie-microwave_ Cassie-oven Cassie-Rowan-Allen Cassie-subway Cassie-yogurt
do
  echo "=== Replaying human $ep ==="
  python -m rl_tracking_dev.replay_trajectory_seg_aria \
    --data_file "${DATASET_ROOT}/Aria/human/data/Hand_Poses/${ep}/1/retarget.json" \
    --cam_name cameraview \
    --flip
done


# ========== 3) Blending (inpainting) ==========
for ep in \
  Cassie-bartending Cassie-bartending2 Cassie-blinds Cassie-boardgame_ Cassie-cat2 Cassie-cat_food \
  Cassie-cooking2 Cassie-door Cassie-earbuds Cassie-eyedrops_ Cassie-flower Cassie-fridge \
  Cassie-laptop Cassie-microwave_ Cassie-oven Cassie-Rowan-Allen Cassie-subway Cassie-yogurt
do
  echo "=== Inpainting human $ep ==="
  python -m tests.inpainting \
    --rgb-dir "${DATASET_ROOT}/Aria/human/data/Egocentric_RGB_Videos/${ep}/1" \
    --rem-dir "${DATASET_ROOT}/Aria/human/data/Removed/${ep}/1" \
    --src-name mask.mp4 \
    --mask-name mask_bin.mp4 \
    --bg-name removed_w_mask_5.mp4 \
    --out-name inpainted_video.mp4 \
    --clone-mode mixed
done



# -------- Process Aria Dataset -------- #


# -------- drawer (801–850) --------
for id in $(seq 801 850); do
  echo "=== Processing drawer $id ==="
  python -m retargeting.show_aria_new retarget \
    --w2cam "${DATASET_ROOT}/Aria/drawer/data/Egocentric_Camera_Parameters/${id}/1/upd_egocentric_frame_extrinsic.npy" \
    --hand-json "${DATASET_ROOT}/Aria/drawer/data/Hand_Poses/${id}/1/all_pose.json" \
    --tag-world "${CODE_ROOT}/retargeting/tag_in_world.txt"
done

# -------- mustard (901–950) --------
for id in $(seq 901 950); do
  echo "=== Processing mustard $id ==="
  python -m retargeting.show_aria_new retarget \
    --w2cam "${DATASET_ROOT}/Aria/mustard/data/Egocentric_Camera_Parameters/${id}/1/upd_egocentric_frame_extrinsic.npy" \
    --hand-json "${DATASET_ROOT}/Aria/mustard/data/Hand_Poses/${id}/1/all_pose.json" \
    --tag-world "${CODE_ROOT}/retargeting/tag_in_world.txt"
done


# Segment
# -------- drawer (801–850) --------
for id in $(seq 801 850); do
  echo "=== Replaying drawer $id ==="
  python -m rl_tracking_dev.replay_trajectory_seg_aria \
    --data_file "${DATASET_ROOT}/Aria/drawer/data/Hand_Poses/${id}/1/retarget.json" \
    --cam_name cameraview \
    --occlusion
done


# -------- mustard (901–950) --------
for id in $(seq 901 950); do
  echo "=== Replaying mustard $id ==="
  python -m rl_tracking_dev.replay_trajectory_seg_aria \
    --data_file "${DATASET_ROOT}/Aria/mustard/data/Hand_Poses/${id}/1/retarget.json" \
    --cam_name cameraview \
    --occlusion
done

# blending
# -------- drawer (801–850) --------
for id in $(seq 801 850); do
  echo "=== Inpainting drawer $id ==="
  python -m tests.inpainting \
    --rgb-dir "${DATASET_ROOT}/Aria/drawer/data/Egocentric_RGB_Videos/${id}/1" \
    --rem-dir "${DATASET_ROOT}/Aria/drawer/data/Removed/${id}/1" \
    --src-name mask.mp4 \
    --mask-name mask_bin.mp4 \
    --bg-name removed_w_mask_5.mp4 \
    --out-name inpainted_video.mp4 \
    --clone-mode mixed
done

# -------- mustard (901–950) --------
for id in $(seq 901 950); do
  echo "=== Inpainting mustard $id ==="
  python -m tests.inpainting \
    --rgb-dir "${DATASET_ROOT}/Aria/mustard/data/Egocentric_RGB_Videos/${id}/1" \
    --rem-dir "${DATASET_ROOT}/Aria/mustard/data/Removed/${id}/1" \
    --src-name mask.mp4 \
    --mask-name mask_bin.mp4 \
    --bg-name removed_w_mask_5.mp4 \
    --out-name inpainted_video.mp4 \
    --clone-mode mixed
done
