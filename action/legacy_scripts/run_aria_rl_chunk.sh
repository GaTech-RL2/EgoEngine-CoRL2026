#!/usr/bin/env bash
set -u
set -o pipefail

ARIA_ROOT_BASE="/home/ycl/projects/datasets"
CODE_ROOT="/home/ycl/projects/workspace/spider"
TASK_NAME="${TASK_NAME:-}"
HZ="${HZ:-20}"

MUSTARD_ROOT="${MUSTARD_ROOT:-${ARIA_ROOT_BASE}/mustard/data}"
DRAWER_ROOT="${DRAWER_ROOT:-${ARIA_ROOT_BASE}/drawer/data}"
FLOWER_ROOT="${FLOWER_ROOT:-${ARIA_ROOT_BASE}/flower/data}"
HAMMER_ROOT="${HAMMER_ROOT:-${ARIA_ROOT_BASE}/hammer/data}"

# RL config (same as run_taco_rl.sh style)
DATASET_DIR="/home/ycl/projects/workspace/spider/example_datasets"
DATASET_NAME="aria"
ROBOT_TYPE="xhand"
EMBODIMENT_TYPE="right"
DATA_ID=0

SOURCE_EMBODIMENT="right"
HAND_MODES=("right")

RL_DEVICE="cuda:0"
RL_NUM_ENVS="4096"
RL_EVAL_NUM_ENVS="256"
RL_FUTURE_CTRL_STEPS="0"
RL_EPISODE_CTRL_STEPS="64"
RL_NUM_STEPS_PER_ENV="16"
RL_LEARNING_ITERS="${RL_LEARNING_ITERS:-300}"
RL_EARLY_STOP_ON_SUCCESS="${RL_EARLY_STOP_ON_SUCCESS:-1}"
RL_CKPT_INTERVAL="500"
RL_VIZ_INTERVAL="40"
RL_ROLLOUT_VIDEO_CTRL_STEPS="0"

WANDB_MEDIA_ONLY="1"
WANDB_PROJECT="egoengine-spider-rl"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
RL_EXPERIMENT_SUFFIX="${RL_EXPERIMENT_SUFFIX:-}"

RL_CHUNK_SIZE="${RL_CHUNK_SIZE:-20}"
RL_CHUNK_PLAN_MULTIPLIER="${RL_CHUNK_PLAN_MULTIPLIER:-2}"
RL_CHUNK_EXTRA_FRAMES="${RL_CHUNK_EXTRA_FRAMES:-auto}"
RL_CHUNK_EARLY_STOP_ON_SUCCESS="${RL_CHUNK_EARLY_STOP_ON_SUCCESS:-1}"
RL_CHUNK_SUCCESS_EXTRA_ITERS="${RL_CHUNK_SUCCESS_EXTRA_ITERS:-1}"
RL_CHUNK_VIZ_INTERVAL="${RL_CHUNK_VIZ_INTERVAL:-20}"
RL_CHUNK_PLATEAU_EVALS="${RL_CHUNK_PLATEAU_EVALS:-0}"
RL_CHUNK_PLATEAU_MIN_DELTA="${RL_CHUNK_PLATEAU_MIN_DELTA:-0.25}"
RL_CHUNK_TRAIN_PLATEAU_WINDOW="${RL_CHUNK_TRAIN_PLATEAU_WINDOW:-20}"
RL_CHUNK_TRAIN_PLATEAU_MIN_DELTA="${RL_CHUNK_TRAIN_PLATEAU_MIN_DELTA:-0.5}"
RL_CHUNK_TRAIN_SUCCESS_TRIGGER_EVAL="${RL_CHUNK_TRAIN_SUCCESS_TRIGGER_EVAL:-1}"
RL_CHUNK_TRAIN_SUCCESS_TRIGGER_COUNT="${RL_CHUNK_TRAIN_SUCCESS_TRIGGER_COUNT:-2}"
RL_CHUNK_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX="${RL_CHUNK_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX:-8}"
LOAD_CKPT="${LOAD_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_MODE=0

SIM_DT="0.01"
CTRL_DT="0.05"
HORIZON="1.6"
KNOT_DT="0.4"

OBJ_POS_REW_SCALE="1.0"
OBJ_ROT_REW_SCALE="0.3"
JOINT_REW_SCALE="0.0001"
BASE_POS_REW_SCALE="0.001"
BASE_ROT_REW_SCALE="0.001"
VEL_REW_SCALE="0.000"
CONTACT_REW_SCALE="0.0"
LIFTING_REW_SCALE="0.0"
OBJECT_POS_THRESHOLD="0.05"
OBJECT_ROT_THRESHOLD="1"
OBJECT_MASS_SCALE="0.1"
ACTION_SMOOTHNESS_REW_SCALE="0.005"
DR_PAIR_MARGIN_MIN="0.0"
DR_PAIR_MARGIN_MAX="0.0"
DR_XY_OFFSET_MIN="0.0"
DR_XY_OFFSET_MAX="0.0"
NUM_DR="0"
DR_OBJECT_MASS_SCALE_MIN="1.0"
DR_OBJECT_MASS_SCALE_MAX="1.0"

FIRST_CTRL_NOISE="0.5"
LAST_CTRL_NOISE="0.65"
FINAL_NOISE="0.1"
JOINT_NOISE_SCALE="0.3"
POS_NOISE_SCALE="0.08"
ROT_NOISE_SCALE="0.05"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

RL_PYTHON_BIN="${RL_PYTHON_BIN:-python}"
RL_LOG_DIR="${CODE_ROOT}/outputs/rl_rsl"

EPISODES=()
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
    --load-ckpt|--load_ckpt|--load)
      LOAD_CKPT="$2"
      shift 2
      ;;
    --resume-ckpt|--resume_ckpt)
      RESUME_MODE=1
      RESUME_CKPT="$2"
      shift 2
      ;;
    --resume)
      RESUME_MODE=1
      if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
        RESUME_CKPT="$2"
        shift 2
      else
        shift
      fi
      ;;
    -h|--help)
      printf '%s\n' "Usage: bash run_aria_rl_chunk.sh --task-name {mustard|drawer|flower|hammer|all} [--episodes EP ...] [--load-ckpt PATH] [--resume [PATH]]"
      printf '%s\n' "Chunk env vars: RL_CHUNK_SIZE=20 RL_CHUNK_PLAN_MULTIPLIER=2 RL_CHUNK_VIZ_INTERVAL=20 RL_CHUNK_TRAIN_SUCCESS_TRIGGER_EVAL=1 RL_CHUNK_TRAIN_SUCCESS_TRIGGER_COUNT=2 RL_CHUNK_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX=8 RL_CHUNK_TRAIN_PLATEAU_WINDOW=20 RL_CHUNK_TRAIN_PLATEAU_MIN_DELTA=0.5 RL_CHUNK_SUCCESS_EXTRA_ITERS=1"
      exit 0
      ;;
    *)
      printf '%s\n' "[ERROR] unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "${TASK_NAME}" ]; then
  printf '%s\n' "[ERROR] please run task by task: --task-name mustard|drawer|flower|hammer"
  exit 1
fi

LOAD_CKPT_FLAG=()
if [[ -n "${LOAD_CKPT}" ]]; then
  LOAD_CKPT_FLAG+=(--load-ckpt "${LOAD_CKPT}")
fi

if [[ -n "${LOAD_CKPT}" && "${RESUME_MODE}" == "1" ]]; then
  printf '%s\n' "[ERROR] use only one of --load-ckpt or --resume"
  exit 1
fi

load_aria_rl_config() {
  local task="$1"
  eval "$(python3 - "${CODE_ROOT}/configs" "${task}" "${CODE_ROOT}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_dir = Path(sys.argv[1])
task = sys.argv[2]
code_root = Path(sys.argv[3])

task_path = cfg_dir / f"aria_{task}_rl.json"
base_name = "aria_rl.json"
if task_path.exists():
    task_json = json.loads(task_path.read_text())
    base_name = task_json.get("base_config", base_name)
else:
    task_json = {}

base = json.loads((cfg_dir / base_name).read_text())

def emit(name, value):
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{name}={shlex.quote(str(value))}")

dataset = base.get("dataset", {})
if "aria_root_base" in dataset:
    emit("ARIA_ROOT_BASE", dataset["aria_root_base"])
if "dataset_dir" in dataset:
    emit("DATASET_DIR", dataset["dataset_dir"])
if "dataset_name" in dataset:
    emit("DATASET_NAME", dataset["dataset_name"])
if "robot_type" in dataset:
    emit("ROBOT_TYPE", dataset["robot_type"])
if "source_embodiment" in dataset:
    emit("SOURCE_EMBODIMENT", dataset["source_embodiment"])
if "hz" in dataset:
    emit("HZ", dataset["hz"])
if "hand_modes" in dataset:
    modes = " ".join(shlex.quote(str(x)) for x in dataset["hand_modes"])
    print(f"HAND_MODES=({modes})")
roots = dataset.get("roots", {})
for name, var in {
    "mustard": "MUSTARD_ROOT",
    "drawer": "DRAWER_ROOT",
    "flower": "FLOWER_ROOT",
    "hammer": "HAMMER_ROOT",
}.items():
    if name in roots:
        emit(var, str(Path(dataset.get("aria_root_base", "")) / roots[name]) if dataset.get("aria_root_base") else roots[name])

run = base.get("run", {})
if "python_bin" in run:
    emit("RL_PYTHON_BIN", run["python_bin"])
if "device" in run:
    emit("RL_DEVICE", run["device"])
if "log_dir" in run:
    log_dir = Path(run["log_dir"])
    emit("RL_LOG_DIR", log_dir if log_dir.is_absolute() else code_root / log_dir)

section_maps = [
    (base.get("rl", {}), {
        "num_envs": "RL_NUM_ENVS",
        "eval_num_envs": "RL_EVAL_NUM_ENVS",
        "future_ctrl_steps": "RL_FUTURE_CTRL_STEPS",
        "max_episode_ctrl_steps": "RL_EPISODE_CTRL_STEPS",
        "num_steps_per_env": "RL_NUM_STEPS_PER_ENV",
        "learning_iterations": "RL_LEARNING_ITERS",
        "early_stop_on_success": "RL_EARLY_STOP_ON_SUCCESS",
        "ckpt_interval": "RL_CKPT_INTERVAL",
        "viz_interval": "RL_VIZ_INTERVAL",
        "rollout_video_ctrl_steps": "RL_ROLLOUT_VIDEO_CTRL_STEPS",
    }),
    (base.get("wandb", {}), {
        "media_only": "WANDB_MEDIA_ONLY",
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "run_name": "WANDB_RUN_NAME",
    }),
    (base.get("timing", {}), {
        "sim_dt": "SIM_DT",
        "ctrl_dt": "CTRL_DT",
        "horizon": "HORIZON",
        "knot_dt": "KNOT_DT",
    }),
    (base.get("reward", {}), {
        "pos_rew_scale": "OBJ_POS_REW_SCALE",
        "rot_rew_scale": "OBJ_ROT_REW_SCALE",
        "joint_rew_scale": "JOINT_REW_SCALE",
        "base_pos_rew_scale": "BASE_POS_REW_SCALE",
        "base_rot_rew_scale": "BASE_ROT_REW_SCALE",
        "vel_rew_scale": "VEL_REW_SCALE",
        "contact_rew_scale": "CONTACT_REW_SCALE",
        "lifting_rew_scale": "LIFTING_REW_SCALE",
        "object_pos_threshold": "OBJECT_POS_THRESHOLD",
        "object_rot_threshold": "OBJECT_ROT_THRESHOLD",
        "object_mass_scale": "OBJECT_MASS_SCALE",
        "action_smoothness_rew_scale": "ACTION_SMOOTHNESS_REW_SCALE",
    }),
    (base.get("domain_randomization", {}), {
        "num_dr": "NUM_DR",
        "pair_margin_range": ("DR_PAIR_MARGIN_MIN", "DR_PAIR_MARGIN_MAX"),
        "xy_offset_range": ("DR_XY_OFFSET_MIN", "DR_XY_OFFSET_MAX"),
        "object_mass_scale_range": ("DR_OBJECT_MASS_SCALE_MIN", "DR_OBJECT_MASS_SCALE_MAX"),
    }),
    (base.get("noise", {}), {
        "first_ctrl_noise_scale": "FIRST_CTRL_NOISE",
        "last_ctrl_noise_scale": "LAST_CTRL_NOISE",
        "final_noise_scale": "FINAL_NOISE",
        "joint_noise_scale": "JOINT_NOISE_SCALE",
        "pos_noise_scale": "POS_NOISE_SCALE",
        "rot_noise_scale": "ROT_NOISE_SCALE",
    }),
]

flat_map = {}
for section, mapping in section_maps:
    for key, var in mapping.items():
        if key not in section:
            continue
        value = section[key]
        if isinstance(var, tuple):
            emit(var[0], value[0])
            emit(var[1], value[1])
        else:
            emit(var, value)
        flat_map[key] = var

overrides = task_json.get("overrides", {})
for key, value in overrides.items():
    var = flat_map.get(key)
    if var is None:
        continue
    if isinstance(var, tuple):
        emit(var[0], value[0])
        emit(var[1], value[1])
    else:
        emit(var, value)
PY
)"
}

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

collect_episodes() {
  local task="$1"
  local root="$2"
  if [ ${#EPISODES[@]} -gt 0 ]; then
    printf '%s\n' "${EPISODES[@]}"
  else
    find "${root}/Hand_Poses" -mindepth 1 -maxdepth 1 -type d -name "${task}_*" -printf '%f\n' | sort -V
  fi
}

trajectory_frame_count() {
  local traj_path="$1"
  "${RL_PYTHON_BIN}" - "${traj_path}" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1]) as data:
    print(int(data["qpos"].shape[0]))
PY
}

chunk_padding_frames() {
  if [[ "${RL_CHUNK_EXTRA_FRAMES}" != "auto" ]]; then
    printf '%s\n' "${RL_CHUNK_EXTRA_FRAMES}"
    return
  fi
  awk -v horizon="${HORIZON}" -v hz="${HZ}" 'BEGIN { printf "%d\n", int(horizon * hz + 0.999999) + 2 }'
}

write_chunk_data() {
  local src_path="$1"
  local out_path="$2"
  local start_frame="$3"
  local plan_frames="$4"
  local padding_frames="$5"
  local prev_rollout_path="${6:-}"
  "${RL_PYTHON_BIN}" - "${src_path}" "${out_path}" "${start_frame}" "${plan_frames}" "${padding_frames}" "${prev_rollout_path}" <<'PY'
import sys
from pathlib import Path

import numpy as np

src_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
start = int(sys.argv[3])
plan_frames = int(sys.argv[4])
padding_frames = int(sys.argv[5])
prev_rollout = Path(sys.argv[6]) if sys.argv[6] else None

with np.load(src_path) as data:
    qpos = data["qpos"]
    n = int(qpos.shape[0])
    stop = min(n, start + plan_frames + padding_frames)
    if stop <= start + 1:
        raise SystemExit(f"empty chunk: start={start} stop={stop} n={n}")
    payload = {}
    for key in data.files:
        arr = data[key]
        if getattr(arr, "ndim", 0) > 0 and int(arr.shape[0]) == n:
            payload[key] = arr[start:stop].copy()
        else:
            payload[key] = arr.copy()

if prev_rollout is not None and prev_rollout.is_file():
    with np.load(prev_rollout) as prev:
        if "qpos" in payload and "qpos" in prev:
            payload["qpos"][0] = prev["qpos"][-1]
        if "qvel" in payload and "qvel" in prev:
            payload["qvel"][0] = prev["qvel"][-1]
        if "ctrl" in payload and "ctrl" in prev:
            payload["ctrl"][0] = prev["ctrl"][-1]

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_path, **payload)
print(f"[CHUNK_DATA] path={out_path} start={start} stop={stop} timesteps={stop-start} plan_timesteps={plan_frames} padding_timesteps={padding_frames}")
PY
}

aggregate_chunk_outputs() {
  local chunk_summary_path="$1"
  local out_dir="$2"
  local python_bin="$3"
  "${python_bin}" - "${chunk_summary_path}" "${out_dir}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with summary_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

reward_actual = sum(float(r["execute_reward_actual"]) for r in rows)
reward_max = sum(float(r["execute_reward_max"]) for r in rows)
steps_actual = max((float(r["chunk_end"]) for r in rows), default=0.0)
steps_max = max((float(r["total_frames"]) for r in rows), default=0.0)
elapsed = sum(float(r["training_elapsed"]) for r in rows)
success = 1 if steps_max > 0 and steps_actual >= steps_max else 0
speed = steps_actual / elapsed if elapsed > 0 else (math.inf if steps_actual > 0 else 0.0)

payload = {
    "chunks": rows,
    "reward_actual": reward_actual,
    "reward_max": reward_max,
    "reward_percent": (100.0 * reward_actual / reward_max) if reward_max > 0 else None,
    "steps_actual": steps_actual,
    "steps_max": steps_max,
    "steps_percent": (100.0 * steps_actual / steps_max) if steps_max > 0 else None,
    "training_elapsed": elapsed,
    "speed_steps_per_second": speed,
    "success": success,
}
(out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
with (out_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["success", "steps", "reward", "speed_steps_per_second", "training_elapsed"])
    reward = f"{reward_actual:.6f}/{reward_max:.6f}"
    steps = f"{steps_actual:.0f}/{steps_max:.0f}"
    writer.writerow([success, steps, reward, f"{speed:.6f}", f"{elapsed:.6f}"])

traj = []
for row in rows:
    path = Path(row["best_trajectory_path"])
    if not path.is_file():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("traj", [])
    execute_n = int(round(float(row["chunk_end"]) - float(row["chunk_start"])))
    traj.extend(frames[:execute_n])
if traj:
    first = Path(rows[0]["best_trajectory_path"])
    base = json.loads(first.read_text(encoding="utf-8"))
    base["traj"] = traj
    base["num_frames"] = len(traj)
    (out_dir / "best_trajectory_merged.json").write_text(json.dumps(base, indent=2), encoding="utf-8")

video_paths = [Path(r["video_path"]) for r in rows if r.get("video_path")]
video_paths = [p for p in video_paths if p.is_file()]
if video_paths:
    try:
        import imageio.v2 as imageio  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        (out_dir / "video_concat_error.txt").write_text(str(exc), encoding="utf-8")
    else:
        out_video = out_dir / "rollout_merged.mp4"
        writer = None
        try:
            for row in rows:
                video_path = Path(row["video_path"])
                if not video_path.is_file():
                    continue
                rollout_path = Path(row["best_rollout_path"])
                max_frames = None
                if rollout_path.is_file():
                    with np.load(rollout_path) as data:
                        if "qpos" in data:
                            max_frames = max(1, int(data["qpos"].shape[0]) - 1)
                reader = imageio.get_reader(str(video_path))
                meta = reader.get_meta_data()
                fps = meta.get("fps", 100)
                if writer is None:
                    writer = imageio.get_writer(str(out_video), fps=fps)
                for idx, frame in enumerate(reader):
                    if max_frames is not None and idx >= max_frames:
                        break
                    writer.append_data(frame)
                reader.close()
        finally:
            if writer is not None:
                writer.close()

print(
    "[CHUNK_FINAL] "
    f"summary={out_dir / 'summary.json'} "
    f"video={out_dir / 'rollout_merged.mp4'} "
    f"reward={reward_actual:.6f}/{reward_max:.6f} "
    f"steps={steps_actual:.0f}/{steps_max:.0f} "
    f"success={success} "
    f"speed={speed:.6f}"
)
PY
}

TASKS=()
case "${TASK_NAME}" in
  all) TASKS=(mustard drawer flower hammer) ;;
  mustard|drawer|flower|hammer) TASKS=("${TASK_NAME}") ;;
  *)
    printf '%s\n' "[ERROR] unsupported --task-name: ${TASK_NAME}"
    exit 1
    ;;
esac

load_aria_rl_config "${TASKS[0]}"
"${RL_PYTHON_BIN}" -c "import rsl_rl" >/dev/null 2>&1 || {
  printf '%s\n' "[ERROR] rsl_rl not found in ${RL_PYTHON_BIN}. Try: pip install rsl-rl-lib"
  exit 1
}

WANDB_MEDIA_FLAG=()
if [[ "${WANDB_MEDIA_ONLY}" == "1" ]]; then
  WANDB_MEDIA_FLAG+=(--wandb-media-only)
fi

tmp_summary="$(mktemp)"
trap 'rm -f "${tmp_summary}"' EXIT

attempted_count=0
skipped_count=0
failed_count=0
completed_count=0

# ========== RL one by one ==========
for task_name in "${TASKS[@]}"; do
  load_aria_rl_config "${task_name}"
  DATA_ROOT="$(task_root "${task_name}")" || exit 1
  if [ ! -d "${DATA_ROOT}/Hand_Poses" ]; then
    printf '%s\n' "[ERROR] Hand_Poses not found: ${DATA_ROOT}/Hand_Poses"
    failed_count=$((failed_count + 1))
    continue
  fi

  while IFS= read -r episode; do
    [ -z "$episode" ] && continue

    TASK="${task_name}"
    DATA_ID="$(task_id_from_episode "${task_name}" "${episode}")"

    TASK_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}/${DATA_ID}"
    TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${SOURCE_EMBODIMENT}/${TASK}"
    MANO_TASK_ROOT_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${SOURCE_EMBODIMENT}/${TASK}"
    KIN_PATH="${TASK_DIR}/trajectory_kinematic.npz"
    SCENE_EQ_PATH="${TASK_ROOT_DIR}/scene_eq.xml"
    SRC_DIR="${DATA_ROOT}/Hand_Poses/${episode}/1"

    if [ -f "${SRC_DIR}/trajectory_kinematic.npz" ] && [ -f "${SRC_DIR}/scene_eq.xml" ]; then
      mkdir -p "${TASK_DIR}" "${TASK_ROOT_DIR}" "${MANO_TASK_ROOT_DIR}"
      cp -f "${SRC_DIR}/trajectory_kinematic.npz" "${KIN_PATH}"
      cp -f "${SRC_DIR}/scene_eq.xml" "${TASK_ROOT_DIR}/scene_eq.xml"
      [ -f "${SRC_DIR}/scene.xml" ] && cp -f "${SRC_DIR}/scene.xml" "${TASK_ROOT_DIR}/scene.xml"
      if [ -f "${SRC_DIR}/task_info.json" ]; then
        cp -f "${SRC_DIR}/task_info.json" "${TASK_ROOT_DIR}/task_info.json"
        cp -f "${SRC_DIR}/task_info.json" "${MANO_TASK_ROOT_DIR}/task_info.json"
      fi
    fi

    if [ ! -f "${KIN_PATH}" ] || [ ! -f "${SCENE_EQ_PATH}" ]; then
      printf '%s\n' "[SKIP] missing processed files for ${TASK}_${DATA_ID}"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    for hand_mode in "${HAND_MODES[@]}"; do
      printf '%s\n' "=== RL Aria ${TASK} - ${episode} (${hand_mode}) ==="

      TOTAL_FRAMES="$(trajectory_frame_count "${KIN_PATH}")"
      if [ "${TOTAL_FRAMES}" -le 1 ]; then
        printf '%s\n' "[SKIP] trajectory too short for ${TASK}_${DATA_ID}: frames=${TOTAL_FRAMES}"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      CHUNK_IDX=0
      CHUNK_START=0
      PREV_BEST_ROLLOUT=""
      PREV_BEST_CKPT=""
      DEMO_FAILED=0

      CUR_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
      if [ -z "${CUR_WANDB_RUN_NAME}" ]; then
        CUR_WANDB_RUN_NAME="rl_${TASK}_${DATA_ID}_${hand_mode}"
      fi
      if [ -n "${RL_EXPERIMENT_SUFFIX}" ]; then
        CUR_WANDB_RUN_NAME="${CUR_WANDB_RUN_NAME}_${RL_EXPERIMENT_SUFFIX}"
      fi

      SUMMARY_EXPERIMENT_NAME="${TASK}_${DATA_ID}_rl_chunk_summary"
      if [ -n "${RL_EXPERIMENT_SUFFIX}" ]; then
        SUMMARY_EXPERIMENT_NAME="${SUMMARY_EXPERIMENT_NAME}_${RL_EXPERIMENT_SUFFIX}"
      fi
      SUMMARY_LOG_DIR="${RL_LOG_DIR}/${SUMMARY_EXPERIMENT_NAME}"
      mkdir -p "${SUMMARY_LOG_DIR}"
      CHUNK_SUMMARY_PATH="${SUMMARY_LOG_DIR}/chunks.tsv"
      printf '%s\n' "key	chunk_start	chunk_end	plan_end	execute_reward_actual	execute_reward_max	execute_steps_actual	execute_steps_max	plan_reward_actual	plan_reward_max	plan_steps_actual	plan_steps_max	training_elapsed	success	total_frames	best_rollout_path	best_trajectory_path	video_path	log_path" > "${CHUNK_SUMMARY_PATH}"

      while [ "${CHUNK_START}" -lt "${TOTAL_FRAMES}" ]; do
        attempted_count=$((attempted_count + 1))
        CHUNK_END=$((CHUNK_START + RL_CHUNK_SIZE))
        if [ "${CHUNK_END}" -gt "${TOTAL_FRAMES}" ]; then
          CHUNK_END="${TOTAL_FRAMES}"
        fi
        if [ $((TOTAL_FRAMES - CHUNK_END)) -eq 1 ]; then
          CHUNK_END="${TOTAL_FRAMES}"
        fi
        CUR_CHUNK_SIZE=$((CHUNK_END - CHUNK_START))
        if [ "${CUR_CHUNK_SIZE}" -le 1 ]; then
          break
        fi
        CUR_PLAN_FRAMES=$((CUR_CHUNK_SIZE * RL_CHUNK_PLAN_MULTIPLIER))
        CUR_PADDING_FRAMES="$(chunk_padding_frames)"
        PLAN_END=$((CHUNK_START + CUR_PLAN_FRAMES))
        if [ "${PLAN_END}" -gt "${TOTAL_FRAMES}" ]; then
          PLAN_END="${TOTAL_FRAMES}"
        fi
        CUR_PLAN_SIZE=$((PLAN_END - CHUNK_START))

        EXPERIMENT_NAME="${TASK}_${DATA_ID}_rl_chunk_$(printf '%04d' "${CHUNK_START}")_$(printf '%04d' "${CHUNK_END}")"
        if [ -n "${RL_EXPERIMENT_SUFFIX}" ]; then
          EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RL_EXPERIMENT_SUFFIX}"
        fi
        TASK_LOG_DIR="${RL_LOG_DIR}/${EXPERIMENT_NAME}"
        CHUNK_DATA_PATH="${TASK_LOG_DIR}/trajectory_kinematic_chunk.npz"
        write_chunk_data "${KIN_PATH}" "${CHUNK_DATA_PATH}" "${CHUNK_START}" "${CUR_PLAN_FRAMES}" "${CUR_PADDING_FRAMES}" "${PREV_BEST_ROLLOUT}"
        printf '%s\n' "[CHUNK] execute_timesteps=${CHUNK_START}:${CHUNK_END} plan_timesteps=${CHUNK_START}:${PLAN_END} padding_timesteps=${CUR_PADDING_FRAMES}"
      mkdir -p "${TASK_LOG_DIR}"
      RUN_LOG_PATH="${TASK_LOG_DIR}/train.log"

      RESUME_CKPT_FLAG=()
      if [[ "${RESUME_MODE}" == "1" ]]; then
        CUR_RESUME_CKPT="${RESUME_CKPT}"
        if [[ -z "${CUR_RESUME_CKPT}" ]]; then
          CUR_RESUME_CKPT="${TASK_LOG_DIR}/checkpoints/eval_latest.pt"
        fi
        RESUME_CKPT_FLAG+=(--resume-ckpt "${CUR_RESUME_CKPT}")
      fi

      if [[ "${RESUME_MODE}" == "1" ]]; then
        printf '%s\n' "[INFO] resuming from ${RESUME_CKPT_FLAG[1]}"
        TEE_ARGS=(-a "${RUN_LOG_PATH}")
      else
        : > "${RUN_LOG_PATH}"
        TEE_ARGS=("${RUN_LOG_PATH}")
      fi

      CHUNK_LOAD_CKPT_FLAG=("${LOAD_CKPT_FLAG[@]}")
      if [[ "${RESUME_MODE}" != "1" && -n "${PREV_BEST_CKPT}" ]]; then
        CHUNK_LOAD_CKPT_FLAG=(--load-ckpt "${PREV_BEST_CKPT}")
      fi
      if [[ "${RESUME_MODE}" != "1" && -n "${PREV_BEST_CKPT}" ]]; then
        printf '%s\n' "[CHUNK_MODEL] continuing RL model from previous chunk ckpt=${PREV_BEST_CKPT}"
      elif [[ "${RESUME_MODE}" != "1" && -n "${LOAD_CKPT}" ]]; then
        printf '%s\n' "[CHUNK_MODEL] initializing first chunk from ckpt=${LOAD_CKPT}"
      elif [[ "${RESUME_MODE}" != "1" ]]; then
        printf '%s\n' "[CHUNK_MODEL] initializing first chunk from scratch"
      fi

      if "${RL_PYTHON_BIN}" examples/run_mjwp_rl_rsl.py \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --robot-type "${ROBOT_TYPE}" \
        --embodiment-type "${SOURCE_EMBODIMENT}" \
        --active-hand "${hand_mode}" \
        --task "${TASK}" \
        --data-id "${DATA_ID}" \
        --data-path "${CHUNK_DATA_PATH}" \
        --residual-schedule-start-timestep "${CHUNK_START}" \
        --residual-schedule-total-timesteps "${TOTAL_FRAMES}" \
        --device "${RL_DEVICE}" \
        --num-envs "${RL_NUM_ENVS}" \
        --eval-num-envs "${RL_EVAL_NUM_ENVS}" \
        --future-ctrl-steps "${RL_FUTURE_CTRL_STEPS}" \
        --max-episode-ctrl-steps "${RL_EPISODE_CTRL_STEPS}" \
        --eval-max-ctrl-steps "${CUR_PLAN_SIZE}" \
        --success-eval-ctrl-steps "${CUR_PLAN_SIZE}" \
        --export-execute-ctrl-steps "${CUR_CHUNK_SIZE}" \
        --num-steps-per-env "${RL_NUM_STEPS_PER_ENV}" \
        --learning-iterations "${RL_LEARNING_ITERS}" \
        --early-stop-on-success "${RL_CHUNK_EARLY_STOP_ON_SUCCESS}" \
        --success-extra-iterations "${RL_CHUNK_SUCCESS_EXTRA_ITERS}" \
        --early-stop-on-full-eval-success "1" \
        --train-success-trigger-eval "${RL_CHUNK_TRAIN_SUCCESS_TRIGGER_EVAL}" \
        --train-success-trigger-count "${RL_CHUNK_TRAIN_SUCCESS_TRIGGER_COUNT}" \
        --train-success-trigger-ctrl-steps "${CUR_CHUNK_SIZE}" \
        --train-success-trigger-backoff-max "${RL_CHUNK_TRAIN_SUCCESS_TRIGGER_BACKOFF_MAX}" \
        --plateau-after-success-eval-count "${RL_CHUNK_PLATEAU_EVALS}" \
        --plateau-after-success-min-delta "${RL_CHUNK_PLATEAU_MIN_DELTA}" \
        --train-plateau-after-success-window "${RL_CHUNK_TRAIN_PLATEAU_WINDOW}" \
        --train-plateau-after-success-min-delta "${RL_CHUNK_TRAIN_PLATEAU_MIN_DELTA}" \
        --ckpt-interval "${RL_CKPT_INTERVAL}" \
        --viz-interval "${RL_CHUNK_VIZ_INTERVAL}" \
        --rollout-video-ctrl-steps "${RL_ROLLOUT_VIDEO_CTRL_STEPS}" \
        --log-dir "${RL_LOG_DIR}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-run-name "${CUR_WANDB_RUN_NAME}_chunk_${CHUNK_START}_${CHUNK_END}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        "${WANDB_MEDIA_FLAG[@]}" \
        "${CHUNK_LOAD_CKPT_FLAG[@]}" \
        "${RESUME_CKPT_FLAG[@]}" \
        --sim-dt "${SIM_DT}" \
        --ctrl-dt "${CTRL_DT}" \
        --horizon "${HORIZON}" \
        --knot-dt "${KNOT_DT}" \
        --pos-rew-scale "${OBJ_POS_REW_SCALE}" \
        --rot-rew-scale "${OBJ_ROT_REW_SCALE}" \
        --joint-rew-scale "${JOINT_REW_SCALE}" \
        --base-pos-rew-scale "${BASE_POS_REW_SCALE}" \
        --base-rot-rew-scale "${BASE_ROT_REW_SCALE}" \
        --vel-rew-scale "${VEL_REW_SCALE}" \
        --contact-rew-scale "${CONTACT_REW_SCALE}" \
        --lifting-rew-scale "${LIFTING_REW_SCALE}" \
        --object-pos-threshold "${OBJECT_POS_THRESHOLD}" \
        --object-rot-threshold "${OBJECT_ROT_THRESHOLD}" \
        --object-mass-scale "${OBJECT_MASS_SCALE}" \
        --action-smoothness-rew-scale "${ACTION_SMOOTHNESS_REW_SCALE}" \
        --num-dr "${NUM_DR}" \
        --pair-margin-range-min "${DR_PAIR_MARGIN_MIN}" \
        --pair-margin-range-max "${DR_PAIR_MARGIN_MAX}" \
        --xy-offset-range-min "${DR_XY_OFFSET_MIN}" \
        --xy-offset-range-max "${DR_XY_OFFSET_MAX}" \
        --object-mass-scale-range-min "${DR_OBJECT_MASS_SCALE_MIN}" \
        --object-mass-scale-range-max "${DR_OBJECT_MASS_SCALE_MAX}" \
        --first-ctrl-noise-scale "${FIRST_CTRL_NOISE}" \
        --last-ctrl-noise-scale "${LAST_CTRL_NOISE}" \
        --final-noise-scale "${FINAL_NOISE}" \
        --joint-noise-scale "${JOINT_NOISE_SCALE}" \
        --pos-noise-scale "${POS_NOISE_SCALE}" \
        --rot-noise-scale "${ROT_NOISE_SCALE}" 2>&1 | tee "${TEE_ARGS[@]}"
      then
        rl_rc=0
      else
        rl_rc=$?
      fi

      rl_line="$(grep -a '^[[]RL[]] task=' "${RUN_LOG_PATH}" | tail -n 1)"
      if [ ${rl_rc} -ne 0 ] || [ -z "${rl_line}" ]; then
        failed_count=$((failed_count + 1))
        DEMO_FAILED=1
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode} log=${RUN_LOG_PATH}"
        break
      fi

      fps="$(printf '%s\n' "${rl_line}" | sed -En 's/.*fps=([^ ]*).*/\1/p')"
      reward_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*reward_actual=([^ ]*).*/\1/p')"
      reward_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*reward_max=([^ ]*).*/\1/p')"
      steps_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*steps_actual=([^ ]*).*/\1/p')"
      steps_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*steps_max=([^ ]*).*/\1/p')"
      execute_reward_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*execute_reward_actual=([^ ]*).*/\1/p')"
      execute_reward_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*execute_reward_max=([^ ]*).*/\1/p')"
      execute_steps_actual="$(printf '%s\n' "${rl_line}" | sed -En 's/.*execute_steps_actual=([^ ]*).*/\1/p')"
      execute_steps_max="$(printf '%s\n' "${rl_line}" | sed -En 's/.*execute_steps_max=([^ ]*).*/\1/p')"
      if [ -z "${execute_reward_actual}" ]; then execute_reward_actual="${reward_actual}"; fi
      if [ -z "${execute_reward_max}" ]; then execute_reward_max="${reward_max}"; fi
      if [ -z "${execute_steps_actual}" ]; then execute_steps_actual="${steps_actual}"; fi
      if [ -z "${execute_steps_max}" ]; then execute_steps_max="${steps_max}"; fi
      success="$(printf '%s\n' "${rl_line}" | sed -En 's/.*success=([^ ]*).*/\1/p')"
      training_elapsed="$(printf '%s\n' "${rl_line}" | sed -En 's/.*training_elapsed=([^ ]*).*/\1/p')"

      reward_pct="$(awk -v a="${execute_reward_actual}" -v b="${execute_reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
      steps_pct="$(awk -v a="${execute_steps_actual}" -v b="${execute_steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
      printf '%s\n' "[SUMMARY] task=${TASK} episode=${episode} hand=${hand_mode} chunk=${CHUNK_START}:${CHUNK_END} execute_reward=${execute_reward_actual}/${execute_reward_max} (${reward_pct}) execute_steps=${execute_steps_actual}/${execute_steps_max} (${steps_pct}) plan_reward=${reward_actual}/${reward_max} plan_steps=${steps_actual}/${steps_max} success=${success}"

      if [ "${success}" != "1" ]; then
        failed_count=$((failed_count + 1))
        DEMO_FAILED=1
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode} chunk=${CHUNK_START}:${CHUNK_END} reason=chunk_not_completed log=${RUN_LOG_PATH}"
        break
      fi
      if [ "${RL_CHUNK_PLATEAU_EVALS}" -gt 0 ] && ! grep -a -q 'plateau after success' "${RUN_LOG_PATH}"; then
        failed_count=$((failed_count + 1))
        DEMO_FAILED=1
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode} chunk=${CHUNK_START}:${CHUNK_END} reason=no_plateau_stop log=${RUN_LOG_PATH}"
        break
      fi

      BEST_ROLLOUT_PATH="${TASK_LOG_DIR}/best_rollout.npz"
      if [ ! -f "${BEST_ROLLOUT_PATH}" ]; then
        failed_count=$((failed_count + 1))
        DEMO_FAILED=1
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode} chunk=${CHUNK_START}:${CHUNK_END} reason=missing_best_rollout path=${BEST_ROLLOUT_PATH}"
        break
      fi
      BEST_CKPT_PATH="${TASK_LOG_DIR}/checkpoints/eval_best.pt"
      if [ ! -f "${BEST_CKPT_PATH}" ]; then
        failed_count=$((failed_count + 1))
        DEMO_FAILED=1
        printf '%s\n' "[FAIL] task=${TASK} episode=${episode} hand=${hand_mode} chunk=${CHUNK_START}:${CHUNK_END} reason=missing_best_ckpt path=${BEST_CKPT_PATH}"
        break
      fi
      BEST_TRAJECTORY_PATH="${TASK_LOG_DIR}/best_trajectory.json"
      VIDEO_PATH="${TASK_LOG_DIR}/viz/rollout_iter_0.mp4"

      completed_count=$((completed_count + 1))
      printf '%s %s %s %s %s %s %s %s %s\n' \
        "${TASK}_${DATA_ID}:${hand_mode}" \
        "${execute_reward_actual}" \
        "${execute_reward_max}" \
        "${execute_steps_actual}" \
        "${training_elapsed}" \
        "${success}" \
        "${CHUNK_END}" \
        "${TOTAL_FRAMES}" \
        "${fps}" >> "${tmp_summary}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TASK}_${DATA_ID}:${hand_mode}" \
        "${CHUNK_START}" \
        "${CHUNK_END}" \
        "${PLAN_END}" \
        "${execute_reward_actual}" \
        "${execute_reward_max}" \
        "${execute_steps_actual}" \
        "${execute_steps_max}" \
        "${reward_actual}" \
        "${reward_max}" \
        "${steps_actual}" \
        "${steps_max}" \
        "${training_elapsed}" \
        "${success}" \
        "${TOTAL_FRAMES}" \
        "${BEST_ROLLOUT_PATH}" \
        "${BEST_TRAJECTORY_PATH}" \
        "${VIDEO_PATH}" \
        "${RUN_LOG_PATH}" >> "${CHUNK_SUMMARY_PATH}"
      PREV_BEST_ROLLOUT="${BEST_ROLLOUT_PATH}"
      PREV_BEST_CKPT="${BEST_CKPT_PATH}"
      CHUNK_IDX=$((CHUNK_IDX + 1))
      CHUNK_START="${CHUNK_END}"
      done
      if [ "${DEMO_FAILED}" -eq 0 ] && [ "${CHUNK_START}" -ge "${TOTAL_FRAMES}" ]; then
        aggregate_chunk_outputs "${CHUNK_SUMMARY_PATH}" "${SUMMARY_LOG_DIR}" "${RL_PYTHON_BIN}"
      fi
    done
  done < <(collect_episodes "${task_name}" "${DATA_ROOT}")
done

if [ ${completed_count} -gt 0 ]; then
  read -r avg_fps mean_reward_actual mean_reward_max mean_steps_actual mean_steps_max success_rate <<EOF_SUMMARY
$(awk '
{
  key=$1
  reward_actual += $2
  reward_max += $3
  speed_steps += $4
  elapsed += $5
  if (!(key in seen)) {
    seen[key]=1
    demos += 1
    final_step[key]=0
    total_frames[key]=$8
  }
  if ($7 > final_step[key]) {
    final_step[key]=$7
  }
  if ($8 > total_frames[key]) {
    total_frames[key]=$8
  }
}
END {
  for (key in seen) {
    steps_actual += final_step[key]
    steps_max += total_frames[key]
    success += (final_step[key] >= total_frames[key] && total_frames[key] > 0) ? 1 : 0
  }
  if (elapsed > 0) {
    fps = sprintf("%.6f", speed_steps / elapsed)
  } else if (speed_steps > 0) {
    fps = "inf"
  } else {
    fps = "0.000000"
  }
  if (demos > 0) {
    printf "%s %.6f %.6f %.6f %.6f %.6f", fps, reward_actual, reward_max, steps_actual, steps_max, success/demos
  } else {
    printf "nan nan nan nan nan nan"
  }
}' "${tmp_summary}")
EOF_SUMMARY
else
  avg_fps="nan"
  mean_reward_actual="nan"
  mean_reward_max="nan"
  mean_steps_actual="nan"
  mean_steps_max="nan"
  success_rate="nan"
fi

if [ "${mean_reward_actual}" != "nan" ]; then
  avg_reward_pct="$(awk -v a="${mean_reward_actual}" -v b="${mean_reward_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
  avg_steps_pct="$(awk -v a="${mean_steps_actual}" -v b="${mean_steps_max}" 'BEGIN { if (b > 0) printf "%.2f%%", 100.0 * a / b; else printf "nan" }')"
  success_rate_pct="$(awk -v x="${success_rate}" 'BEGIN { printf "%.2f%%", 100.0 * x }')"
else
  avg_reward_pct="nan"
  avg_steps_pct="nan"
  success_rate_pct="nan"
fi

printf '%s\n' "[FINAL] demos=${completed_count} fps=${avg_fps} reward=${mean_reward_actual}/${mean_reward_max} (${avg_reward_pct}) steps=${mean_steps_actual}/${mean_steps_max} (${avg_steps_pct}) success_rate=${success_rate_pct}"
if [ ${skipped_count} -gt 0 ] || [ ${failed_count} -gt 0 ]; then
  printf '%s\n' "[FINAL] skipped=${skipped_count} failed=${failed_count} attempted=${attempted_count}"
fi
exit $?
