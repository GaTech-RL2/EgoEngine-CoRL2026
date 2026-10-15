"""Small reproducible CUDA interface check, explicitly not a quality benchmark."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "examples/prepared")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/gpu_smoke_reproduction"
    )
    args = parser.parse_args()
    import mujoco
    import torch
    import warp

    if not torch.cuda.is_available():
        raise RuntimeError("This check requires an allocated NVIDIA CUDA GPU")
    source = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    relative = Path("processed/aria/xhand/right/mustard_1")
    fixture = output / "inputs"
    destination = fixture / relative
    if not (fixture / "assets").exists():
        (fixture / "assets").parent.mkdir(parents=True, exist_ok=True)
        (fixture / "assets").symlink_to(source / "assets", target_is_directory=True)
    shutil.copytree(source / relative, destination, dirs_exist_ok=True)
    trajectory = destination / "0/trajectory_kinematic.npz"
    with np.load(trajectory, allow_pickle=False) as raw:
        count = len(raw["qpos"])
        if count < 5:
            raise ValueError("At least five reference frames are required")
        payload = {
            key: raw[key][:5].copy()
            if raw[key].ndim and len(raw[key]) == count
            else raw[key].copy()
            for key in raw.files
        }
    np.savez(trajectory, **payload)
    print(
        "[smoke] 5 reference frames; 2-control-step chunks; loose tracking limits. API check only.",
        flush=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("XDG_CACHE_HOME", str(output / "cache"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output / "cache/torchinductor"))
    results = {}
    for method in ["replay", "mpc", "rl"]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "action.examples.run_mjwp_mixed",
            "--dataset-dir",
            str(fixture),
            "--dataset-name",
            "aria",
            "--robot-type",
            "xhand",
            "--embodiment-type",
            "right",
            "--active-hand",
            "right",
            "--task",
            "mustard_1",
            "--device",
            "cuda:0",
            "--methods",
            method,
            "--window-size",
            "2",
            "--lookahead-chunks",
            "2",
            "--horizon",
            ".1",
            "--knot-dt",
            ".05",
            "--mpc-num-samples",
            "8",
            "--mpc-max-iters",
            "2",
            "--rl-num-envs",
            "32",
            "--rl-num-steps-per-env",
            "2",
            "--rl-total-learning-iterations",
            "6",
            "--object-pos-threshold",
            "10",
            "--object-rot-threshold",
            "10",
            "--debug-output-dir",
            str(output),
            "--output-name",
            f"smoke_{method}.npz",
        ]
        subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)
        with np.load(output / f"smoke_{method}.npz", allow_pickle=False) as f:
            if not bool(f["success"].item()):
                raise RuntimeError(f"{method} did not complete the short interface check")
            for key in ["ctrl", "qpos", "qvel"]:
                if len(f[key]) != 25 or not np.isfinite(f[key]).all():
                    raise ValueError(f"Invalid {method} {key} export")
            if f["window_end"].tolist() != [10, 20, 25] or f["window_plan_end"].tolist() != [
                20,
                25,
                25,
            ]:
                raise ValueError("Two-chunk plan / one-chunk commit boundaries differ")
            results[method] = {
                "success": True,
                "shapes": {key: list(f[key].shape) for key in ["ctrl", "qpos", "qvel"]},
                "statistics": json.loads(f["statistics_json"].item()),
            }
    report = {
        "scope": "CUDA API/state/timing smoke only; five frames, loose thresholds; no task-quality or speedup claim",
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "warp": warp.__version__,
            "rsl_rl": importlib.metadata.version("rsl-rl-lib"),
            "device": torch.cuda.get_device_name(0),
        },
        "cases": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("[smoke] Replay, MPC and RL passed; report:", output / "report.json")


if __name__ == "__main__":
    main()
