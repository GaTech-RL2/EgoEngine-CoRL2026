"""Run the real adaptive selector with small MPC sampling and normal tracking limits."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIORITY = ["replay", "mpc", "rl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "examples/prepared")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/mode_switching_check"
    )
    parser.add_argument("--task", default="mustard_1")
    parser.add_argument("--dataset-name", default="aria")
    parser.add_argument("--embodiment-type", default="right")
    parser.add_argument("--mpc-num-samples", type=int, default=8)
    parser.add_argument("--mpc-max-iters", type=int, default=2)
    parser.add_argument("--rl-num-envs", type=int, default=1024)
    parser.add_argument("--rl-total-learning-iterations", type=int, default=1500)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "action.examples.run_mjwp_mixed",
        "--dataset-dir",
        str(args.data_root.resolve()),
        "--dataset-name",
        args.dataset_name,
        "--robot-type",
        "xhand",
        "--embodiment-type",
        args.embodiment_type,
        "--active-hand",
        "right",
        "--task",
        args.task,
        "--device",
        "cuda:0",
        "--methods",
        ",".join(PRIORITY),
        "--window-size",
        "20",
        "--lookahead-chunks",
        "2",
        "--mpc-num-samples",
        str(args.mpc_num_samples),
        "--mpc-max-iters",
        str(args.mpc_max_iters),
        "--rl-num-envs",
        str(args.rl_num_envs),
        "--rl-total-learning-iterations",
        str(args.rl_total_learning_iterations),
        "--debug-output-dir",
        str(output),
        "--output-name",
        "mixed.npz",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("XDG_CACHE_HOME", str(output / "cache"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output / "cache/torchinductor"))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    with np.load(output / "mixed.npz", allow_pickle=False) as data:
        attempts = json.loads(data["attempts_json"].item())
        statistics = json.loads(data["statistics_json"].item())
        success = bool(data["success"].item())
        committed_modes = data["window_method_id"].tolist()
        for key in ["ctrl", "qpos", "qvel"]:
            if not np.isfinite(data[key]).all():
                raise ValueError(f"Nonfinite committed {key}")
        windows = {}
        for attempt in attempts:
            windows.setdefault(attempt["window_id"], []).append(attempt)
        summary = []
        for window_id, trials in windows.items():
            modes = [trial["method_name"] for trial in trials]
            if modes != PRIORITY[: len(modes)]:
                raise ValueError(f"Window {window_id} did not restart from Replay: {modes}")
            if any(trial["feasible"] for trial in trials[:-1]):
                raise ValueError("A feasible cheaper candidate was skipped")
            summary.append(
                {
                    "window_id": window_id,
                    "start_step": trials[0]["start_step"],
                    "plan_end_step": trials[0]["plan_end_step"],
                    "attempts": [
                        {
                            "mode": t["method_name"],
                            "feasible": t["feasible"],
                            "steps_executed": t["steps_executed"],
                            "termination_step": t["termination_step"],
                        }
                        for t in trials
                    ],
                }
            )
        reached_rl = any(t["method_name"] == "rl" for t in attempts)
        committed_rl = 2 in committed_modes
    report = {
        "scope": "Natural adaptive fallback with normal 0.05m/1rad limits; reduced MPC sampling; CoDesign goal-conditioned residual PPO; original prepared references",
        "task": args.task,
        "command": command[1:],
        "success": success,
        "reached_rl": reached_rl,
        "committed_rl": committed_rl,
        "priority_and_restart_verified": True,
        "windows": summary,
        "statistics": statistics,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("[adaptive-check]", json.dumps(report, allow_nan=False), flush=True)
    if not reached_rl:
        print(
            "[adaptive-check] This run did not reach RL; its natural fallback path remains untested."
        )


if __name__ == "__main__":
    main()
