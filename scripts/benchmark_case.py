"""Run one prepared case with a frozen runner and independently validate its states."""

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tracking_checks(qpos, reference, object_slice, position_limit=0.05, rotation_limit=1.0):
    """Compare integrated states to the next reference frame, including quaternion sign."""
    if qpos.ndim != 2 or reference.ndim != 2 or qpos.shape[1] != reference.shape[1]:
        raise ValueError("State/reference coordinate dimensions differ")
    if len(reference) <= len(qpos):
        raise ValueError("Post-step validation requires the next reference frame")
    current = qpos[:, object_slice].astype(np.float64)
    goal = reference[1 : len(qpos) + 1, object_slice].astype(np.float64)
    if current.shape[1] != 7:
        raise ValueError("Expected a seven-coordinate active object pose")
    if not len(current):
        return {"finite": True, "max_position_error_m": None,
                "max_orientation_error_rad": None, "violating_steps": 0, "within_limits": True}
    finite = bool(np.isfinite(qpos).all() and np.isfinite(goal).all())
    position = np.linalg.norm(current[:, :3] - goal[:, :3], axis=1)
    a, b = current[:, 3:], goal[:, 3:]
    a_norm, b_norm = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid_quaternions = (a_norm > 1e-8) & (b_norm > 1e-8)
    dot = np.sum(a * b, axis=1) / np.maximum(a_norm * b_norm, 1e-16)
    rotation = 2 * np.arccos(np.clip(np.abs(dot), 0, 1))
    invalid = ~valid_quaternions | ~np.isfinite(position) | ~np.isfinite(rotation)
    violation = invalid | (position > position_limit) | (rotation > rotation_limit)
    return {
        "finite": finite,
        "valid_quaternions": bool(valid_quaternions.all()),
        "max_position_error_m": float(position.max()) if np.isfinite(position).all() else None,
        "max_orientation_error_rad": float(rotation.max()) if np.isfinite(rotation).all() else None,
        "violating_steps": int(violation.sum()),
        "within_limits": bool(finite and not violation.any()),
    }


def source_fingerprint(root):
    names = ["action/examples/run_mjwp_mixed.py", "spider/rl/mjwp_rsl_env.py",
             "spider/rl/rsl_compat.py", "spider/mode_switching.py", "spider/simulators/mjwp.py",
             "spider/config.py", "spider/io.py", "spider/rl/state_layout.py",
             "spider/rl/goal_observation.py", "spider/rl/residual_distribution.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


def verify_profile(root, profile):
    tree = ast.parse((root / "action/examples/run_mjwp_mixed.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_make_rsl_train_cfg")
    namespace = {"Any": Any, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "recipe", "exec"), namespace)
    config = namespace["_make_rsl_train_cfg"](16, 1500, 0)
    modern = "ResidualGaussian" in config["actor"].get("distribution_cfg", {}).get("class_name", "")
    if modern != (profile == "codesign"):
        raise ValueError(f"Runner does not match requested {profile} profile")
    return config


def workflow_options(root, mode):
    tree = ast.parse((root / "action/examples/run_mjwp_mixed.py").read_text())
    arguments = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MixedArgs")
    fields = {n.target.id for n in arguments.body if isinstance(n, ast.AnnAssign)}
    if "rl_window_learning_iterations" not in fields:
        return []  # Preserve the exact historical command for frozen older runners.
    return ["--rl-window-learning-iterations", "1500" if mode == "pure_rl" else "300"]


MODES = {"switching": ("replay", "mpc", "rl"), "pure_rl": ("rl",),
         "replay_mpc": ("replay", "mpc")}


def validate_mode_order(attempts, enabled_methods):
    windows = {}
    for attempt in attempts:
        windows.setdefault(attempt["window_id"], []).append(attempt)
    for trials in windows.values():
        names = [trial["method_name"] for trial in trials]
        if names != list(enabled_methods[:len(names)]):
            raise ValueError("A chunk did not follow its enabled mode priority")
        if any(trial["feasible"] for trial in trials[:-1]):
            raise ValueError("An earlier feasible mode was skipped")


def mode_options(mode, reference_frames, ref_dt):
    if mode == "pure_rl":
        # One policy learns the complete demonstration; no chunk restarts or MPC.
        controls = math.ceil(reference_frames * ref_dt / 0.05)
        return ["--methods", "rl", "--window-size", str(controls),
                "--lookahead-chunks", "1", "--rl-max-episode-ctrl-steps", str(controls)]
    return ["--methods", ",".join(MODES[mode]), "--window-size", "20",
            "--lookahead-chunks", "2"]


def validate_artifact(data_root, case, artifact, enabled_methods=MODES["switching"]):
    # Use the same interpolation for both frozen runners, outside generation timing.
    import mujoco

    from spider.config import Config, process_config
    from spider.io import load_data

    cfg = process_config(Config(
        dataset_dir=str(data_root), dataset_name=case["dataset_name"],
        robot_type=case["robot_type"], embodiment_type=case["embodiment_type"],
        active_hand="right", task=case["task"], device="cpu", simulator="mjwp",
        sim_dt=0.01, ctrl_dt=0.05, horizon=1.6, knot_dt=0.4,
    ))
    references = load_data(cfg, cfg.data_path)[0].numpy()
    model = mujoco.MjModel.from_xml_path(cfg.model_path)
    joint = model.joint("right_object_joint")
    if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected an active object free joint")
    start = int(model.jnt_qposadr[joint.id])
    full_steps = len(references) - cfg.horizon_steps - cfg.ctrl_steps
    with np.load(artifact, allow_pickle=False) as data:
        ctrl, qpos, qvel = (data[name] for name in ("ctrl", "qpos", "qvel"))
        nominal_success = bool(data["success"].item())
        if not len(ctrl) and not nominal_success:
            # A failed first planning window commits no states. The retained
            # runner serializes these as (0, 0), rather than scene-width arrays.
            if qpos.shape == (0, 0):
                qpos = np.empty((0, model.nq))
            if qvel.shape == (0, 0):
                qvel = np.empty((0, model.nv))
        if ctrl.shape != (len(ctrl), model.nu) or qpos.shape != (len(ctrl), model.nq):
            raise ValueError("Control/state layout does not match scene")
        if qvel.shape != (len(ctrl), model.nv) or len(ctrl) > full_steps:
            raise ValueError("Velocity layout or committed trajectory length is invalid")
        finite = bool(all(np.isfinite(x).all() for x in (ctrl, qpos, qvel)))
        tracking = tracking_checks(qpos, references, slice(start, start + 7))
        statistics = json.loads(data["statistics_json"].item())
        attempts = json.loads(data["attempts_json"].item())
        mode_ids = data["window_method_id"].tolist()
        validate_mode_order(attempts, enabled_methods)
        rl_attempts = [a for a in attempts if a["method_name"] == "rl"]
    return {
        "nominal_success": nominal_success,
        "validated_complete": bool(nominal_success and len(ctrl) == full_steps
                                   and finite and tracking["within_limits"]),
        "committed_physics_steps": len(ctrl), "full_physics_steps": full_steps,
        "committed_reference_frames": len(ctrl) / (cfg.ref_dt / cfg.sim_dt),
        "full_reference_frames": full_steps / (cfg.ref_dt / cfg.sim_dt),
        "finite_arrays": finite, "tracking": tracking,
        "window_modes": mode_ids, "accepted_rl_windows": mode_ids.count(2),
        "rl_attempted": bool(rl_attempts), "priority_and_restart_verified": True,
        "rl_attempts": [{key: a.get(key) for key in
                         ("window_id", "feasible", "steps_executed", "rl_iterations_used",
                          "rl_iters_allocated", "time_to_farthest", "optimization_elapsed")}
                        for a in rl_attempts],
        "statistics": statistics,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--profile", choices=["codesign", "baseline"], default="codesign")
    parser.add_argument("--runner-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=tuple(MODES), default="switching")
    args = parser.parse_args()
    data_root, output, runner_root = (p.resolve() for p in
                                      (args.data_root, args.output, args.runner_root))
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((data_root / "manifest.json").read_text())["cases"]
    case = next(case for case in cases if case["task"] == args.case)
    recipe = verify_profile(runner_root, args.profile)
    with np.load(data_root / case["processed_dir"] / "0/trajectory_kinematic.npz") as reference:
        reference_frames = len(reference["qpos"])
    metadata = json.loads((data_root / case["processed_dir"] / "task_info.json").read_text())
    command = [sys.executable, "-u", "-m", "action.examples.run_mjwp_mixed",
               "--dataset-dir", str(data_root), "--dataset-name", case["dataset_name"],
               "--robot-type", case["robot_type"], "--embodiment-type", case["embodiment_type"],
               "--active-hand", "right", "--task", case["task"], "--seed", str(args.seed),
               "--device", "cuda:0", "--sim-dt", "0.01", "--ctrl-dt", "0.05",
               "--horizon", "1.6", "--knot-dt", "0.4",
               *mode_options(args.mode, reference_frames, metadata["ref_dt"]),
               *workflow_options(runner_root, args.mode),
               "--mpc-num-samples", "8", "--mpc-max-iters", "2",
               "--rl-num-envs", "1024", "--rl-num-steps-per-env", "16",
               "--rl-total-learning-iterations", "1500",
               "--object-pos-threshold", "0.05", "--object-rot-threshold", "1.0",
               "--debug-output-dir", str(output), "--output-name", "mixed.npz"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runner_root)
    env.setdefault("XDG_CACHE_HOME", str(output / "cache"))
    env["TORCHINDUCTOR_CACHE_DIR"] = str(output / "cache/torchinductor")
    report = {"case": case["task"], "dataset": case["dataset"], "seed": args.seed,
              "profile": args.profile, "mode": args.mode, "command": command[1:], "recipe": recipe,
              "source_sha256": source_fingerprint(runner_root), "python": platform.python_version(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "host": platform.node(),
              "input_sha256": {name: hashlib.sha256((data_root / case["processed_dir"] / name).read_bytes()).hexdigest()
                               for name in ("scene.xml", "scene_eq.xml", "task_info.json", "0/trajectory_kinematic.npz")},
              "execution_status": "running", "validated_complete": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    result = subprocess.run(command, env=env, cwd=runner_root)
    report.update(process_returncode=result.returncode,
                  subprocess_wall_elapsed=time.perf_counter() - started)
    if result.returncode:
        report["execution_status"] = "execution_error"
    else:
        try:
            report.update(validate_artifact(data_root, case, output / "mixed.npz", MODES[args.mode]))
            report["execution_status"] = "completed"
        except Exception as exc:
            report.update(execution_status="validation_error", validation_error=str(exc))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("[CASE]", json.dumps({key: report.get(key) for key in
                               ("case", "profile", "execution_status", "validated_complete",
                                "committed_reference_frames", "full_reference_frames",
                                "accepted_rl_windows")}), flush=True)
    if report["execution_status"] != "completed":
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
