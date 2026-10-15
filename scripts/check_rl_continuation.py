"""Exercise native RSL checkpoint continuation on a five-frame CUDA API fixture.

This cropped fixture and its loose limits are not task-quality or speedup evidence.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from benchmark_case import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from action.examples.run_mjwp_mixed import MixedArgs, main as run

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["task"] == "mustard_1")
    data_root = output / "fixture"
    source = args.data_root / case["processed_dir"]
    target = data_root / case["processed_dir"]
    (target / "0").mkdir(parents=True)
    (data_root / "assets").symlink_to((args.data_root / "assets").resolve(), target_is_directory=True)
    for name in ("scene.xml", "scene_eq.xml", "task_info.json"):
        shutil.copyfile(source / name, target / name)
    with np.load(source / "0/trajectory_kinematic.npz", allow_pickle=False) as reference:
        frames = len(reference["qpos"])
        values = {k: reference[k][:5].copy() if reference[k].ndim and len(reference[k]) == frames
                  else reference[k].copy() for k in reference.files}
    np.savez(target / "0/trajectory_kinematic.npz", **values)
    run(MixedArgs(dataset_dir=str(data_root), dataset_name="aria", robot_type="xhand",
                  embodiment_type="right", task="mustard_1", device="cuda:0", methods="rl",
                  window_size=2, lookahead_chunks=2, rl_num_envs=32,
                  rl_total_learning_iterations=6, rl_window_learning_iterations=2,
                  object_pos_threshold=0.5, object_rot_threshold=3.0,
                  debug_output_dir=str(output / "runner"), output_name="mixed.npz"))
    with np.load(output / "runner/mixed.npz", allow_pickle=False) as artifact:
        if not bool(artifact["success"].item()):
            raise RuntimeError("API fixture did not complete")
        stats = json.loads(artifact["statistics_json"].item())
        attempts = json.loads(artifact["attempts_json"].item())
        if not all(np.isfinite(artifact[k]).all() for k in ("ctrl", "qpos", "qvel")):
            raise RuntimeError("Nonfinite API fixture outputs")
    if stats["rl_warm_started_windows"] != 2 or len(attempts) != 3:
        raise RuntimeError("Expected two checkpoint continuations across three windows")
    optimizer_steps = []
    for attempt in attempts:
        state = torch.load(attempt["continuation_checkpoint"], map_location="cpu", weights_only=False)
        key = next(k for k in state if "optimizer" in k)
        optimizer_steps.append(max(float(s["step"]) for s in state[key]["state"].values() if "step" in s))
        if state["iter"] != attempt["rl_iterations_used"] - 1:
            raise RuntimeError("A window inherited the prior iteration counter")
    if not all(b > a for a, b in zip(optimizer_steps, optimizer_steps[1:])):
        raise RuntimeError("Optimizer state did not continue across windows")
    report = dict(api_passed=True, scope=__doc__, frames=5, worlds=32,
                  optimizer_steps=optimizer_steps, statistics=stats,
                  source_sha256=source_fingerprint(args.runner_root))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[NATIVE CONTINUATION]", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
