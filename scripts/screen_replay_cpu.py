"""Use the retained replay runner on CPU to prioritize later GPU confirmations.

CPU results are discovery hints, never GPU success or speedup measurements.
"""

import argparse
import gc
import json
from pathlib import Path

from benchmark_case import validate_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    from action.examples.run_mjwp_mixed import MixedArgs, main as replay

    cases = json.loads((args.data_root / "manifest.json").read_text())["cases"]
    features = {r["case"]: r for r in json.loads(args.features.read_text())}
    cases.sort(key=lambda c: features[c["task"]]["first_motion_frame"] or 0, reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        destination = args.output / case["task"]
        if destination.exists():
            raise FileExistsError(destination)
        row = dict(case=case["task"], dataset=case["dataset"], backend="cpu",
                   purpose="Candidate discovery only; GPU confirmation required")
        try:
            replay(MixedArgs(dataset_dir=str(args.data_root), dataset_name=case["dataset_name"],
                             robot_type=case["robot_type"], embodiment_type=case["embodiment_type"],
                             task=case["task"], device="cpu", methods="replay", seed=0,
                             debug_output_dir=str(destination), output_name="mixed.npz"))
            report = validate_artifact(args.data_root, case, destination / "mixed.npz", ("replay",))
            row.update(execution_status="completed", validated_complete=report["validated_complete"],
                       committed_frames=report["committed_reference_frames"],
                       full_frames=report["full_reference_frames"], tracking=report["tracking"])
        except Exception as exc:
            row.update(execution_status="error", error=f"{type(exc).__name__}: {exc}",
                       validated_complete=False)
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("[CPU DISCOVERY]", json.dumps(row), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
