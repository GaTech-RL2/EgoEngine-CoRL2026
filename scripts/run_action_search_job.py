"""Dispatch one immutable action-search plan entry (also usable as a Slurm array)."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    job = plan["jobs"][index]
    for relative, expected in plan["source_sha256"].items():
        actual = hashlib.sha256((Path(plan["runner_root"]) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen action source changed: {relative}")
    name = f"codesign__{job['mode']}__{job['case']}__seed{job['seed']}"
    destination = Path(plan["output_root"]) / name
    if (destination / "report.json").exists():
        raise FileExistsError(f"Refusing to overwrite a previous run: {destination}")
    command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_case.py")),
               "--data-root", plan["data_root"], "--runner-root", plan["runner_root"],
               "--output", str(destination), "--case", job["case"],
               "--profile", "codesign", "--mode", job["mode"], "--seed", str(job["seed"])]
    print("[SEARCH]", json.dumps({"index": index, "job": job, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
