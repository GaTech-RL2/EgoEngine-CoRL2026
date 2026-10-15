"""Run an exploratory switching candidate and its full-RL control on one GPU."""

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
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    jobs = plan["jobs"][2 * index:2 * index + 2]
    if len(jobs) != 2 or [j["mode"] for j in jobs] != ["switching", "pure_rl"]:
        raise ValueError("Expected a switching / Pure RL pair")
    if (jobs[0]["case"], jobs[0]["seed"]) != (jobs[1]["case"], jobs[1]["seed"]):
        raise ValueError("Pair inputs and seed must match")
    for relative, expected in plan["source_sha256"].items():
        if hashlib.sha256((Path(plan["runner_root"]) / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen action source changed: {relative}")
    previous = None
    for job in jobs:
        output = Path(plan["output_root"]) / f"codesign__{job['mode']}__{job['case']}__seed{job['seed']}"
        if (output / "report.json").exists():
            raise FileExistsError(output)
        if previous is not None and not previous.get("validated_complete", False):
            output.mkdir(parents=True)
            report = dict(case=job["case"], seed=job["seed"], mode=job["mode"], profile="codesign",
                          dataset=previous["dataset"], execution_status="skipped_switching_failed",
                          validated_complete=False, reason="Exploratory selection: switching did not complete",
                          slurm_job_id=os.environ.get("SLURM_JOB_ID"))
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("[PAIR SKIP]", json.dumps(report), flush=True)
            continue
        command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_case.py")),
                   "--data-root", plan["data_root"], "--runner-root", plan["runner_root"],
                   "--output", str(output), "--case", job["case"], "--profile", "codesign",
                   "--mode", job["mode"], "--seed", str(job["seed"])]
        print("[PAIR]", json.dumps(job), flush=True)
        subprocess.run(command, check=True)
        previous = json.loads((output / "report.json").read_text())


if __name__ == "__main__":
    main()
