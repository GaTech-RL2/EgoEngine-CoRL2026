"""Summarize completed, failed and unfinished matched-budget case runs."""

import argparse
import csv
import json
from pathlib import Path


def summarize(reports, case_names, seeds=(0,)):
    rows = []
    for report in reports:
        stats = report.get("statistics", {})
        rows.append({
            "case": report["case"], "dataset": report["dataset"],
            "profile": report["profile"], "seed": report["seed"],
            "execution_status": report["execution_status"],
            "validated_complete": report.get("validated_complete", False),
            "nominal_success": report.get("nominal_success"),
            "committed_frames": report.get("committed_reference_frames"),
            "full_frames": report.get("full_reference_frames"),
            "accepted_rl_windows": report.get("accepted_rl_windows"),
            "rl_iterations_used": stats.get("rl_iterations_used"),
            "legacy_training_seconds": stats.get("training_elapsed"),
            "complete_optimization_seconds": stats.get("optimization_elapsed"),
            "generation_sim_steps": stats.get("generation_sim_steps"),
            "evaluation_sim_steps": stats.get("evaluation_sim_steps"),
            "max_position_error_m": report.get("tracking", {}).get("max_position_error_m"),
            "max_orientation_error_rad": report.get("tracking", {}).get("max_orientation_error_rad"),
        })
    identities = [(r["case"], r["profile"], r["seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate case/profile/seed reports")
    aggregates = {}
    for profile in ("baseline", "codesign"):
        selected = [r for r in rows if r["profile"] == profile]
        completed = [r for r in selected if r["execution_status"] == "completed"]
        aggregates[profile] = {
            "planned_cases": len(case_names), "planned_runs": len(case_names) * len(seeds),
            "reported_runs": len(selected),
            "completed_runs": len(completed),
            "validated_full_completions": sum(r["validated_complete"] for r in completed),
            "execution_or_validation_errors": sum(r["execution_status"].endswith("error")
                                                  for r in selected),
            "running": sum(r["execution_status"] == "running" for r in selected),
            "missing_runs": [{"case": case, "seed": seed} for case in case_names for seed in seeds
                             if not any(r["case"] == case and r["seed"] == seed for r in selected)],
        }
    pairs = []
    for case, seed in ((case, seed) for case in case_names for seed in seeds):
        selected = {r["profile"]: r for r in rows if r["case"] == case and r["seed"] == seed}
        if any(p not in selected or selected[p]["execution_status"] != "completed"
               for p in ("baseline", "codesign")):
            continue
        old, new = selected["baseline"], selected["codesign"]
        ratio = None
        if old["validated_complete"] and new["validated_complete"]:
            before, after = old["legacy_training_seconds"], new["legacy_training_seconds"]
            if before is not None and after is not None and after > 0:
                ratio = before / after
        pairs.append({"case": case, "seed": old["seed"],
                      "baseline_complete": old["validated_complete"],
                      "codesign_complete": new["validated_complete"],
                      "legacy_time_ratio_when_both_complete": ratio})
    return {"aggregate": aggregates, "matched_cases": pairs,
            "rows": sorted(rows, key=lambda r: (r["dataset"], r["case"], r["profile"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in sorted(args.runs.glob("*/report.json"))]
    names = [row["task"] for row in json.loads(args.manifest.read_text())["cases"]]
    summary = summarize(reports, names, args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if summary["rows"]:
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
            writer.writeheader()
            writer.writerows(summary["rows"])
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
