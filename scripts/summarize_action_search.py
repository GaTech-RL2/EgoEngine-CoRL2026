"""Keep the complete exploratory cohort and compare only validated, matched full runs."""

import argparse
import csv
import json
from pathlib import Path


def measured_work(report):
    """Copy reported fields verbatim; never infer the author's metric from units."""
    stats = report.get("statistics", {})
    return dict(generation_sim_steps=stats.get("generation_sim_steps"),
                evaluation_sim_steps=stats.get("evaluation_sim_steps"),
                env_steps=stats.get("env_steps"),
                full_physics_steps=report.get("full_physics_steps"),
                committed_physics_steps=report.get("committed_physics_steps"),
                reported_cost=stats.get("cost"),
                reported_cost_including_evaluation=stats.get("cost_including_evaluation"))


def summarize(reports, planned_jobs):
    indexed = {}
    for report in reports:
        identity = (report["case"], report["seed"], report["mode"])
        if identity in indexed:
            raise ValueError(f"Duplicate run: {identity}")
        if report["profile"] != "codesign":
            raise ValueError("This comparison requires the same CoDesign RL recipe")
        indexed[identity] = report
    planned = {(j["case"], j["seed"], j["mode"]) for j in planned_jobs}
    if set(indexed) - planned:
        raise ValueError("A report has no entry in the frozen search/confirmation plans")
    rows = []
    for case, seed, mode in sorted(planned):
        report = indexed.get((case, seed, mode), {})
        stats = report.get("statistics", {})
        complete = report.get("execution_status") == "completed" and report.get("validated_complete", False)
        rows.append(dict(case=case, seed=seed, mode=mode,
                         execution_status=report.get("execution_status", "not_started"),
                         validated_complete=bool(complete),
                         committed_frames=report.get("committed_reference_frames"),
                         full_frames=report.get("full_reference_frames"),
                         legacy_seconds=stats.get("training_elapsed"),
                         optimization_seconds=stats.get("optimization_elapsed"),
                         pipeline_wall_seconds=stats.get("pipeline_wall_elapsed"),
                         **measured_work(report),
                         rl_iterations=stats.get("rl_iterations_used"),
                         host=report.get("host")))
    comparisons = []
    for case, seed in sorted({(c, s) for c, s, _ in planned}):
        switching = indexed.get((case, seed, "switching"))
        pure = indexed.get((case, seed, "pure_rl"))
        if switching is None or pure is None:
            continue
        both_complete = all(r.get("execution_status") == "completed" and r.get("validated_complete")
                            for r in (switching, pure))
        same_inputs = bool(switching.get("input_sha256")) and switching.get("input_sha256") == pure.get("input_sha256")
        same_source = bool(switching.get("source_sha256")) and switching.get("source_sha256") == pure.get("source_sha256")
        new_work, old_work = measured_work(switching), measured_work(pure)
        same_horizon = bool(new_work["full_physics_steps"] and
                            new_work["full_physics_steps"] == old_work["full_physics_steps"] and
                            switching.get("full_reference_frames") == pure.get("full_reference_frames"))
        eligible = bool(both_complete and same_inputs and same_source and same_horizon)
        comparisons.append(dict(case=case, seed=seed, both_validated_complete=bool(both_complete),
                                same_inputs=same_inputs, same_rl_source=same_source,
                                same_horizon=same_horizon,
                                same_host=switching.get("host") == pure.get("host"),
                                eligible=eligible,
                                switching_reported_cost=new_work["reported_cost"],
                                pure_rl_reported_cost=old_work["reported_cost"],
                                cost_ranking_enabled=False))
    aggregate = {}
    for mode in ("replay_mpc", "switching", "pure_rl"):
        selected = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = dict(planned=len(selected),
                               completed=sum(r["execution_status"] == "completed" for r in selected),
                               validated_complete=sum(r["validated_complete"] for r in selected),
                               running=sum(r["execution_status"] == "running" for r in selected),
                               not_started=sum(r["execution_status"] == "not_started" for r in selected),
                               skipped=sum(r["execution_status"].startswith("skipped_") for r in selected),
                               errors=sum(r["execution_status"].endswith("error") for r in selected))
    return dict(scope="Exploratory sample search; screened cases are not an unbiased benchmark sample",
                primary_metric="Use the author's original metric implementation; do not redefine its denominator or exclusions",
                metric_status="Original metric integration under verification; independent normalization withdrawn",
                aggregate=aggregate, rows=rows, comparisons=comparisons,
                candidates_at_least_2x=[],
                replay_mpc_candidates=[dict(case=r["case"], seed=r["seed"]) for r in rows
                                       if r["mode"] == "replay_mpc" and r["validated_complete"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--plan", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = [job for p in args.plan for job in json.loads(p.read_text())["jobs"]]
    reports = [json.loads(p.read_text()) for p in sorted(args.runs.glob("*/report.json"))]
    summary = summarize(reports, jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    print(json.dumps({"aggregate": summary["aggregate"],
                      "candidates_at_least_2x": summary["candidates_at_least_2x"],
                      "replay_mpc_candidates": summary["replay_mpc_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
