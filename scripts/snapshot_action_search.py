"""Snapshot reports and completed trajectories without modifying frozen runners."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

from summarize_action_search import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plan", nargs="+", required=True)
    parser.add_argument("--include-policies", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plans = [root / name for name in args.plan]
    jobs = [j for p in plans for j in json.loads(p.read_text())["jobs"]]
    raw = []
    for path in sorted((root / "runs").glob("*/report.json")):
        content = path.read_bytes()
        try:
            report = json.loads(content)
        except json.JSONDecodeError:
            continue  # An in-progress write is retried by the next snapshot.
        raw.append((path, content, report))
    summary = summarize([r for _, _, r in raw], jobs)
    destination = root / "summary"
    destination.mkdir(exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    provenance = {"postprocessor_sha256": hashlib.sha256(
        Path(__file__).with_name("summarize_action_search.py").read_bytes()).hexdigest(),
        "primary_metric": summary["primary_metric"],
        "frozen_artifacts_modified": False}
    (destination / "accounting.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with tarfile.open(root / "results_snapshot.tar.gz", "w:gz") as archive:
        for path, content, report in raw:
            info = tarfile.TarInfo(str(path.relative_to(root)))
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
            if report["execution_status"] == "completed":
                for name in ("mixed.npz", "mixed.json"):
                    artifact = path.parent / name
                    if artifact.is_file():
                        archive.add(artifact, arcname=artifact.relative_to(root))
            if args.include_policies and report.get("validated_complete"):
                for artifact in sorted(path.parent.glob("_mixed_rl/window_*/policy_*.pt")):
                    archive.add(artifact, arcname=artifact.relative_to(root))
        files = plans + list(destination.glob("*"))
        files += list((root / "runs").glob("*/_mixed_rl/window_*/learning_trace.json"))
        files += list((root / "infrastructure_errors").rglob("*"))
        files += [root / name for name in ("run_registry.json", "input_equivalence.json",
                                           "reference_features.json", "cpu_discovery/summary.json",
                                           "native_api/report.json")]
        for path in files:
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root))
    print(json.dumps({"aggregate": summary["aggregate"],
                      "metric_status": summary["metric_status"],
                      "candidates_at_least_2x": summary["candidates_at_least_2x"]}, indent=2))


if __name__ == "__main__":
    main()
