import importlib.util
import unittest
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "scripts/summarize_action_search.py"
spec = importlib.util.spec_from_file_location("summarize_action_search", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SearchComparisonTest(unittest.TestCase):
    def reports(self):
        return [dict(case="fixture", seed=0, mode=mode, profile="codesign",
                     execution_status="completed", validated_complete=True,
                     full_physics_steps=500, committed_physics_steps=500,
                     full_reference_frames=100, committed_reference_frames=100,
                     input_sha256={"data": "unchanged"}, source_sha256={"rl": "unchanged"},
                     statistics={"training_elapsed": seconds, "generation_sim_steps": 1000,
                                 "evaluation_sim_steps": 12, "cost": cost, "env_steps": 57})
                for mode, seconds, cost in (("switching", 0, 37.125), ("pure_rl", 100, 800))]

    def check(self, reports):
        plan = [{k: r[k] for k in ("case", "seed", "mode")} for r in reports]
        return module.summarize(reports, plan)

    def test_reported_metric_is_preserved_without_recomputing_denominator(self):
        result = self.check(self.reports())
        row = next(r for r in result["rows"] if r["mode"] == "switching")
        self.assertEqual(row["reported_cost"], 37.125)
        self.assertEqual(row["env_steps"], 57)
        self.assertNotIn("simstep_cost", row)
        self.assertNotIn("paired_success_cost", result)

    def test_missing_metric_is_not_inferred_from_physics_work_or_iterations(self):
        reports = self.reports()
        del reports[0]["statistics"]["cost"]
        del reports[0]["statistics"]["env_steps"]
        row = next(r for r in self.check(reports)["rows"] if r["mode"] == "switching")
        self.assertIsNone(row["reported_cost"])
        self.assertIsNone(row["env_steps"])

    def test_large_raw_cost_or_seconds_ratio_does_not_enable_unverified_ranking(self):
        result = self.check(self.reports())
        self.assertTrue(result["comparisons"][0]["eligible"])
        self.assertFalse(result["comparisons"][0]["cost_ranking_enabled"])
        self.assertFalse(result["candidates_at_least_2x"])

    def test_failure_and_changed_source_input_or_horizon_cannot_be_paired(self):
        for key, value in (("validated_complete", False), ("input_sha256", {"changed": "value"}),
                           ("source_sha256", {"changed": "value"}), ("full_physics_steps", 501)):
            reports = self.reports()
            reports[0][key] = value
            self.assertFalse(self.check(reports)["comparisons"][0]["eligible"])

    def test_screening_is_not_switching_evidence(self):
        reports = self.reports()
        reports[0]["mode"] = "replay_mpc"
        result = self.check(reports)
        self.assertFalse(result["comparisons"])
        self.assertEqual(len(result["replay_mpc_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
