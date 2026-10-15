import importlib.util
import unittest
import tempfile
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/benchmark_case.py"
spec = importlib.util.spec_from_file_location("benchmark_case", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
summary_spec = importlib.util.spec_from_file_location(
    "summarize_case_benchmark", path.with_name("summarize_case_benchmark.py")
)
summary_module = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary_module)


class IndependentTrackingTest(unittest.TestCase):
    def test_workflow_uses_full_rl_budget_and_preserves_old_runner_compatibility(self):
        root = path.parents[1]
        self.assertEqual(benchmark.workflow_options(root, 'pure_rl'),
                         ['--rl-window-learning-iterations', '1500'])
        self.assertEqual(benchmark.workflow_options(root, 'switching'),
                         ['--rl-window-learning-iterations', '300'])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'action/examples'
            folder.mkdir(parents=True)
            (folder/'run_mjwp_mixed.py').write_text('class MixedArgs:\n    seed: int = 0\n')
            self.assertEqual(benchmark.workflow_options(Path(tmp), 'pure_rl'), [])

    def test_pure_rl_covers_full_reference_without_chunk_resets(self):
        options = benchmark.mode_options("pure_rl", 209, 0.05)
        self.assertEqual(options, ["--methods", "rl", "--window-size", "209",
                                  "--lookahead-chunks", "1",
                                  "--rl-max-episode-ctrl-steps", "209"])
        benchmark.validate_mode_order([dict(window_id=0, method_name="rl", feasible=True)], ("rl",))
        with self.assertRaises(ValueError):
            benchmark.validate_mode_order([dict(window_id=0, method_name="mpc", feasible=True)], ("rl",))
        self.assertEqual(benchmark.mode_options("switching", 209, 0.05),
                         ["--methods", "replay,mpc,rl", "--window-size", "20",
                          "--lookahead-chunks", "2"])

    def test_comparison_cannot_pair_different_seeds_or_count_execution_errors_as_complete(self):
        def row(profile, seed, status="completed"):
            return dict(case="case", dataset="fixture", profile=profile, seed=seed,
                        execution_status=status, validated_complete=status == "completed")
        result = summary_module.summarize([row("baseline", 0), row("codesign", 1)],
                                          ["case"], (0, 1))
        self.assertEqual(result["matched_cases"], [])
        self.assertEqual(result["aggregate"]["codesign"]["planned_runs"], 2)
        result = summary_module.summarize([row("baseline", 0), row("codesign", 0, "execution_error")],
                                          ["case"])
        self.assertEqual(result["aggregate"]["codesign"]["validated_full_completions"], 0)
        self.assertEqual(result["aggregate"]["codesign"]["execution_or_validation_errors"], 1)

    def test_post_step_phase_and_quaternion_sign(self):
        reference = np.zeros((4, 14))
        reference[:, 7] = np.arange(4) * 0.1
        reference[:, 10] = 1
        actual = reference[1:].copy()
        actual[:, 10] = -1
        result = benchmark.tracking_checks(actual, reference, slice(7, 14))
        self.assertTrue(result["within_limits"])
        self.assertEqual(result["max_position_error_m"], 0)
        self.assertEqual(result["max_orientation_error_rad"], 0)

    def test_invalid_quaternion_and_one_failed_step_cannot_pass(self):
        reference = np.zeros((4, 7))
        reference[:, 3] = 1
        actual = reference[1:].copy()
        actual[1, 0] = 0.051
        result = benchmark.tracking_checks(actual, reference, slice(0, 7))
        self.assertFalse(result["within_limits"])
        self.assertEqual(result["violating_steps"], 1)
        actual[1, 0] = 0
        actual[1, 3] = 0
        result = benchmark.tracking_checks(actual, reference, slice(0, 7))
        self.assertFalse(result["within_limits"])
        self.assertFalse(result["valid_quaternions"])


if __name__ == "__main__":
    unittest.main()
