import ast
import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from spider.rl.rsl_compat import adapt_mlp_config


class CompatibilityTest(unittest.TestCase):
    def test_codesign_recipe_changes_are_explicit_against_imported_baseline(self):
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root / "action/examples/run_mjwp_mixed.py").read_text())
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "_make_rsl_train_cfg"
        )
        ns = {"Any": Any, "math": math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "recipe", "exec"), ns)
        expected = json.loads((root / "docs/upstream/RL_RECIPE.json").read_text())
        actual = ns["_make_rsl_train_cfg"](16, 300, 0)
        expected["algorithm"]["entropy_coef"] = 0.001
        expected["actor"].pop("stochastic")
        expected["actor"].pop("init_noise_std")
        expected["actor"]["distribution_cfg"] = {
            "class_name": "spider.rl.residual_distribution:ResidualGaussian",
            "init_std": math.exp(-2), "std_type": "log",
            "std_range": [math.exp(-5), math.exp(-1)], "learn_std": True,
        }
        self.assertEqual(actual, expected)

    def recipe(self):
        return {
            "algorithm": {"learning_rate": 3e-4, "gamma": 0.99, "num_learning_epochs": 5},
            "actor": {
                "hidden_dims": [256, 128],
                "activation": "elu",
                "obs_normalization": True,
                "stochastic": True,
                "init_noise_std": 1.0,
            },
            "critic": {
                "hidden_dims": [256, 128],
                "activation": "elu",
                "obs_normalization": True,
                "stochastic": False,
            },
        }

    def test_modern_constructor_keeps_recipe_and_gaussian_scale(self):
        class Modern:
            def __init__(self, distribution_cfg=None):
                pass

        config = self.recipe()
        before = deepcopy(config)
        result = adapt_mlp_config(config, Modern)
        self.assertEqual(config, before)
        self.assertEqual(result["algorithm"], config["algorithm"])
        self.assertEqual(result["actor"]["hidden_dims"], [256, 128])
        self.assertEqual(result["actor"]["distribution_cfg"]["init_std"], 1.0)
        self.assertEqual(result["actor"]["distribution_cfg"]["std_type"], "scalar")
        self.assertIsNone(result["critic"]["distribution_cfg"])

    def test_legacy_constructor_retains_original_keys(self):
        class Legacy:
            def __init__(self, stochastic=False, init_noise_std=1):
                pass

        config = self.recipe()
        self.assertEqual(adapt_mlp_config(config, Legacy), config)


if __name__ == "__main__":
    unittest.main()
