"""Translate constructor spelling across RSL-RL versions without changing PPO settings."""

import inspect
from copy import deepcopy


def adapt_mlp_config(config, model_class=None):
    """Keep legacy configs, or express their policy distribution in RSL-RL 5.3."""
    if model_class is None:
        from rsl_rl.models.mlp_model import MLPModel

        model_class = MLPModel
    result = deepcopy(config)
    if "distribution_cfg" not in inspect.signature(model_class.__init__).parameters:
        return result
    for name in ("actor", "critic"):
        model = result[name]
        if "stochastic" not in model:
            continue
        stochastic = model.pop("stochastic")
        std = model.pop("init_noise_std", 1.0)
        model["distribution_cfg"] = (
            {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": std,
                "std_type": "scalar",
                "learn_std": True,
            }
            if stochastic
            else None
        )
    return result
