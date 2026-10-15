"""Plot policy validation progress from an actual mixed trajectory."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/rl_learning.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent.resolve() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.trajectory, allow_pickle=False) as data:
        attempts = json.loads(data["attempts_json"].item())
        complete = bool(data["success"].item())
    windows = [row for row in attempts if row.get("learning_trace")]
    if not windows:
        raise ValueError("No RL policy learning trace in this artifact")
    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 3.5), squeeze=False)
    for ax, window in zip(axes[0], windows):
        trace = window["learning_trace"]
        iterations = [row["iteration"] for row in trace]
        steps = [row["valid_steps"] for row in trace]
        target = window["plan_end_step"] - window["start_step"]
        ax.plot(iterations, steps, color="#9cb8dc", linewidth=1, label="Evaluated policy")
        ax.plot(iterations, np.maximum.accumulate(steps), color="#1764a2", label="Best valid progress")
        ax.axhline(target, color="#28794e", linestyle="--", label="Required full plan")
        ax.set(title=f"RL chunk {window['window_id']} — {'passed' if window['feasible'] else 'failed'}",
               xlabel="PPO iteration", ylabel="Valid physics steps", ylim=(0, target * 1.08))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"Policy evaluation during RL training | full trajectory {'complete' if complete else 'incomplete'}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
