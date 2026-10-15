from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import tyro

from paper_results_analysis.analyze_mixed_results import (
    Args as LoadArgs,
    MixedResult,
    _resolve_inputs,
    load_mixed_result,
)


WINDOW_COLORS = {
    "replay": "#1E88E5",
    "rl": "#4CAF50",
    "mpc": "#FB8C00",
    "unsolved": "#9E9E9E",
}


@dataclass
class Args:
    """Plot mixed-result windows as a color grid."""

    inputs: tuple[Path, ...] = ()
    glob: str = ""
    recursive: bool = False
    num_of_window: int = 10
    cell_size: float = 0.8
    output_dir: Path = Path("paper_results_analysis/window_plots")
    dpi: int = 200
    show: bool = False


def _window_solved(window) -> bool:
    return (
        not bool(window.terminated)
        and int(window.steps_executed) >= int(window.end_step - window.start_step)
    )


def _color_for_window(window) -> str:
    if not _window_solved(window):
        return WINDOW_COLORS["unsolved"]
    if window.method_name == "replay":
        return WINDOW_COLORS["replay"]
    if window.method_name == "rl":
        return WINDOW_COLORS["rl"]
    if window.method_name == "mpc":
        return WINDOW_COLORS["mpc"]
    return WINDOW_COLORS["unsolved"]


def _resolve_results(args: Args) -> list[MixedResult]:
    load_args = LoadArgs(inputs=args.inputs, glob=args.glob, recursive=args.recursive)
    return [load_mixed_result(path) for path in _resolve_inputs(load_args)]


def _build_output_path(output_dir: Path, result: MixedResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{result.path.stem}_window_grid.png"


def _plot_result(result: MixedResult, *, num_of_window: int, cell_size: float) -> plt.Figure:
    if num_of_window <= 0:
        raise ValueError(f"num_of_window must be positive, got {num_of_window}")

    num_windows = result.num_windows
    num_rows = max(1, math.ceil(num_windows / num_of_window))
    fig_width = max(4.0, num_of_window * cell_size + 2.5)
    fig_height = max(3.0, num_rows * cell_size + 2.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for window in result.windows:
        row = window.index // num_of_window
        col = window.index % num_of_window
        y = num_rows - 1 - row
        rect = Rectangle(
            (col, y),
            1.0,
            1.0,
            facecolor=_color_for_window(window),
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            col + 0.5,
            y + 0.5,
            str(window.index),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlim(0, num_of_window)
    ax.set_ylim(0, num_rows)
    ax.set_aspect("equal")
    ax.set_xticks(range(num_of_window))
    ax.set_yticks(range(num_rows))
    ax.set_xticklabels([str(index) for index in range(num_of_window)])
    ax.set_yticklabels([str(index) for index in reversed(range(num_rows))])
    ax.set_xlabel("Window Column")
    ax.set_ylabel("Grid Row")
    ax.set_title(f"{result.path.name}\n{num_windows} windows")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=WINDOW_COLORS["replay"], label="Solved by Replay"),
        Patch(facecolor=WINDOW_COLORS["rl"], label="Solved by RL"),
        Patch(facecolor=WINDOW_COLORS["mpc"], label="Solved by MPC"),
        Patch(facecolor=WINDOW_COLORS["unsolved"], label="Unsolved"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)

    fig.tight_layout()
    return fig


def main(args: Args) -> None:
    results = _resolve_results(args)

    print(f"Loaded {len(results)} result file(s).")
    for result in results:
        fig = _plot_result(
            result,
            num_of_window=args.num_of_window,
            cell_size=args.cell_size,
        )

        output_path = _build_output_path(args.output_dir.expanduser().resolve(), result)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

        if args.show:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(Args))
