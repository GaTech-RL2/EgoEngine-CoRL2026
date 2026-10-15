from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

METHOD_TO_ID = {"replay": 0, "mpc": 1, "rl": 2}
ID_TO_METHOD = {value: key for key, value in METHOD_TO_ID.items()}

REQUIRED_KEYS = (
    "ctrl",
    "reward",
    "error",
    "method_id",
    "window_start",
    "window_end",
    "window_method_id",
    "window_steps_executed",
    "window_terminated",
    "window_mean_error",
    "window_final_error",
    "window_total_reward",
    "window_rl_iters_allocated",
)


@dataclass
class WindowSummary:
    index: int
    start_step: int
    end_step: int
    method_id: int
    method_name: str
    steps_executed: int
    terminated: bool
    mean_error: float
    final_error: float
    total_reward: float
    rl_iters_allocated: int


@dataclass
class MixedResult:
    path: Path
    ctrl: np.ndarray
    reward: np.ndarray
    error: np.ndarray
    method_id: np.ndarray
    windows: list[WindowSummary]
    success: bool | None = None

    @property
    def num_steps(self) -> int:
        return int(self.ctrl.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.ctrl.shape[1]) if self.ctrl.ndim == 2 else 0

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def terminated_early(self) -> bool:
        if self.success is not None:
            return not self.success
        return any(window.terminated for window in self.windows)

    @property
    def final_error(self) -> float:
        if self.error.size == 0:
            return float("nan")
        return float(self.error[-1])

    @property
    def mean_error(self) -> float:
        if self.error.size == 0:
            return float("nan")
        return float(np.mean(self.error))

    @property
    def total_reward(self) -> float:
        if self.reward.size == 0:
            return 0.0
        return float(np.sum(self.reward))

    @property
    def mean_reward(self) -> float:
        if self.reward.size == 0:
            return float("nan")
        return float(np.mean(self.reward))


@dataclass
class Args:
    """Load one or more mixed-result NPZ files and summarize them."""

    inputs: tuple[Path, ...] = ()
    glob: str = ""
    recursive: bool = False
    print_window_details: bool = True
    csv_out: Path | None = None


def _resolve_inputs(args: Args) -> list[Path]:
    paths = {path.expanduser().resolve() for path in args.inputs}
    if args.glob:
        pattern = args.glob
        root = Path(".")
        matches = root.rglob(pattern) if args.recursive else root.glob(pattern)
        paths.update(match.resolve() for match in matches if match.is_file())

    if not paths:
        default_path = Path("results/mixed_replay_smoke_debug.npz").resolve()
        if default_path.is_file():
            return [default_path]
        raise FileNotFoundError(
            "No input files were provided. Pass --inputs ... or --glob ... ."
        )

    resolved = sorted(paths)
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    return resolved


def _validate_payload(path: Path, payload: np.lib.npyio.NpzFile) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")


def _method_name(method_id: int) -> str:
    return ID_TO_METHOD.get(int(method_id), f"unknown({int(method_id)})")


def load_mixed_result(path: Path) -> MixedResult:
    with np.load(path, allow_pickle=True) as payload:
        _validate_payload(path, payload)

        ctrl = np.asarray(payload["ctrl"])
        reward = np.asarray(payload["reward"])
        error = np.asarray(payload["error"])
        method_id = np.asarray(payload["method_id"])

        window_start = np.asarray(payload["window_start"])
        window_end = np.asarray(payload["window_end"])
        window_method_id = np.asarray(payload["window_method_id"])
        window_steps_executed = np.asarray(payload["window_steps_executed"])
        window_terminated = np.asarray(payload["window_terminated"])
        window_mean_error = np.asarray(payload["window_mean_error"])
        window_final_error = np.asarray(payload["window_final_error"])
        window_total_reward = np.asarray(payload["window_total_reward"])
        window_rl_iters_allocated = np.asarray(payload["window_rl_iters_allocated"])
        success = bool(payload["success"].item()) if "success" in payload.files else None

    num_windows = len(window_start)
    expected_lengths = {
        "window_end": len(window_end),
        "window_method_id": len(window_method_id),
        "window_steps_executed": len(window_steps_executed),
        "window_terminated": len(window_terminated),
        "window_mean_error": len(window_mean_error),
        "window_final_error": len(window_final_error),
        "window_total_reward": len(window_total_reward),
        "window_rl_iters_allocated": len(window_rl_iters_allocated),
    }
    inconsistent = {
        key: value for key, value in expected_lengths.items() if value != num_windows
    }
    if inconsistent:
        raise ValueError(
            f"{path} has inconsistent window array lengths: {inconsistent}, "
            f"expected {num_windows}"
        )

    if ctrl.ndim != 2:
        raise ValueError(f"{path} expected ctrl to be rank-2, got shape {ctrl.shape}")
    if reward.shape != (ctrl.shape[0],):
        raise ValueError(
            f"{path} expected reward shape {(ctrl.shape[0],)}, got {reward.shape}"
        )
    if error.shape != (ctrl.shape[0],):
        raise ValueError(
            f"{path} expected error shape {(ctrl.shape[0],)}, got {error.shape}"
        )
    if method_id.shape != (ctrl.shape[0],):
        raise ValueError(
            f"{path} expected method_id shape {(ctrl.shape[0],)}, got {method_id.shape}"
        )

    windows = [
        WindowSummary(
            index=index,
            start_step=int(window_start[index]),
            end_step=int(window_end[index]),
            method_id=int(window_method_id[index]),
            method_name=_method_name(int(window_method_id[index])),
            steps_executed=int(window_steps_executed[index]),
            terminated=bool(window_terminated[index]),
            mean_error=float(window_mean_error[index]),
            final_error=float(window_final_error[index]),
            total_reward=float(window_total_reward[index]),
            rl_iters_allocated=int(window_rl_iters_allocated[index]),
        )
        for index in range(num_windows)
    ]

    return MixedResult(
        path=path,
        ctrl=ctrl,
        reward=reward,
        error=error,
        method_id=method_id,
        windows=windows,
        success=success,
    )


def _format_method_counts(method_ids: np.ndarray) -> str:
    if method_ids.size == 0:
        return "none"
    method_ids_int = np.asarray(method_ids, dtype=np.int64)
    unique_ids, counts = np.unique(method_ids_int, return_counts=True)
    return ", ".join(
        f"{_method_name(int(method))}:{int(count)}"
        for method, count in zip(unique_ids, counts, strict=True)
    )


def _summary_row(result: MixedResult) -> dict[str, object]:
    return {
        "path": str(result.path),
        "num_steps": result.num_steps,
        "action_dim": result.action_dim,
        "num_windows": result.num_windows,
        "terminated_early": result.terminated_early,
        "total_reward": result.total_reward,
        "mean_reward": result.mean_reward,
        "mean_error": result.mean_error,
        "final_error": result.final_error,
        "step_method_counts": _format_method_counts(result.method_id),
        "window_method_counts": _format_method_counts(
            np.asarray([window.method_id for window in result.windows], dtype=np.int64)
        ),
    }


def _window_rows(result: MixedResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in result.windows:
        rows.append(
            {
                "path": str(result.path),
                "window_index": window.index,
                "start_step": window.start_step,
                "end_step": window.end_step,
                "method_id": window.method_id,
                "method_name": window.method_name,
                "steps_executed": window.steps_executed,
                "terminated": window.terminated,
                "mean_error": window.mean_error,
                "final_error": window.final_error,
                "total_reward": window.total_reward,
                "rl_iters_allocated": window.rl_iters_allocated,
            }
        )
    return rows


def _write_csv(csv_out: Path, results: list[MixedResult]) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "row_type",
        "path",
        "num_steps",
        "action_dim",
        "num_windows",
        "terminated_early",
        "total_reward",
        "mean_reward",
        "mean_error",
        "final_error",
        "step_method_counts",
        "window_method_counts",
        "window_index",
        "start_step",
        "end_step",
        "method_id",
        "method_name",
        "steps_executed",
        "terminated",
        "rl_iters_allocated",
    ]

    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=summary_fields, extrasaction="ignore"
        )
        writer.writeheader()

        for result in results:
            summary_row = _summary_row(result)
            summary_row["row_type"] = "summary"
            writer.writerow(summary_row)

            for window_row in _window_rows(result):
                window_row["row_type"] = "window"
                writer.writerow(window_row)


def _print_result(result: MixedResult, *, print_window_details: bool) -> None:
    summary = _summary_row(result)
    print(f"\n== {result.path} ==")
    print(
        "steps={num_steps}, action_dim={action_dim}, windows={num_windows}, "
        "terminated_early={terminated_early}".format(**summary)
    )
    print(
        "reward(total={total_reward:.6f}, mean={mean_reward:.6f}), "
        "error(mean={mean_error:.6f}, final={final_error:.6f})".format(**summary)
    )
    print(f"step methods: {summary['step_method_counts']}")
    print(f"window methods: {summary['window_method_counts']}")

    if not print_window_details:
        return

    print("window details:")
    for window in result.windows:
        print(
            "  "
            f"[{window.index}] steps={window.start_step}:{window.end_step} "
            f"method={window.method_name} executed={window.steps_executed} "
            f"terminated={window.terminated} mean_error={window.mean_error:.6f} "
            f"final_error={window.final_error:.6f} total_reward={window.total_reward:.6f} "
            f"rl_iters={window.rl_iters_allocated}"
        )


def main(args: Args) -> None:
    inputs = _resolve_inputs(args)
    results = [load_mixed_result(path) for path in inputs]

    print(f"Loaded {len(results)} result file(s).")
    for result in results:
        _print_result(result, print_window_details=bool(args.print_window_details))

    if args.csv_out is not None:
        csv_out = args.csv_out.expanduser().resolve()
        _write_csv(csv_out, results)
        print(f"\nWrote CSV summary to {csv_out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
