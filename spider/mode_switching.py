"""Chunk-level mode selection and generation accounting, without a GPU dependency."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Callable


@dataclass(frozen=True)
class PlanningWindow:
    index: int
    start: int
    execute_end: int
    plan_end: int


def planning_windows(
    total_steps: int, control_steps: int, chunk_size: int = 20, lookahead_chunks: int = 2
) -> list[PlanningWindow]:
    """Convert control-step chunks to simulator indices, including the short tail."""
    if total_steps < 0 or control_steps <= 0 or chunk_size <= 0 or lookahead_chunks <= 0:
        raise ValueError("Invalid trajectory length, timestep ratio, or chunk size")
    width = control_steps * chunk_size
    return [
        PlanningWindow(
            i,
            start,
            min(start + width, total_steps),
            min(start + width * lookahead_chunks, total_steps),
        )
        for i, start in enumerate(range(0, total_steps, width))
    ]


def select_mode(
    methods: list[str],
    reset: Callable[[], None],
    solve: Callable[[str], Any],
    feasible: Callable[[Any], bool],
    commit: Callable[[Any], None],
) -> tuple[Any | None, list[Any]]:
    """Probe from one boundary; commit a feasible prefix or restore that boundary."""
    attempts = []
    for method in methods:
        reset()
        try:
            trial = solve(method)
        except BaseException:
            reset()
            raise
        attempts.append(trial)
        try:
            if feasible(trial):
                commit(trial)
                return trial, attempts
        except BaseException:
            reset()
            raise
    reset()
    return None, attempts


@dataclass
class WorkLedger:
    """Count actual world-physics steps; keep benchmark-excluded evaluation visible."""

    generation: dict[str, int] = field(default_factory=dict)
    evaluation: dict[str, int] = field(default_factory=dict)
    vector_generation: dict[str, int] = field(default_factory=dict)
    phase: str = "generation"

    def record(self, worlds: int, mode: str):
        bucket = self.evaluation if self.phase == "evaluation" else self.generation
        bucket[mode] = bucket.get(mode, 0) + int(worlds)
        if self.phase == "generation":
            self.vector_generation[mode] = self.vector_generation.get(mode, 0) + 1

    @contextmanager
    def evaluating(self):
        previous = self.phase
        self.phase = "evaluation"
        try:
            yield
        finally:
            self.phase = previous

    def snapshot(self):
        return {
            "generation_sim_steps": sum(self.generation.values()),
            "evaluation_sim_steps": sum(self.evaluation.values()),
            "generation_by_mode": dict(self.generation),
            "evaluation_by_mode": dict(self.evaluation),
            "vector_generation_by_mode": dict(self.vector_generation),
        }


class OptimizationClock:
    """Legacy optimization-only wall time, with synchronized evaluation exclusions."""

    def __init__(self, synchronize=lambda: None, now=time.perf_counter):
        self.synchronize = synchronize
        self.now = now
        self.excluded = 0.0
        self._exclude_depth = 0
        self.synchronize()
        self.start = self.now()

    def elapsed(self):
        self.synchronize()
        return max(0.0, self.now() - self.start - self.excluded)

    @contextmanager
    def exclude(self):
        outer = self._exclude_depth == 0
        if outer:
            self.synchronize()
            begin = self.now()
        self._exclude_depth += 1
        try:
            yield
        finally:
            self._exclude_depth -= 1
            if outer:
                self.synchronize()
                self.excluded += self.now() - begin


@dataclass
class FarthestProgress:
    steps: int = 0
    elapsed: float = 0.0

    def observe(self, steps: int, elapsed: float):
        # A reward improvement at the same progress must not overwrite its first time.
        if steps > self.steps:
            self.steps, self.elapsed = int(steps), float(elapsed)


@dataclass
class LegacyTrainingTime:
    """Match the retained RL logger: the first callback establishes the warmup end."""

    warmup: float | None = None

    def at_hook(self, optimization_elapsed: float) -> float:
        if self.warmup is None:
            self.warmup = float(optimization_elapsed)
        return self.elapsed(optimization_elapsed)

    def elapsed(self, optimization_elapsed: float) -> float:
        return max(0.0, float(optimization_elapsed) - (self.warmup or 0.0))
