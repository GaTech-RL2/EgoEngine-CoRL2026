# Adaptive Replay → MPC → RL

`action/examples/run_mjwp_mixed.py` implements the greedy two-chunk procedure described
in [EgoEngine Appendix C.1](https://arxiv.org/html/2606.12604v1). The input is a
prepared scene and kinematic trajectory. Perception and FoundationPose are
outside this entry point.

## Selection and state

A chunk has **20 control steps** by default. At `sim_dt=0.01` and
`ctrl_dt=0.05`, this is 100 physics steps. The runner checks the current and next
chunk from the same saved simulator state, trying enabled modes in this order:

1. Replay the reference controls.
2. Refine controls with the existing sampling MPC solver.
3. Train the existing residual PPO policy for this planning window.

A candidate is feasible only if the entire available planning window completes
without the existing object tracking termination condition or nonfinite outputs.
Passing the current chunk but failing its lookahead rejects the candidate.
Only the current chunk's controls and its captured boundary state are committed.
The next boundary starts selection from Replay again. The final short chunk and
its available lookahead are clipped to the reference length. If every enabled
mode fails, the saved boundary is restored and the export is marked unsuccessful;
failed partial controls are not appended to the generated trajectory.

Snapshots preserve positions, velocities, controls, time, actuator state,
solver warmstart, applied forces, mocap and equality activation where supported.
MPC probes use a separate vectorized environment. RL policy evaluations use the
single-world environment and restore its boundary after each probe. The shared
work ledger is never rolled back.

`--methods` enables a subset; its spelling/order does not change the priority.
`--window-size` counts control steps; `--lookahead-chunks 2` reproduces the paper.
The old `--method-seed` option remains accepted for CLI compatibility and has no
effect on greedy selection. The RL iteration budget is shared by all windows,
with unspent iterations from an early solution available to later windows.
Each RL window receives up to `--rl-window-learning-iterations 300`, bounded by
the remaining global budget. This restores the original chunk launcher's
per-window limit.
The last accepted native RSL checkpoint carries actor, critic, normalization and
optimizer state into later RL windows, including across Replay/MPC segments.
Its iteration counter restarts at zero, matching the retained `--load-ckpt`
flow. No policy is carried between demonstrations.
The RL observation, independent resets, residual scales and exploration follow
the CoDesign port documented in [RL_RECIPE.md](RL_RECIPE.md). Mode selection,
normal tracking limits, prepared references and the timing convention below
are retained.

## Metrics: preserve the existing implementation

Use the retained original RL logger in
[`action/examples/run_mjwp_rl_rsl.py`](../action/examples/run_mjwp_rl_rsl.py),
the original chunk launcher and result-analysis scripts as the metric source.
Do not redefine their timestep unit, denominator, exclusions or progress rule.
The original RL history already records
`env_steps = (it + 1) * num_envs * num_steps_per_env` and associates it with
policy evaluation progress. This differs from the added physics-work ledger;
the two fields must not be substituted for each other.

The result inventory copies existing `cost`, `env_steps` and diagnostic counters
verbatim, leaving absent values absent. It does not recompute Cost or publish
an efficiency ranking while the original metric integration is being verified.

The original first-farthest timing hooks remain intact as supplemental records.
[Metric source provenance](upstream/METRIC_SOURCES.json) distinguishes verified
original code from added diagnostic instrumentation.

## Run prepared examples

From this unified checkout, after installing the CUDA dependencies:

```bash
egoengine action --case mustard_1
egoengine action --case brush_brush_bowl_20230927_027
egoengine action --case mustard_1 --methods replay
```

The unified entry point supplies portable case parameters. Raw runner options
remain available via `--action-args`. Results are written to
`outputs/<case>/action/trajectory_mjwp_mixed.npz` and a matching `.json` file.
NPZ retains the prior arrays and adds actual per-physics-step `qpos`/`qvel`,
initial state, success, plan boundaries and attempt/statistics JSON.

`egoengine render --case mustard_1` replays the original RB-Y1 `robot_cfg`,
object poses and camera track. It is independent of optimized EEF states.
`egoengine export --trajectory ...` pairs refined EEF actions with those visual
observations on the reference clock. `--renderer xhand --trajectory ...` remains
available only for direct inspection of the floating-wrist physics scene.

## Validation

```bash
python -m unittest discover -s action/tests -v
```

The action protocol CPU suite executes the actual mixed runner functions with injected backends
and tests lookahead rejection, escalation, return to Replay, prefix-only commit,
short tails, rollback on total failure and exceptions, nonfinite rejection,
metadata interpolation, strict JSON/failure export, work accounting and the real
RL logger hook's timing/early-stop behavior. CUDA/Warp integration must be tested
in a GPU runtime before reporting performance. The public `scripts/gpu_smoke.py`
was also run successfully on one A40 with Replay, MPC and RL separately. It uses
five reference frames and loose thresholds, and checks boundaries/state exports
and accounting. `scripts/visual_smoke.py` joins an actual RL output to rendering,
compositing and HDF5. See [the reproduction guide](REPRODUCIBILITY.md) for the
environment-specific checks and their limits.
