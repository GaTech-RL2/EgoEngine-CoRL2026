# CoDesign residual PPO

The adaptive runner uses CoDesign's goal-conditioned residual-learning
formulation in the MuJoCo Warp / RSL-RL backend. Source versions and the ported
components are recorded in [CODESIGN_RECIPE.json](upstream/CODESIGN_RECIPE.json).

## Observation

Each active XHand receives 70 values, in world coordinates:

| Slice | Values |
| --- | --- |
| 0:18 | Current floating wrist and finger joint positions |
| 18:24 | Current thumb and index tip XYZ |
| 24:31 | Current object XYZ and wxyz quaternion |
| 31:45 | Reference thumb and index tip XYZ and wxyz quaternion |
| 45:63 | Reference wrist and finger joint positions |
| 63:70 | Reference object pose |

Reference tip poses come from scene FK on the prepared reference `qpos`, rather
than actuator commands or task-specific offsets. Current FK is refreshed after
integration and reset. Training and deterministic evaluation share the same
observation builder and control conversion. TACO selects its active object by
the named scene joint. Both-hand mode includes both hands and objects.

The optional WUJI collision-landmark observation is specific to a changing hand
morphology. It is not used for this fixed XHand release.

## Actions and training

Normalized residuals are clipped to [-1, 1] and added to the reference controls.
The default correction ranges are 0.10 m for wrist translation, 0.30 rad for
wrist rotation, and 0.30 rad for fingers. These ranges are independent of MPC
sampling noise. Actuator limits are enforced. Override them with
`--rl-root-position-scale`, `--rl-root-rotation-scale`, and `--rl-finger-scale`.

The policy clips its mean and sampled actions. Its learned log standard
deviation starts at -2 and is bounded to [-5, -1]. This follows the CoDesign
clipped-Gaussian convention. Entropy weight is 0.001. Actor and critic use
separate [256, 128] ELU MLPs and running observation normalization. PPO retains
16 rollout steps, five epochs, four minibatches, learning rate 3e-4, adaptive
KL target 0.01, gamma 0.99, lambda 0.95, clipping 0.2 and gradient clipping 1.0.
The imported settings remain recorded in `upstream/RL_RECIPE.json`.

Each world keeps its own phase and resets independently. A finished world
returns to the actual committed boundary state, including velocities and
integration runtime fields. Surviving worlds continue. A rollout boundary does
not reset an episode. Window completion and timeout end the episode without
timeout value bootstrapping. Post-step validation uses the next physics-frame
reference consistently in RL training and evaluation.
Rewards after termination are zero for the remaining substeps of that control
tick, with a fixed control-step denominator independent of other worlds.

This port retains EgoEngine's task-independent object tracking and lifting
rewards, physics, and normal 0.05 m / 1 rad failure limits. CoDesign's Isaac Lab
contact implementation, value-target scaler, and morphology-dependent features
are not reproduced by the current RSL-RL backend. This is a migration of the
observation, residual action and episode/exploration scheme, not a bitwise
reproduction of the entire CoDesign training stack.

## Artifacts and timing

Each attempted RL window exports `_mixed_rl/window_NNNN/learning_trace.json`
and `policy_best.pt`. The checkpoint records the best evaluated actor state,
configuration, observation version, dimensions, window boundaries, and whether
its full plan passed validation. A partial checkpoint is labeled infeasible.
The paired dataset uses only accepted controls and actual simulated states.
Successful windows also save native RSL `policy_continue.pt` checkpoints.
Subsequent RL windows load the actor, critic, normalizers and optimizer from the
last accepted window, with a new iteration counter, matching the original
chunk script. The default window limit is 300 iterations within the shared 1500
total. Checkpoint loading is setup; saving remains an excluded export hook.

Mode selection and the original first-farthest-progress timing remain defined
in [MODE_SWITCHING.md](MODE_SWITCHING.md). Evaluations and artifact writing are
excluded from optimization time and listed separately. All training physics
steps, including failed worlds and probes, remain in the work ledger.

Run the normal-threshold check with:

```bash
python scripts/check_mode_switching.py --mpc-num-samples 8 --mpc-max-iters 2 \
  --rl-num-envs 1024 --rl-total-learning-iterations 300 \
  --output outputs/codesign_300
```

To plot actual deterministic validation progress (requires Matplotlib):

```bash
python scripts/plot_rl_learning.py outputs/codesign_1500/mixed.npz \
  --output outputs/codesign_1500/rl_learning.png
```
