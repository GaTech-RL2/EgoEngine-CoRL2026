# Use my robot: Franka + XHand

This example replaces the RB-Y1 arm in the visual branch with a Franka Panda arm.
It keeps the recorded right XHand joint commands and the existing XHand XML,
meshes and materials. The action branch continues to generate floating-wrist
EEF + XHand actions.

The example uses the three prepared Aria Mustard demonstrations. Camera tracks,
object motion, removed backgrounds and the original compositor are reused. The
default RB-Y1 commands remain available without a robot configuration.

## Run the example

Use the Python 3.10 visual environment from the main README:

```bash
python -m pip install -r requirements/visual-tested.txt
python -m pip install --no-deps -e .

# Complete visual demo and paired reference-EEF dataset:
python -m egoengine demo --case mustard_1 \
  --robot-config examples/robots/franka_xhand.json --output outputs/franka_xhand

# Three cases, with 480-pixel render width:
bash scripts/reproduce_franka_visual.sh
```

Individual stages use the same configuration:

```bash
python -m egoengine render --case mustard_1 --width 480 \
  --robot-config examples/robots/franka_xhand.json --output outputs/franka_xhand
python -m egoengine composite --case mustard_1 --output outputs/franka_xhand \
  --render-dir outputs/franka_xhand/mustard_1/render
python -m egoengine export --case mustard_1 --output outputs/franka_xhand \
  --video outputs/franka_xhand/mustard_1/visual.mp4
```

To pair the visuals with a successful refined action trajectory, pass that NPZ
to `export --trajectory`. To train and render together, `egoengine run` accepts
the same `--robot-config` option. Visual wrist targets still come from the
original retarget; RL states do not drive arm fitting.

## What changes

1. Load the original RB-Y1 retarget in its original environment and read the
   right XHand root pose through forward kinematics.
2. Load the existing robosuite Panda model, attach the unchanged XHand through
   the normal gripper factory and place its fixed base using the configuration.
3. Fit only Panda's seven arm joints to the recorded wrist poses. Copy all twelve
   recorded XHand joint values directly, outside the arm optimizer.
4. Replay the resulting joint configurations with the original camera and object
   updates. Select the new robot's visual geometry by its model hierarchy.
5. Generate foreground/visibility layers using the original render-difference
   procedure, then run the existing compositor with its original parameters.

Panda's visual mesh splits and material definitions come from its existing XML.
Missing referenced meshes are restored from the robosuite 1.5.1 distribution;
their source and hashes are recorded in `docs/upstream/FRANKA_VISUAL_ASSETS.json`.
Group-0 collision geometry is excluded from rendering.

Scene lighting is inherited from the original replay independently of the arm.
For Aria, this includes the warm overhead light defined in the RB-Y1 worldbody.
The replacement keeps that light in world coordinates, along with the original
headlight settings. Before rendering, the adapter verifies that the compiled
scene lights match the original environment. Mesh colors and material properties
remain those of the authored Panda and XHand assets.

## Rendered examples

The panels show source RGB, the newly rendered Franka + original XHand foreground,
and the output from the retained compositor at matching frame indices.

![Franka Mustard 1](https://raw.githubusercontent.com/GaTech-RL2/EgoEngine-CoRL2026/main/docs/media/franka_xhand_mustard_1.gif?v=48536b29efc7)

| Mustard 2 | Mustard 3 |
| --- | --- |
| ![Franka Mustard 2](https://raw.githubusercontent.com/GaTech-RL2/EgoEngine-CoRL2026/main/docs/media/franka_xhand_mustard_2.gif?v=331f5d25104a) | ![Franka Mustard 3](https://raw.githubusercontent.com/GaTech-RL2/EgoEngine-CoRL2026/main/docs/media/franka_xhand_mustard_3.gif?v=85a061ad878b) |

## Configuration

Start with [`examples/robots/franka_xhand.json`](../examples/robots/franka_xhand.json).

| Field | Meaning |
| --- | --- |
| `robot_module`, `robot_class` | Importable robosuite arm model, here the existing `Panda` class |
| `arm_joints` | Ordered arm hinge joints without the `robot0_` prefix |
| `base_position_in_desk` | Fixed robot-base position in the prepared desk frame, in meters |
| `base_quat_wxyz` | Fixed base orientation relative to the desk |
| `gripper_name` | Existing `XRightHandAria`; no hand-model editing is needed |
| `wrist_body` | Compiled body used as the new robot's wrist target |
| `source_wrist_body` | Same physical hand-root frame in the original replay |

The desk frame is the prepared `init_state.desk`; the adapter applies the same
−0.7 m world-X shift as the original visual replay to both the desk and camera.
The base pose is fixed for the entire clip. The supplied configuration is shared
across all three example cases; there are no per-task trajectory offsets.

For another fixed single right arm, provide its robosuite model and asset paths,
then change the arm module/class, ordered joint names and calibrated base pose.
The model must expose a right-hand attachment frame and finite hinge-joint limits.
Use its model's normal gripper mounting transform to express the physical adapter
between the flange and XHand. Keep the XHand frame convention consistent on both
sides of the wrist mapping. This first example covers single-arm Aria inputs;
bimanual TACO support needs an explicit two-arm mapping.

## Outputs and checks

```text
outputs/franka_xhand/mustard_1/
  render/scene.mp4
  render/foreground.mp4
  render/visibility_mask.mp4
  render/robot_commands.npz
  render/render.json
  render/robot_replay.log
  visual.mp4
  visual.json
  dataset.hdf5
```

`robot_commands.npz` contains seven Franka arm commands, twelve unchanged XHand
commands, target/actual wrist poses and per-frame wrist errors. Inspect those
errors and the rendered mask when calibrating a new robot. `render.json` records
the fixed base pose, source/configuration hashes, compiled scene lighting and
visibility statistics.

The HDF5 `actions` retain EEF + XHand semantics. The optional
`obs/robot_joint_commands` stores the nineteen visual replay joint values, with
their names and robot identity as attributes. These visual replay values are
separate from the action-branch controls. This example demonstrates visual
generation and dataset export; it does not establish physical Franka execution.

The original RL recipe, mode switching and metric code are unchanged by this
adapter.

After generating all three cases, verify the recorded hand commands, camera and
object tracks, rendered wrist poses, scene lighting, authored materials and
exported action labels:

```bash
python scripts/validate_robot_visual.py
python scripts/build_robot_visual_media.py
```

The measured report is [`validation/FRANKA_XHAND.json`](validation/FRANKA_XHAND.json).

| Case | Complete visual/dataset frames | Maximum wrist position error | Maximum wrist angle error |
| --- | ---: | ---: | ---: |
| Mustard 1 | 93 | 11.93 mm | 0.03667 rad |
| Mustard 2 | 105 | <0.001 mm | <0.000002 rad |
| Mustard 3 | 108 | 8.02 mm | 0.02538 rad |

The same fixed base/configuration was used for all three. Every frame has a
visible robot; all recorded XHand values and EEF labels match the original inputs.
The wrist fit respects the Panda joint limits, so it is approximate for some
poses. All 52 authored Panda/XHand materials were checked against the compiled
model. The rendered-scene wrist poses also match those from the arm mapping.
