# Retained components

| Component | Source | Retained terms |
| --- | --- | --- |
| Action optimization (`spider/`, `action/`) | [SPIDER-derived EgoEngine-MPC](https://github.com/Randle-Github/EgoEngine-MPC) | [CC BY-NC notice](licenses/SPIDER.txt) |
| Visual/retargeting source (`visual/`) | [GaTech-RL2 robosuite fork](https://github.com/GaTech-RL2/robosuite) | [MIT notice and MuJoCo attribution](licenses/ROBOSUITE.txt) |
| Referenced Panda visual/collision mesh files | robosuite 1.5.1 distribution; `docs/upstream/FRANKA_VISUAL_ASSETS.json` | [Distribution notice](licenses/ROBOSUITE_PYPI_1.5.1.txt); existing Panda XML/materials preserved |
| Physics runtime | [MuJoCo](https://github.com/google-deepmind/mujoco), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Installed separately; upstream notices apply |
| Residual policy training | [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | Installed separately; upstream notices apply |
| Prepared TACO / Aria and XHand/object assets | `examples/prepared/manifest.json` | Historical research examples; source data/asset terms are retained |

Commit hashes are recorded in `UPSTREAMS.json`. The example manifest retains asset
hashes and case provenance. Combining the source does not relicense upstream code
or dataset assets. The original author credits in imported files are retained.
