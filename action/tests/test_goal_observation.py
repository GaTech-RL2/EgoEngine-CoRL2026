import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from spider.rl.goal_observation import GoalObservation, restore_masked_state

ROOT = Path(__file__).resolve().parents[2]


class GoalObservationTest(unittest.TestCase):
    def test_residual_ranges_do_not_change_with_mpc_noise(self):
        source = ast.parse((ROOT / "spider/rl/mjwp_rsl_env.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_build_action_scale")
        ns = dict(torch=torch, Config=object)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "scale", "exec"), ns)
        env_cfg = SimpleNamespace(root_position_scale=0.10, root_rotation_scale=0.30,
                                  finger_scale=0.30)
        cfg = SimpleNamespace(nu=18, device="cpu", embodiment_type="right",
                              pos_noise_scale=0.0325, rot_noise_scale=0.05,
                              joint_noise_scale=0.3, first_ctrl_noise_scale=0.5)
        first = ns["_build_action_scale"](cfg, env_cfg)
        cfg.pos_noise_scale = cfg.rot_noise_scale = cfg.first_ctrl_noise_scale = 100
        second = ns["_build_action_scale"](cfg, env_cfg)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[:3], torch.full((3,), 0.1))
        torch.testing.assert_close(first[3:], torch.full((15,), 0.3))

    def test_window_step_keeps_surviving_worlds_running(self):
        tree = ast.parse((ROOT / "action/examples/run_mjwp_mixed.py").read_text())
        runner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_run_rl_window")
        window = next(node for node in runner.body if isinstance(node, ast.ClassDef))

        class Base:
            def _get_obs(self):
                return self._sim_step_buf[:, None].float()

            def _sync_scalar_clocks_from_env0(self):
                pass

            def _obs_tensordict(self):
                return self.obs_buf

        physics_steps = [0]

        def step(*args):
            physics_steps[0] += 1

        ns = dict(torch=torch, Config=object, MJWPRslResidualEnv=Base, MJWPRLEnvCfg=object,
                  EnvSnapshot=object, wp=SimpleNamespace(to_torch=lambda x: x),
                  restore_masked_state=restore_masked_state, step_env=step,
                  reset_inactive_side_to_ref=lambda *args: None,
                  get_reward=lambda *args: (torch.ones(2), {}),
                  get_terminate=lambda *args: torch.tensor([physics_steps[0] == 1, False]))
        exec(compile(ast.Module(body=[window], type_ignores=[]), "window", "exec"), ns)
        env = ns["WindowResidualEnv"].__new__(ns["WindowResidualEnv"])
        env.cfg = SimpleNamespace(ctrl_steps=2)
        env.env_cfg = SimpleNamespace(action_clip=1.0)
        env.device, env.num_envs, env.num_actions = "cpu", 2, 1
        env.max_ref_idx, env.max_episode_length = 10, 8
        env._window_start_step, env._window_end_step = 0, 8
        for name in ["_sim_step_buf", "_start_sim_step_buf", "_episode_ctrl_steps_buf",
                     "episode_length_buf"]:
            setattr(env, name, torch.zeros(2, dtype=torch.long))
        for name in ["episode_return_buf", "episode_base_return_buf", "episode_extra_return_buf"]:
            setattr(env, name, torch.zeros(2))
        env._prev_actions, env._has_prev_action = torch.zeros(2, 1), torch.zeros(2, dtype=torch.bool)
        env.qpos_ref = env.qvel_ref = torch.zeros(11, 1)
        env._ref_at_indices = lambda phase: ((torch.zeros(2, 1),), phase)
        env._active_object_z_tracking_reward = lambda *args: torch.zeros(2)
        env.residual_actions_to_ctrl = lambda *args: (torch.zeros(2, 1), None, None)
        env._window_snapshot = SimpleNamespace(qpos=torch.zeros(1), qvel=torch.zeros(1),
                                               ctrl=torch.zeros(1), time=0.0, state_fields={})
        env.env = SimpleNamespace(data_wp=SimpleNamespace(qpos=torch.ones(2, 1),
                                  qvel=torch.ones(2, 1), ctrl=torch.ones(2, 1), time=torch.ones(2)))
        obs, reward, done, extras = env.step(torch.zeros(2, 1))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(env._sim_step_buf.tolist(), [0, 2])
        self.assertEqual(env.episode_length_buf.tolist(), [0, 1])
        self.assertEqual(obs.flatten().tolist(), [0.0, 2.0])
        self.assertFalse(extras["time_outs"].any())
        self.assertEqual(env.env.data_wp.qpos[:, 0].tolist(), [0.0, 1.0])
        # Ending the whole batch earlier must not rescale a world's reward.
        ns["get_terminate"] = lambda *args: torch.ones(2, dtype=torch.bool)
        _, all_finished_reward, _, _ = env.step(torch.zeros(2, 1))
        torch.testing.assert_close(all_finished_reward[0], reward[0])

    def test_reference_joint_and_tip_goals_match_scene_fk_for_both_datasets(self):
        paths = list((ROOT / "examples/prepared/processed").rglob("scene.xml"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(case=path.parent.name):
                model = mujoco.MjModel.from_xml_path(str(path))
                reference = np.load(path.parent / "0/trajectory_kinematic.npz")["qpos"][:2]
                reference = torch.tensor(reference, dtype=torch.float32)
                bimanual = "gigahand" in str(path)
                builder = GoalObservation(model, reference, "bimanual" if bimanual else "right", "right")
                data = mujoco.MjData(model)
                data.qpos[:] = reference[0].numpy()
                mujoco.mj_kinematics(model, data)
                sites = torch.tensor(data.site_xpos, dtype=torch.float32).unsqueeze(0)
                actual = builder.build(reference[:1], sites, 0)
                self.assertEqual(actual.shape, (1, 70))
                torch.testing.assert_close(actual[:, :18], actual[:, 45:63])
                torch.testing.assert_close(actual[:, 24:31], actual[:, 63:70])
                torch.testing.assert_close(actual[:, 18:21], actual[:, 31:34])
                torch.testing.assert_close(actual[:, 21:24], actual[:, 38:41])
                # Changing the goal must not change the observed current state.
                other = builder.build(reference[:1], sites, 1)
                torch.testing.assert_close(actual[:, :31], other[:, :31])
                self.assertFalse(torch.equal(actual[:, 31:], other[:, 31:]))

    def test_reset_only_finished_world_and_restore_actual_boundary_runtime(self):
        fields = {"qpos": torch.arange(8).reshape(2, 4).float(),
                  "qvel": torch.ones(2, 4), "ctrl": torch.ones(2, 3),
                  "time": torch.tensor([1.0, 2.0]),
                  "qacc_warmstart": torch.ones(2, 4),
                  "eq_active": torch.ones(2, 1, dtype=torch.bool)}
        before = {name: value.clone() for name, value in fields.items()}
        snapshot = SimpleNamespace(qpos=torch.full((4,), 17.0), qvel=torch.full((4,), 18.0),
                                   ctrl=torch.full((3,), 19.0), time=20.0,
                                   state_fields={"qacc_warmstart": torch.full((4,), 21.0),
                                                 "eq_active": torch.tensor([False])})
        restore_masked_state(SimpleNamespace(**fields), snapshot, torch.tensor([True, False]), lambda x: x)
        for name in fields:
            torch.testing.assert_close(fields[name][1], before[name][1])
        self.assertEqual(fields["qpos"][0, 0].item(), 17)
        self.assertEqual(fields["time"][0].item(), 20)
        self.assertFalse(fields["eq_active"][0, 0])


if __name__ == "__main__":
    unittest.main()
