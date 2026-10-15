"""Tests for the interfaces joining the imported action and visual branches."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from egoengine.data import aligned_trajectory, cases
from egoengine.export import export_dataset
from egoengine.video import writer
from egoengine.visual import camera_quaternion, composite


class PipelineTest(unittest.TestCase):
    def test_bimanual_policy_observes_the_active_object(self):
        import mujoco

        from spider.rl.state_layout import bimanual_object_slice

        for case in cases():
            if case.info["embodiment_type"] != "bimanual":
                continue
            model = mujoco.MjModel.from_xml_path(str(case.scene))
            reference = case.reference()["qpos"]
            objects = reference[:, -14:]
            for side in ("right", "left"):
                index = int(model.jnt_qposadr[model.joint(f"{side}_object_joint").id])
                np.testing.assert_array_equal(
                    objects[:, bimanual_object_slice(model, side)], reference[:, index : index + 7]
                )

    def test_six_camera_tracks_remove_only_recorded_warmup(self):
        available = cases()
        self.assertEqual(len(available), 6)
        for case in available:
            poses, k = case.camera()
            self.assertEqual(len(poses), len(case.reference()["qpos"]))
            self.assertEqual(k.shape, (3, 3))
            np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=2e-5)

    def test_camera_axes_preserve_world_position_and_flip_y_z(self):
        from scipy.spatial.transform import Rotation

        q = camera_quaternion(np.eye(4))
        np.testing.assert_allclose(
            Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), np.diag([1, -1, -1]), atol=1e-10
        )

    def test_taco_target_relative_camera_is_composed_into_world(self):
        for case in cases():
            if case.info["dataset"] != "TACO":
                continue
            raw = json.loads((case.visual / "retarget.json").read_text())
            index = int(raw["warm_up_steps"])
            expected = np.asarray(raw["traj"][index]["target"]) @ np.linalg.inv(
                np.asarray(raw["camera_traj"]["target2camera"][index])
            )
            np.testing.assert_allclose(case.camera()[0][0], expected, atol=1e-10)

    def fixture_case(self, root):
        ref = dict(qpos=np.zeros((3, 2)), qvel=np.zeros((3, 2)), ctrl=np.ones((3, 1)))
        return SimpleNamespace(
            name="fixture",
            ref_dt=0.05,
            reference=lambda: ref,
            visual=root,
            info={"dataset": "Aria"},
            camera=lambda: (
                np.repeat(np.eye(4)[None], 3, axis=0),
                np.array([[10.0, 0, 4], [0, 10, 4], [0, 0, 1.0]]),
            ),
        )

    def make_video(self, path, n=3):
        out = writer(path, 8, 8, 20)
        for i in range(n):
            out.write(np.full((8, 8, 3), i * 60, dtype=np.uint8))
        out.release()

    def test_action_states_use_pre_action_frame_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self.fixture_case(Path(tmp))
            path = Path(tmp) / "action.npz"
            np.savez(
                path,
                success=True,
                statistics_json=json.dumps({"sim_steps_per_reference": 2}),
                qpos=np.repeat(np.arange(1, 7)[:, None], 2, axis=1),
                qvel=np.ones((6, 2)),
                initial_qpos=np.zeros(2),
                initial_qvel=np.zeros(2),
                ctrl=np.arange(6)[:, None],
            )
            aligned = aligned_trajectory(case, path)
            np.testing.assert_array_equal(aligned["qpos"][:, 0], [0, 2, 4])
            np.testing.assert_array_equal(aligned["ctrl"][:, 0], [0, 2, 4])
            np.testing.assert_allclose(aligned["timestamps"], [0, 0.05, 0.1])
            self.assertTrue(aligned["success"])

    def test_hdf5_frames_actions_and_intrinsics_stay_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = self.fixture_case(root)
            video = root / "cropped_video.mp4"
            self.make_video(video)
            path = export_dataset(case, video, root / "dataset.hdf5", image_size=4)
            with h5py.File(path) as f:
                demo = f["data/demo_0"]
                self.assertEqual(demo["obs/agentview_image"].shape, (3, 4, 4, 3))
                self.assertEqual(demo["actions"].shape, (3, 1))
                np.testing.assert_allclose(demo["timestamps"][:], [0, 0.05, 0.1])
                self.assertEqual(demo["camera_intrinsics"][0, 0], 5)
                self.assertEqual(demo.attrs["action_source"], "kinematic_reference")

    def test_mismatched_frames_fail_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = self.fixture_case(root)
            video = root / "short.mp4"
            self.make_video(video, n=2)
            with self.assertRaises(ValueError):
                export_dataset(case, video, root / "dataset.hdf5")
            self.assertFalse((root / "dataset.hdf5").exists())

    def test_missing_taco_background_is_explicit(self):
        case = next(c for c in cases() if c.info["dataset"] == "TACO")
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(FileNotFoundError, "Prepared visual stage missing"),
        ):
            composite(case, tmp)

    def test_visual_delegates_to_original_replay_without_building_a_robot(self):
        from unittest.mock import patch

        from egoengine.visual import render

        case = object()
        with (patch("egoengine.legacy_visual.render_original", return_value="original-output") as original,
              patch("egoengine.visual.calibrated_model", side_effect=AssertionError("synthetic model"))):
            result = render(case, "unused", action_path="refined-eef.npz")
        self.assertEqual(result, "original-output")
        original.assert_called_once_with(case, "unused", None, None)

    def test_original_visual_can_pair_with_refined_eef_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = self.fixture_case(root)
            video = root / "cropped_video.mp4"
            self.make_video(video)
            video.with_suffix(".json").write_text(json.dumps({
                "case": case.name, "source": "original_retarget_render",
                "retarget_sha256": "original-configurations",
            }))
            action = root / "action.npz"
            np.savez(action, success=True,
                     statistics_json=json.dumps({"sim_steps_per_reference": 2}),
                     qpos=np.full((6, 2), 0.25), qvel=np.zeros((6, 2)),
                     initial_qpos=np.zeros(2), initial_qvel=np.zeros(2),
                     ctrl=np.full((6, 1), 0.5))
            dataset = export_dataset(case, video, root / "dataset.hdf5", action)
            with h5py.File(dataset) as f:
                demo = f["data/demo_0"]
                self.assertEqual(demo.attrs["visual_source"], "original_retarget_render")
                self.assertEqual(demo.attrs["action_source"], "adaptive_physics")
                np.testing.assert_allclose(demo["actions"][:], 0.5)

    def test_visual_arm_commands_do_not_replace_eef_action_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = self.fixture_case(root)
            video = root / "cropped_video.mp4"
            self.make_video(video)
            (root / "render").mkdir()
            names = [f"joint_{i}" for i in range(19)]
            commands = np.arange(57).reshape(3, 19)
            np.savez(root / "render/robot_commands.npz", joint_names=names, joint_commands=commands)
            video.with_suffix(".json").write_text(json.dumps({
                "case": case.name, "source": "original_retarget_render", "renderer": "configured_robot",
                "render_directory": "render", "commands_file": "robot_commands.npz",
                "robot_name": "FrankaXHand",
            }))
            dataset = export_dataset(case, video, root / "dataset.hdf5")
            with h5py.File(dataset) as f:
                demo = f["data/demo_0"]
                np.testing.assert_array_equal(demo["actions"][:], case.reference()["ctrl"])
                np.testing.assert_array_equal(demo["obs/robot_joint_commands"][:], commands)
                self.assertEqual(demo.attrs["visual_robot"], "FrankaXHand")
                self.assertEqual(json.loads(demo.attrs["robot_joint_names"]), names)
                self.assertNotIn("obs/rby1_joint_commands", demo)
            np.savez(root / "render/robot_commands.npz", joint_names=names,
                     joint_commands=commands[:2])
            failed = root / "misaligned.hdf5"
            with self.assertRaisesRegex(ValueError, "command length mismatch"):
                export_dataset(case, video, failed)
            self.assertFalse(failed.exists())
            for invalid in (np.full((3, 19), np.nan), np.ones((3, 18))):
                np.savez(root / "render/robot_commands.npz", joint_names=names,
                         joint_commands=invalid)
                with self.assertRaisesRegex(ValueError, "joint layout or values"):
                    export_dataset(case, video, failed)
                self.assertFalse(failed.exists())



if __name__ == "__main__":
    unittest.main()
