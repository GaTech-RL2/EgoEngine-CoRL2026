"""Exercise Aria export without a simulator; keep measured references intact."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


class _Solver:
    def __init__(self, path):
        pass

    def compute_ik(self, q_dict, **kwargs):
        q = q_dict.copy()
        for name in q:
            if name.startswith("right_hand_"):
                q[name] = 0.8
        return {"q": q}

    def compute_fk(self, q, names):
        return {name: np.eye(4, dtype=np.float32) for name in names}


class ReferencePreservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        visualizer = types.ModuleType("retargeting.pb_visualizer")
        visualizer.Entity = lambda **kwargs: types.SimpleNamespace(**kwargs)
        visualizer.PybulletVisualizer = None
        kinematics = types.ModuleType("retargeting.rby1_kinematics")
        kinematics.MinkSolver = _Solver
        kinematics.Target = lambda **kwargs: types.SimpleNamespace(**kwargs)
        with patch.dict(sys.modules, {
            visualizer.__name__: visualizer,
            kinematics.__name__: kinematics,
        }):
            source = Path(__file__).resolve().parents[1] / "retargeting/show_aria_new.py"
            spec = importlib.util.spec_from_file_location("retargeting._reference_preservation_test", source)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_task_labels_cannot_change_measured_motion(self):
        m = self.module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_dir = root / "Hand_Poses" / "episode" / "1"
            object_dir = root / "Object_Poses" / "episode" / "1"
            camera_dir = root / "Egocentric_Camera_Parameters" / "episode" / "1"
            for directory in (hand_dir, object_dir, camera_dir):
                directory.mkdir(parents=True)
            points = np.random.default_rng(17).normal(0.0, 0.05, (21, 3)).astype(np.float32)
            hand_file = hand_dir / "all_pose.json"
            hand_file.write_text(json.dumps({"frames": [{"right_camera": points.tolist()}] * 3}))
            w2cam = camera_dir / "extrinsic.npy"
            np.save(w2cam, np.repeat(np.eye(4)[None], 3, axis=0))
            np.savetxt(camera_dir / "egocentric_intrinsic.txt", np.eye(3))
            object_pose = np.eye(4, dtype=np.float32)
            object_pose[:3, 3] = [0.42, -0.18, 0.09]
            np.save(object_dir / "object.npy", np.repeat(object_pose[None], 3, axis=0))
            tag = root / "tag.txt"
            np.savetxt(tag, np.eye(4))
            target = root / "target.txt"
            target_pose = np.eye(4)
            target_pose[:3, 3] = [0.65, -0.25, 0.8]
            np.savetxt(target, target_pose)

            world_from_camera = np.eye(4, dtype=np.float32)
            world_from_camera[:3, :3] = m.R_CAM_TO_ARIA
            expected_object = world_from_camera @ object_pose
            expected_tips = (points[m.ARIA["tips"]] @ m.CAM_TO_HAND_R.T) @ m.R_CAM_TO_ARIA.T
            trajectories = []
            for task in ("mustard", "drawer", "flower", "hammer"):
                out = root / f"{task}.json"
                m.retarget_to_robot(
                    str(w2cam), str(hand_file), str(tag), str(out), False, 0.0,
                    task, 0, "robot.urdf", "robot.xml", "desk.urdf", (0.9, 0.0, 0.699),
                    target_pose_path=str(target),
                )
                data = json.loads(out.read_text())
                trajectories.append(data["traj"])
                self.assertEqual(len(data["traj"]), 3)
                for frame in data["traj"]:
                    np.testing.assert_allclose(frame["tool"], expected_object, atol=1e-6)
                    tips = [frame["fingertips_world"][name] for name in m.TIP_NAMES]
                    np.testing.assert_allclose(tips, expected_tips, atol=1e-6)
                    for name, value in frame["robot_cfg"].items():
                        if name.startswith("right_hand_"):
                            self.assertAlmostEqual(value, 0.8)
                if task == "hammer":
                    np.testing.assert_allclose(data["init_state"]["target"], target_pose)
            for trajectory in trajectories[1:]:
                for baseline, frame in zip(trajectories[0], trajectory):
                    self.assertEqual(baseline["robot_cfg"], frame["robot_cfg"])
                    np.testing.assert_allclose(baseline["tool"], frame["tool"])


if __name__ == "__main__":
    unittest.main()
