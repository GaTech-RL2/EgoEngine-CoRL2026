import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from egoengine.data import PROJECT_ROOT
from egoengine.robot_visual import base_pose, load_config


class RobotConfigTest(unittest.TestCase):
    def test_base_uses_the_original_shifted_desk_frame(self):
        config = load_config(PROJECT_ROOT / "examples/robots/franka_xhand.json")
        desk = np.eye(4)
        desk[:3, 3] = [0.9, 0, 0.699]
        world = base_pose(config, {"init_state": {"desk": desk.tolist()}})
        np.testing.assert_allclose(world[:3, 3], [-0.7, -0.7, 0.749])
        np.testing.assert_allclose(world[:3, :3], np.eye(3))
        np.testing.assert_allclose(desk[:3, 3], [0.9, 0, 0.699])

    def test_invalid_mount_orientation_and_duplicate_arm_joints_are_rejected(self):
        config = load_config(PROJECT_ROOT / "examples/robots/franka_xhand.json")
        for field, value in (
            ("base_quat_wxyz", [0, 0, 0, 0]),
            ("arm_joints", ["joint1", "joint1"]),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "robot.json"
                path.write_text(json.dumps({**config, field: value}))
                with self.assertRaises(ValueError):
                    load_config(path)
