import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from egoengine.data import PROJECT_ROOT
from egoengine.robot_visual import base_pose, lighting_state, load_config, register_robot


class RobotConfigTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("gymnasium") and importlib.util.find_spec("numba"),
        "Install requirements/visual-tested.txt for original visual integration",
    )
    def test_arm_replacement_preserves_world_lights_with_a_rotated_base(self):
        import sys

        import mujoco

        sys.path.insert(0, str(PROJECT_ROOT / "visual"))
        from robosuite.models.robots.manipulators.rby1_robot import RBY1ArmsOnlyAria

        source = RBY1ArmsOnlyAria(idn=0)
        lights = source.worldbody.findall("light")
        self.assertTrue(lights)
        source_model = mujoco.MjModel.from_xml_string(source.get_xml())
        config = load_config(PROJECT_ROOT / "examples/robots/franka_xhand.json")
        config["base_quat_wxyz"] = [0.5, 0.5, 0.5, 0.5]
        placement = base_pose(config, {"init_state": {"desk": np.eye(4).tolist()}})
        robot_class = register_robot(config, placement, lights)
        for _ in range(2):
            robot = robot_class(idn=0)
            robot.set_base_xpos(placement[:3, 3])
            replacement = mujoco.MjModel.from_xml_string(robot.get_xml())
            self.assertEqual(lighting_state(source_model), lighting_state(replacement))
            self.assertEqual(len(robot.worldbody.findall("light")), len(lights))
            self.assertIsNot(robot.worldbody.find("light"), lights[0])

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
