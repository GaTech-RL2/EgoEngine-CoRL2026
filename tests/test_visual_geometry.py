"""Verify the retained visual sources and authored material values."""

import hashlib
import importlib.util
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from egoengine.data import PROJECT_ROOT, get_case


class OriginalVisualTest(unittest.TestCase):
    def test_retained_visual_sources_match_manifest(self):
        manifest = json.loads((PROJECT_ROOT / 'docs/upstream/ORIGINAL_VISUAL_PIPELINE.json').read_text())
        for name, expected in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest(), expected, name)

    @unittest.skipUnless(importlib.util.find_spec('gymnasium') and importlib.util.find_spec('numba'),
                         'Install requirements/visual-tested.txt for original visual integration')
    def test_authored_materials_survive_the_original_environment(self):
        sys.path.insert(0, str(PROJECT_ROOT / 'visual'))
        from rl_tracking_dev.replay_trajectory_seg_aria import make_tracking_env

        case = get_case('mustard_1')
        args = SimpleNamespace(data_file=str(case.visual / 'retarget.json'), cam_name='cameraview')
        env = make_tracking_env(args, use_cam_obs=False)()
        try:
            model = env.env.sim.model._model
            sources = [
                ('visual/robosuite/models/assets/robots/rby1/robot_patched_torso_x_aria.xml', 'robot0_'),
                ('visual/robosuite/models/assets/grippers/x_right_hand_aria.xml', 'gripper0_right_'),
            ]
            checked = set()
            for path, prefix in sources:
                for material in ET.parse(PROJECT_ROOT / path).findall('./asset/material'):
                    native = model.mat(prefix + material.get('name'))
                    for key in ('rgba', 'specular', 'shininess'):
                        if material.get(key) is not None:
                            expected = np.fromstring(material.get(key), sep=' ')
                            np.testing.assert_allclose(getattr(native, key), expected, atol=2e-7)
                    checked.add(material.get('name'))
            self.assertIn('RIGHT_ARM_WHITE_ARIA', checked)
            self.assertIn('RIGHT_ARM_BLACK_ARIA', checked)
            self.assertIn('hand_white', checked)
            self.assertIn('hand_black', checked)
            self.assertFalse(env.env.render_collision_mesh)
            self.assertTrue(env.env.render_visual_mesh)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
