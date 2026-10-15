import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
from robosuite.models.robots.manipulators import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import find_parent

import xml.etree.ElementTree as ET
import tempfile
import os
import math

class RBY1(ManipulatorModel):
    """
    RB-Y1 is a wheeled robot with a manipulator arm created by Rainbow Robotics.
    """

    arms = ["right", "left"]

    def __init__(self, xpath="robots/rby1/robot.xml", idn=0):

        super().__init__(xml_path_completion(xpath), idn=idn)

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {"right": "XRightHand", "left": "XLeftHand"}

    @property
    def default_controller_config(self):
        return {
            "right":     "rby1_two_arm",
            "left":      "rby1_two_arm",
            "head": "joint_position",
            "torso": "joint_position",
        }

    @property
    def init_qpos(self):
        # full model has body + 2×7 arm joints
        return np.zeros(len(self._elements["joints"]))

    @property
    def base_xpos_offset(self):
        return {
            "bins":  (-0.5, -0.1, 0),
            "empty": (-0.29,  0,   0),
            "table": lambda table_length: (-0.20 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_eef", "left": "left_eef"}

    def _remove_joint_actuation(self, part_name):
        for joint in self.worldbody.findall(".//joint"):
            if part_name in joint.get("name"):
                parent_body = find_parent(self.worldbody, joint)
                parent_body.remove(joint)
                self._joints.remove(joint.get("name").replace(self.naming_prefix, ""))
        for motor in self.actuator.findall(".//motor"):
            if part_name in motor.get("name"):
                parent_body = find_parent(self.actuator, motor)
                parent_body.remove(motor)
                self._actuators.remove(motor.get("name").replace(self.naming_prefix, ""))
        for motor in self.actuator.findall(".//position"):
            if part_name in motor.get("name"):
                parent_body = find_parent(self.actuator, motor)
                parent_body.remove(motor)
                self._actuators.remove(motor.get("name").replace(self.naming_prefix, ""))
        for fixed in self.tendon.findall(".//fixed"):
            for joint in fixed.findall(".//joint"):
                if part_name in joint.get("joint"):
                    parent_body = find_parent(self.tendon, fixed)
                    parent_body.remove(fixed)
                    break
        # remove joint in equality
        for equality in self.equality.findall(".//joint"):
            # If either joint1 or joint2 is the joint we want to remove
            if part_name in equality.get("joint1") or part_name in equality.get("joint2"):
                parent_body = find_parent(self.equality, equality)
                parent_body.remove(equality)
        for sensor in self.root.findall(".//jointpos"):
            if part_name in sensor.get("joint"):
                find_parent(self.root, sensor).remove(sensor)
        for sensor in self.root.findall(".//jointvel"):
            if part_name in sensor.get("joint"):
                find_parent(self.root, sensor).remove(sensor)
        for sensor in self.root.findall(".//jointactuatorfrc"):
            if part_name in sensor.get("joint"):
                find_parent(self.root, sensor).remove(sensor)

    def _remove_free_joint(self):
        # remove freejoint
        for freejoint in self.worldbody.findall(".//freejoint"):
            find_parent(self.worldbody, freejoint).remove(freejoint)


class RBY1ArmsOnly(RBY1):
    def __init__(self, idn=0):

        # orig = xml_path_completion("robots/rby1/robot.xml")
        # tree = ET.parse(orig)
        # root = tree.getroot()
        #
        # # desired fixed angles (deg) for each joint
        # frozen_joints = {
        #     "torso_0": 0,
        #     "torso_1": -0.6,
        #     "torso_2": 1.2,
        #     "torso_3": -0.6,
        #     "torso_4": 0.0,
        #     "torso_5": 0.0,
        # }
        #
        # for joint in root.iter('joint'):
        #     jname = joint.get('name')
        #     if jname in frozen_joints:
        #         qpos = frozen_joints[jname]
        #         joint.set('range', f'{qpos-1e-8} {qpos}')
        #
        # for joint in root.iter('motor'):
        #     jname = joint.get('joint')
        #     if jname in frozen_joints:
        #         qpos = frozen_joints[jname]
        #         joint.set('ctrlrange', f'{qpos-1e-8} {qpos}')

        # write it next to the original so all assets still resolve
        # patched = os.path.join(os.path.dirname(orig), "robot_patched_torso_x.xml")
        # tree.write(patched)
        patched = xml_path_completion("robots/rby1/robot_patched_torso_x.xml")

        # 3) call the parent with *your* patched XML
        super().__init__(xpath=patched, idn=idn)

        self._remove_free_joint()
        self._remove_joint_actuation("wheel")
        self._remove_joint_actuation("head")
        # self._remove_joint_actuation("torso")


    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {"right": "XRightHand", "left": "XLeftHand"}

    @property
    def default_controller_config(self):
        return {
            "right":     "rby1_two_arm",
            "left":      "rby1_two_arm",
            "head": "joint_position",
            "torso": "joint_position",
        }

    @property
    def init_qpos(self):
        q = np.zeros(20)
        return q


class RBY1ArmsOnlyAria(RBY1ArmsOnly):
    def __init__(self, idn=0):
        patched = xml_path_completion("robots/rby1/robot_patched_torso_x_aria.xml")
        RBY1.__init__(self, xpath=patched, idn=idn)

        self._remove_free_joint()
        self._remove_joint_actuation("wheel")
        self._remove_joint_actuation("head")

    @property
    def default_gripper(self):
        return {"right": "XRightHandAria", "left": "XLeftHand"}
