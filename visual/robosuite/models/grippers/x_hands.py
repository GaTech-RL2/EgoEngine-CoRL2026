from __future__ import annotations

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils          import xml_path_completion

class _XHandBase(GripperModel):
    """Common functionality for left / right X‑Hands."""

    # ------------ mandatory robosuite hooks -------------------------------- #
    def format_action(self, action):
        """No scaling: 12 controls → 12 position‑actuated finger joints."""
        assert len(action) == self.dof
        return np.asarray(action)

    # ------------ metadata -------------------------------------------------- #
    @property
    def dof(self) -> int:
        return 12

    @property
    def init_qpos(self):
        return np.zeros(self.dof)

    @property
    def grasp_qpos(self):
        #  −1 → fully open ,  +1 → fully closed
        return {-1: np.full(self.dof, -1.5),
                 1: np.full(self.dof,  1.5)}

    @property
    def speed(self) -> float:
        return 0.15
    
    @property
    def important_sensors(self):
        if isinstance(self, XHandLeft):
            return {
                "force_ee":  "gripper0_left_force_ee",
                "torque_ee": "gripper0_left_torque_ee",
            }
        else:   # XHandRight
            return {
                "force_ee":  "gripper0_right_force_ee",
                "torque_ee": "gripper0_right_torque_ee",
            }

class XHandLeft(_XHandBase):
    """12‑DoF **left** X‑Hand gripper."""

    def __init__(self, idn: int | str = 0):
        super().__init__(xml_path_completion("grippers/x_left_hand.xml"), idn=idn)

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_hand_thumb_bend_link_col",
                "left_hand_thumb_rota_link1_col",
                "left_hand_thumb_rota_link2_col",
            ],
            "right_finger": [
                "left_hand_index_bend_link_col",
                "left_hand_index_rota_link1_col",
                "left_hand_index_rota_link2_col",
                "left_hand_mid_link1_col",
                "left_hand_mid_link2_col",
                "left_hand_ring_link1_col",
                "left_hand_ring_link2_col",
                "left_hand_pinky_link1_col",
                "left_hand_pinky_link2_col",
            ],
            "left_fingerpad": [
                "left_hand_thumb_bend_link_col",
                "left_hand_thumb_rota_link1_col",
                "left_hand_thumb_rota_link2_col",
            ],
            "right_fingerpad": [
                "left_hand_index_bend_link_col",
                "left_hand_index_rota_link1_col",
                "left_hand_index_rota_link2_col",
                "left_hand_mid_link1_col",
                "left_hand_mid_link2_col",
                "left_hand_ring_link1_col",
                "left_hand_ring_link2_col",
                "left_hand_pinky_link1_col",
                "left_hand_pinky_link2_col",
            ],
        }

# --------------------------------------------------------------------------- #
#  RIGHT hand                                                                 #
# --------------------------------------------------------------------------- #
class XHandRight(_XHandBase):
    """12‑DoF **right** X‑Hand gripper."""

    def __init__(self, idn: int | str = 0):
        super().__init__(xml_path_completion("grippers/x_right_hand.xml"), idn=idn)

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "right_hand_thumb_bend_link_col",
                "right_hand_thumb_rota_link1_col",
                "right_hand_thumb_rota_link2_col",
            ],
            "right_finger": [
                "right_hand_index_bend_link_col",
                "right_hand_index_rota_link1_col",
                "right_hand_index_rota_link2_col",
                "right_hand_mid_link1_col",
                "right_hand_mid_link2_col",
                "right_hand_ring_link1_col",
                "right_hand_ring_link2_col",
                "right_hand_pinky_link1_col",
                "right_hand_pinky_link2_col",
            ],
            "left_fingerpad": [
                "right_hand_thumb_bend_link_col",
                "right_hand_thumb_rota_link1_col",
                "right_hand_thumb_rota_link2_col",
            ],
            "right_fingerpad": [
                "right_hand_index_bend_link_col",
                "right_hand_index_rota_link1_col",
                "right_hand_index_rota_link2_col",
                "right_hand_mid_link1_col",
                "right_hand_mid_link2_col",
                "right_hand_ring_link1_col",
                "right_hand_ring_link2_col",
                "right_hand_pinky_link1_col",
                "right_hand_pinky_link2_col",
            ],
        }


class XHandRightAria(XHandRight):
    """12-DoF right X-Hand gripper for Aria-specific visuals."""

    def __init__(self, idn: int | str = 0):
        _XHandBase.__init__(self, xml_path_completion("grippers/x_right_hand_aria.xml"), idn=idn)
