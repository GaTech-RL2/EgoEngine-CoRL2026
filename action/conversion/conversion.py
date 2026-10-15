import numpy as np
from scipy.spatial.transform import Rotation as R


# Fixed rigid transform: right_arm_6 frame -> eef frame
_ARM6_TO_EEF_POS = np.array([0.0, 0.0, -0.102], dtype=np.float64)
_ARM6_TO_EEF_QUAT_WXYZ = np.array([0.0, 0.70710678118, 0.70710678118, 0.0], dtype=np.float64)


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return q_wxyz[[1, 2, 3, 0]]


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return q_xyzw[[3, 0, 1, 2]]


def _pose7_to_T(pose7_wxyz: np.ndarray) -> np.ndarray:
    pose7_wxyz = np.asarray(pose7_wxyz, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose7_wxyz[:3]
    T[:3, :3] = R.from_quat(_wxyz_to_xyzw(pose7_wxyz[3:7])).as_matrix()
    return T


def _T_to_pose7(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = T[:3, 3]
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([p, _xyzw_to_wxyz(q_xyzw)], axis=0)


def _make_T(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    T[:3, :3] = R.from_quat(_wxyz_to_xyzw(q)).as_matrix()
    return T


def _T_arm6_to_eef() -> np.ndarray:
    return _make_T(_ARM6_TO_EEF_POS, _ARM6_TO_EEF_QUAT_WXYZ)


def link_right_arm_6_to_eef(right_arm_6_pose7_wxyz: np.ndarray) -> np.ndarray:
    """right_arm_6 (7D,wxyz) -> eef (7D,wxyz)."""
    T_world_arm6 = _pose7_to_T(right_arm_6_pose7_wxyz)
    T_world_eef = T_world_arm6 @ _T_arm6_to_eef()
    return _T_to_pose7(T_world_eef)


def eef_to_right_arm_6(right_eef_pose7_wxyz: np.ndarray) -> np.ndarray:
    """eef (7D,wxyz) -> right_arm_6 (7D,wxyz)."""
    T_world_eef = _pose7_to_T(right_eef_pose7_wxyz)
    T_world_arm6 = T_world_eef @ np.linalg.inv(_T_arm6_to_eef())
    return _T_to_pose7(T_world_arm6)
