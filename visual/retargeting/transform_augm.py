import numpy as np

def _hat(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

def exp_so3(w):
    """Exponential map so(3)->SO(3) for a small axis-angle vector w (rad)."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        # 1st-order for tiny angles
        return np.eye(3) + _hat(w)
    K = _hat(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def perturb_rotation(R, sigma_rot_deg=2.0, rng=None, left_multiply=True):
    """
    Apply small rotational noise to R.
    left_multiply=True -> world-frame noise: R' = R_noise @ R
    left_multiply=False -> body-frame noise:  R' = R @ R_noise
    """
    rng = np.random.default_rng(rng)
    w = rng.normal(0.0, np.deg2rad(sigma_rot_deg), size=3)   # small axis-angle (rad)
    R_noise = exp_so3(w)
    return (R_noise @ R) if left_multiply else (R @ R_noise)

def perturb_translation(t, sigma_t=0.005, rng=None):
    """Add small translation noise (meters)."""
    rng = np.random.default_rng(rng)
    return t + rng.normal(0.0, sigma_t, size=3)

def perturb_se3(R, t, sigma_rot_deg=2.0, sigma_t=0.005, rng=None, left_multiply=True):
    """Perturb a pose given by (R, t)."""
    Rp = perturb_rotation(R, sigma_rot_deg, rng, left_multiply)
    tp = perturb_translation(t, sigma_t, rng)
    return Rp, tp

def perturb_T(T, sigma_rot_deg=2.0, sigma_t=0.005, rng=None, left_multiply=True):
    """Perturb a 4x4 homogeneous transform."""
    R, t = T[:3, :3], T[:3, 3]
    Rp, tp = perturb_se3(R, t, sigma_rot_deg, sigma_t, rng, left_multiply)
    Tp = np.eye(4)
    Tp[:3, :3] = Rp
    Tp[:3, 3]  = tp
    return Tp


def rot_angle_from_mats(R1, R2, degrees=True):
    """Geodesic angle between two rotation matrices."""
    R_rel = R2 @ R1.T
    cos_th = (np.trace(R_rel) - 1.0) * 0.5
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    return np.degrees(th) if degrees else th


def yaw_diff_from_mats(R1, R2, degrees=True):
    """
    Signed angle (R2 relative to R1) around world Z in (-pi, pi].
    Assumes R maps body->world. If your convention is world->body, swap 1/2.
    """
    R_rel = R2 @ R1.T
    yaw = np.arctan2(R_rel[1,0], R_rel[0,0])  # ZYX yaw of R_rel
    return np.degrees(yaw) if degrees else yaw