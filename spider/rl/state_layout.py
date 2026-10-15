"""Resolve object coordinates from scene joints instead of assuming object order."""

import mujoco


def bimanual_object_slice(model, side):
    if side not in ("right", "left"):
        raise ValueError(f"Unknown active side: {side}")
    joint = model.joint(f"{side}_object_joint")
    if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected a seven-coordinate object free joint")
    start = int(model.jnt_qposadr[joint.id]) - (model.nq - 14)
    if start not in (0, 7):
        raise ValueError("Expected two object free joints at the end of scene qpos")
    return slice(start, start + 7)
