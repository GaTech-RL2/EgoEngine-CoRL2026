"""CoDesign's joint/tip/object goals, resolved from the actual MuJoCo scene.

One active hand: q, tip positions, object pose, goal tip poses, goal q,
goal object pose (2 * hand_nq + 34 values). All poses use world coordinates
and wxyz quaternions, as in the prepared references.
"""

import mujoco
import numpy as np
import torch

from spider.rl.state_layout import bimanual_object_slice


class GoalObservation:
    def __init__(self, model, qpos_ref, embodiment_type, active_hand):
        self.nq_obj = 14 if embodiment_type == "bimanual" else 7
        hand_nq = model.nq - self.nq_obj
        sides = [active_hand] if active_hand in ("right", "left") else ["right", "left"]
        self.hand_slice = slice(0, hand_nq)
        self.object_slice = slice(model.nq - self.nq_obj, model.nq)
        if embodiment_type == "bimanual" and len(sides) == 1:
            half = hand_nq // 2
            self.hand_slice = slice(0, half) if active_hand == "right" else slice(half, hand_nq)
            relative = bimanual_object_slice(model, active_hand)
            start = model.nq - self.nq_obj + relative.start
            self.object_slice = slice(start, start + 7)
        self.site_ids = [model.site(f"{side}_{finger}_tip").id
                         for side in sides for finger in ("thumb", "index")]
        reference = qpos_ref.detach().cpu().numpy()
        poses = np.empty((len(reference), len(self.site_ids), 7), dtype=np.float32)
        data = mujoco.MjData(model)
        for frame, qpos in enumerate(reference):
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            for column, site_id in enumerate(self.site_ids):
                poses[frame, column, :3] = data.site_xpos[site_id]
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
                poses[frame, column, 3:] = quat if quat[0] >= 0 else -quat
        self.reference_tips = torch.as_tensor(poses, device=qpos_ref.device)
        self.reference_qpos = qpos_ref

    def build(self, qpos, site_xpos, phase):
        phase = torch.as_tensor(phase, device=qpos.device, dtype=torch.long).reshape(-1)
        phase = phase.clamp(0, len(self.reference_qpos) - 1)
        goal = self.reference_qpos.index_select(0, phase)
        tips = self.reference_tips.index_select(0, phase)
        return torch.cat([
            qpos[:, self.hand_slice],
            site_xpos[:, self.site_ids].flatten(1),
            qpos[:, self.object_slice],
            tips.flatten(1),
            goal[:, self.hand_slice],
            goal[:, self.object_slice],
        ], dim=-1).to(torch.float32)


def restore_masked_state(data, snapshot, mask, to_tensor):
    """Restore only finished worlds, including integration/runtime state."""
    values = dict(snapshot.state_fields, qpos=snapshot.qpos, qvel=snapshot.qvel,
                  ctrl=snapshot.ctrl, time=snapshot.time)
    for name, value in values.items():
        target = to_tensor(getattr(data, name))
        target[mask] = value
