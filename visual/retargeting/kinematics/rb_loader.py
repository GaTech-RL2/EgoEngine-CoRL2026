#!/usr/bin/env python3
# rby1_bimanual_wave_motion.py

import os, sys, math, numpy as np
from isaacgym import gymapi          # import FIRST
from isaacgym import gymtorch        # brings torch in afterwards
import torch                         # safe after gymapi/gymtorch

URDF = "/mnt/data2/dexmimic/workspace/h2s2r/human2sim2robot/human2sim2robot/sim_training/tasks/custom_envs/assets/rbyxhand/model.urdf"

# ─────────────── create sim / env / viewer ───────────────
def make_sim():
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt, sp.substeps, sp.use_gpu_pipeline = 1/60, 2, True
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.physx.use_gpu, sp.physx.num_position_iterations = True, 6
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

    plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0,0,1)
    gym.add_ground(sim, plane)

    warm = gymapi.Vec3(1.0, 0.95, 0.88)
    gym.set_light_parameters(sim, 0, warm, warm, gymapi.Vec3(-.4,-1,-1.3))

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    env    = gym.create_env(sim, gymapi.Vec3(-2,-2,-2), gymapi.Vec3(2,2,2), 1)
    return gym, sim, viewer, env

# ───────────────────── robot wrapper ─────────────────────
class RBY1:
    def __init__(self, gym, sim, env):
        opt = gymapi.AssetOptions(); opt.fix_base_link = True; opt.flip_visual_attachments = True
        asset = gym.load_asset(sim, *os.path.split(URDF), opt)
        self.actor = gym.create_actor(env, asset, gymapi.Transform(), "rby1", 0, 0)

        props = gym.get_actor_dof_properties(env, self.actor)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(50.0); props["damping"].fill(4.0)
        gym.set_actor_dof_properties(env, self.actor, props)

        self.names = list(gym.get_actor_dof_names(env, self.actor))
        self.n     = len(self.names)
        lo, hi     = np.array(props["lower"]), np.array(props["upper"])
        self.center= torch.tensor((lo+hi)/2, device="cuda")
        self.amp   = torch.tensor(0.4*np.ones(self.n), device="cuda")  # ±0.4 rad for movers

        # mover indices
        r_arm  = [i for i,n in enumerate(self.names) if ("right_arm_" in n or "right_hand_" in n)]
        l_arm  = [i for i,n in enumerate(self.names) if ("left_arm_"  in n or "left_hand_"  in n)]
        r_hand = [i for i,n in enumerate(self.names) if  "right_xhand_" in n]
        l_hand = [i for i,n in enumerate(self.names) if  "left_xhand_"  in n]
        self.move = torch.tensor(l_arm+l_hand+r_arm+r_hand, dtype=torch.int64, device="cuda")

        # print diagnostics
        print("ALL DOFs:", self.n)
        print("Controlled:", len(self.move))
        for i in self.move.cpu(): print(self.names[i])

        # immutable index tensor once
        self.all_idx = torch.arange(self.n, dtype=torch.int32, device="cuda")

    def target_tensor(self, t) -> torch.Tensor:
        pos = torch.zeros(self.n, device="cuda")
        phase = torch.linspace(0, 2*math.pi, len(self.move), device="cuda")
        pos[self.move] = self.center[self.move] + self.amp[self.move]*torch.cos(t+phase)
        return pos

# ─────────────────────────── main ────────────────────────
if __name__ == "__main__":
    gym, sim, viewer, env = make_sim()
    robot = RBY1(gym, sim, env)

    gym.prepare_sim(sim)   # prepare AFTER actor creation

    t = 0.0
    while not gym.query_viewer_has_closed(viewer):
        # build GPU tensor for this frame
        tgt = robot.target_tensor(torch.tensor(t, device="cuda"))

        # write targets
        gym.set_dof_position_target_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(tgt),
            gymtorch.unwrap_tensor(robot.all_idx),
            robot.n
        )

        # step sim / render
        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True)
        t += 1/60

    gym.destroy_viewer(viewer); gym.destroy_sim(sim)
