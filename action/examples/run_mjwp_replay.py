from __future__ import annotations

import argparse
import os
import time

import mujoco
from dataclasses import dataclass
from pathlib import Path

from spider.config import Config, process_config
from spider.io import load_data
from spider.simulators.mjwp import (
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    reset_inactive_side_to_ref,
    setup_env,
    setup_mj_model,
    step_env,
    sync_env,
)
from spider.viewers import render_image, setup_viewer, update_viewer


@dataclass
class ReplayArgs:
    dataset_dir: str = "example_datasets"
    dataset_name: str = "gigahand"
    robot_type: str = "xhand"
    embodiment_type: str = "bimanual"
    active_hand: str = "right"
    task: str = "brush_brush_bowl_20231005_186"
    data_id: int = 0
    device: str = "cuda:0"

    sim_dt: float = 0.01
    ctrl_dt: float = 0.05
    horizon: float = 1.6
    knot_dt: float = 0.4

    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.0
    base_pos_rew_scale: float = 0.0
    base_rot_rew_scale: float = 0.0
    vel_rew_scale: float = 0.0
    contact_rew_scale: float = 0.0
    object_pos_threshold: float = 0.05
    object_rot_threshold: float = 1.0
    object_mass_scale: float = 1.0
    show_viewer: bool = False
    viewer: str = "mujoco"
    save_video: bool = False
    video_path: str = ""
    early_termination: int = 1


def parse_args() -> ReplayArgs:
    parser = argparse.ArgumentParser("MJWP replay-only baseline")
    parser.add_argument("--dataset-dir", default="example_datasets")
    parser.add_argument("--dataset-name", default="gigahand")
    parser.add_argument("--robot-type", default="xhand")
    parser.add_argument("--embodiment-type", default="bimanual")
    parser.add_argument("--active-hand", default="right")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--ctrl-dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=1.6)
    parser.add_argument("--knot-dt", type=float, default=0.4)

    parser.add_argument("--pos-rew-scale", type=float, default=1.0)
    parser.add_argument("--rot-rew-scale", type=float, default=0.3)
    parser.add_argument("--joint-rew-scale", type=float, default=0.0)
    parser.add_argument("--base-pos-rew-scale", type=float, default=0.0)
    parser.add_argument("--base-rot-rew-scale", type=float, default=0.0)
    parser.add_argument("--vel-rew-scale", type=float, default=0.0)
    parser.add_argument("--contact-rew-scale", type=float, default=0.0)
    parser.add_argument("--object-pos-threshold", type=float, default=0.05)
    parser.add_argument("--object-rot-threshold", type=float, default=1.0)
    parser.add_argument("--object-mass-scale", type=float, default=1.0)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--viewer", default="mujoco")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--early-termination", type=int, default=1)
    ns = parser.parse_args()
    return ReplayArgs(
        dataset_dir=ns.dataset_dir,
        dataset_name=ns.dataset_name,
        robot_type=ns.robot_type,
        embodiment_type=ns.embodiment_type,
        active_hand=ns.active_hand,
        task=ns.task,
        data_id=ns.data_id,
        device=ns.device,
        sim_dt=ns.sim_dt,
        ctrl_dt=ns.ctrl_dt,
        horizon=ns.horizon,
        knot_dt=ns.knot_dt,
        pos_rew_scale=ns.pos_rew_scale,
        rot_rew_scale=ns.rot_rew_scale,
        joint_rew_scale=ns.joint_rew_scale,
        base_pos_rew_scale=ns.base_pos_rew_scale,
        base_rot_rew_scale=ns.base_rot_rew_scale,
        vel_rew_scale=ns.vel_rew_scale,
        contact_rew_scale=ns.contact_rew_scale,
        object_pos_threshold=ns.object_pos_threshold,
        object_rot_threshold=ns.object_rot_threshold,
        object_mass_scale=ns.object_mass_scale,
        show_viewer=ns.show_viewer,
        viewer=ns.viewer,
        save_video=ns.save_video,
        video_path=ns.video_path,
        early_termination=ns.early_termination,
    )


def _make_config(args: ReplayArgs) -> Config:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir

    cfg = Config(
        simulator="mjwp",
        dataset_dir=os.path.abspath(str(dataset_dir)),
        dataset_name=args.dataset_name,
        robot_type=args.robot_type,
        embodiment_type=args.embodiment_type,
        active_hand=args.active_hand,
        task=args.task,
        data_id=args.data_id,
        seed=0,
        device=args.device,
        sim_dt=args.sim_dt,
        ctrl_dt=args.ctrl_dt,
        horizon=args.horizon,
        knot_dt=args.knot_dt,
        num_samples=1,
        max_num_iterations=1,
        num_dyn=1,
        show_viewer=args.show_viewer,
        viewer=args.viewer if args.show_viewer else "none",
        save_video=args.save_video,
        save_info=False,
        save_rerun=False,
        save_metrics=False,
        save_config=False,
        pos_rew_scale=args.pos_rew_scale,
        rot_rew_scale=args.rot_rew_scale,
        joint_rew_scale=args.joint_rew_scale,
        base_pos_rew_scale=args.base_pos_rew_scale,
        base_rot_rew_scale=args.base_rot_rew_scale,
        vel_rew_scale=args.vel_rew_scale,
        contact_rew_scale=args.contact_rew_scale,
        object_pos_threshold=args.object_pos_threshold,
        object_rot_threshold=args.object_rot_threshold,
        object_mass_scale=args.object_mass_scale,
        use_torch_compile=False,
    )
    cfg = process_config(cfg)

    scene_xml = os.path.join(os.path.dirname(cfg.data_path), "..", "scene.xml")
    scene_eq_xml = os.path.join(os.path.dirname(cfg.data_path), "..", "scene_eq.xml")
    if not os.path.exists(scene_xml) and os.path.exists(scene_eq_xml):
        cfg.model_path = scene_eq_xml
    return cfg




def _set_front_view(mj_model: mujoco.MjModel, viewer) -> None:
    if viewer is None or not hasattr(viewer, "cam"):
        return
    cam = viewer.cam
    front_cam_id = -1
    for cam_name in ("front_camera", "front"):
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        except Exception:
            cam_id = -1
        if cam_id >= 0:
            front_cam_id = cam_id
            break
    if front_cam_id < 0 and getattr(mj_model, "ncam", 0) > 0:
        front_cam_id = 0
    if front_cam_id >= 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = front_cam_id

def _max_step_reward(cfg: Config) -> float:
    pos_term = float(cfg.pos_rew_scale) * float(cfg.object_pos_threshold)
    rot_term = float(cfg.rot_rew_scale) * float(cfg.object_rot_threshold)
    if cfg.embodiment_type == "bimanual":
        if cfg.active_hand in ["right", "left"]:
            return float((pos_term**2 + rot_term**2) ** 0.5)
        return float((2.0 * (pos_term**2 + rot_term**2)) ** 0.5)
    if cfg.embodiment_type in ["right", "left", "humanoid_object"]:
        return float((pos_term**2 + rot_term**2) ** 0.5)
    if cfg.embodiment_type in ["humanoid"]:
        base_pos_term = float(cfg.base_pos_rew_scale) * float(cfg.base_pos_threshold)
        base_rot_term = float(cfg.base_rot_rew_scale) * float(cfg.base_rot_threshold)
        return float((base_pos_term**2 + base_rot_term**2) ** 0.5)
    return 0.0

def main(args: ReplayArgs) -> None:
    cfg = _make_config(args)
    ref_data = load_data(cfg, cfg.data_path)
    env = setup_env(cfg, ref_data)

    mj_model = None
    mj_data = None
    mj_data_ref = None
    run_viewer = None
    if cfg.show_viewer or args.save_video:
        mj_model = setup_mj_model(cfg)
        mj_data = mujoco.MjData(mj_model)
        mj_data_ref = mujoco.MjData(mj_model)
        qpos_ref0, qvel_ref0, ctrl_ref0, _, _ = ref_data
        mj_data.qpos[:] = qpos_ref0[0].detach().cpu().numpy()
        mj_data.qvel[:] = qvel_ref0[0].detach().cpu().numpy()
        mj_data.ctrl[:] = ctrl_ref0[0].detach().cpu().numpy()
        mujoco.mj_step(mj_model, mj_data)
        mj_data.time = 0.0
        if cfg.show_viewer:
            run_viewer = setup_viewer(cfg, mj_model, mj_data)

    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    total_steps = int(qpos_ref.shape[0] - cfg.horizon_steps - cfg.ctrl_steps)
    total_steps = max(total_steps, 0)
    if total_steps == 0:
        raise RuntimeError(f"No valid replay steps for task={args.task}")

    reward_sum = 0.0
    reward_sum_nonterminal = 0.0
    terminal_bonus = 0.0
    reward_last = 0.0
    terminated = False
    termination_step = -1
    frames = []
    renderer = None
    if args.save_video:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception as exc:
            raise RuntimeError("Video saving requires imageio") from exc
        assert mj_model is not None
        mj_model.vis.global_.offwidth = max(int(mj_model.vis.global_.offwidth), 720)
        mj_model.vis.global_.offheight = max(int(mj_model.vis.global_.offheight), 480)
        try:
            renderer = mujoco.Renderer(mj_model, height=480, width=720)
        except Exception:
            renderer = mujoco.Renderer(mj_model, height=360, width=640)
    else:
        imageio = None

    t0 = time.perf_counter()
    steps_executed = 0
    if run_viewer is None:
        from contextlib import nullcontext
        viewer_ctx = nullcontext(type("DummyViewer", (), {"is_running": lambda self: True})())
    else:
        viewer_ctx = run_viewer()

    with viewer_ctx as viewer:
        _set_front_view(mj_model, viewer)
        for sim_step in range(total_steps):
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
            _set_front_view(mj_model, viewer)
            step_env(cfg, env, ctrl_ref[sim_step].unsqueeze(0))
            reset_inactive_side_to_ref(cfg, env, qpos_ref[sim_step], qvel_ref[sim_step])
            ref_step = (
                qpos_ref[sim_step],
                qvel_ref[sim_step],
                ctrl_ref[sim_step],
                contact_ref[sim_step],
                contact_pos_ref[sim_step],
            )
            rew, _ = get_reward(cfg, env, ref_step)
            reward_last = float(rew[0].item())
            reward_sum += reward_last
            reward_sum_nonterminal += reward_last
            steps_executed += 1

            if cfg.show_viewer:
                sync_env(cfg, env, mj_data)
                mj_data.qpos[:] = get_qpos(cfg, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(cfg, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_ref[sim_step].detach().cpu().numpy()
                mj_data.time = steps_executed * cfg.sim_dt
                mj_data_ref.qpos[:] = qpos_ref[min(sim_step, qpos_ref.shape[0] - 1)].detach().cpu().numpy()
                update_viewer(cfg, viewer, mj_model, mj_data, mj_data_ref, {})

            if renderer is not None and sim_step % max(1, int(round(cfg.ref_dt / cfg.sim_dt))) == 0:
                assert mj_data is not None
                assert mj_data_ref is not None
                mj_data.qpos[:] = get_qpos(cfg, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(cfg, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_ref[sim_step].detach().cpu().numpy()
                mj_data.time = steps_executed * cfg.sim_dt
                ref_idx = min(sim_step, qpos_ref.shape[0] - 1)
                mj_data_ref.qpos[:] = qpos_ref[ref_idx].detach().cpu().numpy()
                mj_data_ref.qvel[:] = qvel_ref[ref_idx].detach().cpu().numpy()
                mj_data_ref.ctrl[:] = ctrl_ref[ref_idx].detach().cpu().numpy()
                frames.append(render_image(cfg, renderer, mj_model, mj_data, mj_data_ref))

            term = get_terminate(cfg, env, ref_step)
            if bool(term[0].item()):
                term_rew, _ = get_terminal_reward(cfg, env, ref_step)
                terminal_bonus = float(term_rew[0].item())
                if not terminated:
                    terminated = True
                    termination_step = sim_step
                if int(args.early_termination) != 0:
                    reward_sum += terminal_bonus
                    break

    if imageio is not None and frames:
        video_path = Path(args.video_path) if args.video_path else Path(cfg.output_dir) / "replay.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            str(video_path),
            frames,
            fps=max(1, int(round(1.0 / cfg.ref_dt))),
        )
        print(f"[VIDEO] path={video_path}")
    if renderer is not None:
        renderer.close()

    elapsed = max(time.perf_counter() - t0, 1e-8)
    fps = steps_executed / elapsed
    avg_reward = reward_sum / max(steps_executed, 1)
    status = "early_terminated" if terminated else "finished"
    success = 0 if terminated else 1
    max_step_reward = _max_step_reward(cfg)
    max_total_reward = max(total_steps, 1) * max_step_reward

    print(
        "[REPLAY] "
        f"task={args.task} "
        f"status={status} "
        f"fps={fps:.2f} "
        f"reward_actual={reward_sum_nonterminal:.6f} "
        f"reward_max={max_total_reward:.6f} "
        f"steps_actual={steps_executed} "
        f"steps_max={total_steps} "
        f"success={success} "
        f"reward_total={reward_sum:.6f} "
        f"reward_avg={avg_reward:.6f} "
        f"steps={steps_executed}/{total_steps} "
        f"termination_step={termination_step} "
        f"reward_last={reward_last:.6f} "
        f"terminal_bonus={terminal_bonus:.6f}"
    )


if __name__ == "__main__":
    main(parse_args())
