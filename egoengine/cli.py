"""One entry point for the merged action and visual branches."""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA_ROOT, PROJECT_ROOT, cases, get_case


def validate(root, output=None, decode=True):
    import mujoco

    from .video import video_info

    results = []
    for case in cases(root):
        ref = case.reference()
        models = []
        for name in ("scene.xml", "scene_eq.xml"):
            model = mujoco.MjModel.from_xml_path(str(case.processed / name))
            if (
                ref["qpos"].shape[1] != model.nq
                or ref["ctrl"].shape[1] != model.nu
                or ref["qvel"].shape[1] != model.nv
            ):
                raise ValueError(f"Scene/reference dimensions differ: {case.name}")
            data = mujoco.MjData(model)
            data.qpos[:] = ref["qpos"][0]
            data.qvel[:] = ref["qvel"][0]
            data.ctrl[:] = ref["ctrl"][0]
            mujoco.mj_forward(model, data)
            for _ in range(5):
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise ValueError("Nonfinite CPU simulation")
            models.append(name)
        case.camera()
        videos = {p.name: video_info(p, decode=decode) for p in sorted(case.visual.glob("*.mp4"))}
        results.append(
            dict(
                case=case.name,
                frames=len(ref["qpos"]),
                models=models,
                cpu_steps=5,
                camera_aligned=True,
                visual_replay_issue=case.info.get("visual_replay_issue"),
                visual_inputs_complete=case.info["visual_inputs_complete"],
                videos=videos,
            )
        )
        print(
            f"[validate] {case.name}: {len(ref['qpos'])} frames, scene dimensions and camera frame count agree",
            flush=True,
        )
        if case.info.get("visual_replay_issue"):
            print(f"[visual input] {case.info['visual_replay_issue']}", flush=True)
    report = dict(
        cases=results,
        scope="Prepared assets/arrays/camera/video and five CPU steps; no full-task success claim",
    )
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_action(case, output, args):
    required = ["torch", "mujoco_warp", "warp"]
    if "rl" in args.methods.split(","):
        required.extend(["rsl_rl", "tensordict"])
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            "Missing action dependencies: "
            + ", ".join(missing)
            + ". Install requirements/action-tested.txt in a CUDA environment."
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "action.examples.run_mjwp_mixed",
        "--dataset-dir",
        str(case.root),
        "--dataset-name",
        case.info["dataset_name"],
        "--task",
        case.name,
        "--robot-type",
        case.info["robot_type"],
        "--embodiment-type",
        case.info["embodiment_type"],
        "--active-hand",
        "right",
        "--debug-output-dir",
        str(output),
        "--output-name",
        "trajectory_mjwp_mixed.npz",
        "--methods",
        args.methods,
        "--device",
        args.device,
        "--window-size",
        str(args.chunk_size),
        "--lookahead-chunks",
        "2",
    ]
    if args.action_args:
        command.extend(shlex.split(args.action_args))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)
    path = output / "trajectory_mjwp_mixed.npz"
    with np.load(path, allow_pickle=False) as f:
        if not bool(f["success"].item()):
            raise RuntimeError(
                f"All modes failed before the full trajectory completed; inspect {path.with_suffix('.json')}"
            )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egoengine", description="Unified EgoEngine prepared-input workflow"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List six prepared cases and input completeness")
    sub.add_parser("doctor", help="Check base and optional CUDA dependencies")
    check = sub.add_parser("validate", help="Load scenes, arrays, camera tracks and videos")
    check.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/validation.json")
    check.add_argument(
        "--fast", action="store_true", help="Read video headers instead of decoding all frames"
    )
    for name in ("demo", "action", "render", "composite", "export", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--case", default="mustard_1")
        cmd.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
        if name in ("demo", "render", "run"):
            cmd.add_argument("--robot-config", type=Path,
                             help="Visual arm configuration; preserves the recorded XHand commands")
        if name in ("render", "export"):
            cmd.add_argument(
                "--trajectory", type=Path,
                help="Refined EEF actions for export or xhand debugging; RB-Y1 uses retarget.json",
            )
        if name in ("action", "run"):
            cmd.add_argument("--methods", default="replay,mpc,rl")
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--chunk-size", type=int, default=20)
            cmd.add_argument(
                "--action-args",
                default="",
                help="Additional mixed-runner options as one quoted string",
            )
        if name in ("render", "run"):
            cmd.add_argument("--width", type=int, default=None, help="Default: original replay/source width")
            cmd.add_argument(
                "--height",
                type=int,
                default=None,
                help="Default: preserve source image aspect ratio",
            )
            cmd.add_argument(
                "--renderer", choices=["rby1", "xhand"],
                default="rby1",
                help="RB-Y1 replays original retarget actions; xhand is action-scene debugging",
            )
        if name in ("composite",):
            cmd.add_argument("--render-dir", type=Path,
                             help="New rendered layers; omit to use the prepared historical foreground")
        if name == "export":
            cmd.add_argument("--video", type=Path, required=True)
            cmd.add_argument(
                "--visual-source",
                choices=["prepared", "original_retarget_render", "new_action_render", "reference_scene"],
                default="prepared",
            )
        if name in ("demo", "export", "run"):
            cmd.add_argument("--image-size", type=int, default=256)
    media = sub.add_parser(
        "media", help="Regenerate README stage panels, six GIFs and pipeline diagram"
    )
    media.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/media")
    media.add_argument(
        "--render-scenes",
        action="store_true",
        help="Re-render full RB-Y1 references and use their foreground/masks in stage figures",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            modules = [
                "numpy",
                "mujoco",
                "cv2",
                "h5py",
                "PIL",
                "scipy",
                "torch",
                "mujoco_warp",
                "warp",
                "rsl_rl",
                "tensordict",
            ]
            print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in modules}, indent=2))
            return
        if args.command == "list":
            for case in cases(args.data_root):
                print(
                    f"{case.info['dataset']:5} {case.name:40} visual_inputs={case.info['visual_inputs_complete']}"
                )
            return
        if args.command == "validate":
            validate(args.data_root, args.output, decode=not args.fast)
            return
        if args.command == "media":
            from .media import build_media

            build_media(args.data_root, args.output, render_scenes=args.render_scenes)
            return
        case = get_case(args.case, args.data_root)
        output = args.output.resolve() / case.name
        output.mkdir(parents=True, exist_ok=True)
        if args.command == "run" and not (case.visual / "removed_w_mask_5.mp4").is_file():
            raise FileNotFoundError(
                "Full visual generation needs a prepared demonstrator-free background. Use `action` and `render` separately for TACO inspection."
            )
        if args.command in ("action", "run"):
            path = run_action(case, output / "action", args)
            if args.command == "action":
                return
        from .export import export_dataset
        from .visual import composite, render

        if args.command == "demo":
            if not case.info["visual_inputs_complete"]:
                raise ValueError(
                    "TACO lacks prepared backgrounds; use `render` for a reference-scene inspection. Historical videos are available in `media`."
                )
            render(case, output / "render", robot_config=args.robot_config)
            video = composite(case, output, output / "render")
            dataset = export_dataset(
                case, video, output / "dataset.hdf5", image_size=args.image_size
            )
            print(f"[demo] {video}\n[dataset] {dataset}")
        elif args.command == "render":
            render(case, output / "render", args.trajectory, args.width, args.height, args.renderer,
                   robot_config=args.robot_config)
        elif args.command == "composite":
            composite(case, output, args.render_dir)
        elif args.command == "export":
            export_dataset(
                case,
                args.video,
                output / "dataset.hdf5",
                args.trajectory,
                args.image_size,
                args.visual_source,
            )
        elif args.command == "run":
            render(case, output / "render", path if args.renderer == "xhand" else None,
                   args.width, args.height, args.renderer, robot_config=args.robot_config)
            video = composite(case, output, output / "render")
            export_dataset(
                case,
                video,
                output / "dataset.hdf5",
                path,
                args.image_size,
                visual_source="original_retarget_render" if args.renderer == "rby1" else "new_action_render",
            )
            print(f"[run] {output}")
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"egoengine: {exc}\n")
