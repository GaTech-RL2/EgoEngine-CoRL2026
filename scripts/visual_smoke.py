"""Join the five-frame CUDA check to new RGB and paired HDF5 observations."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from egoengine.data import DEFAULT_DATA_ROOT, PROJECT_ROOT, get_case
from egoengine.export import export_dataset
from egoengine.video import video_info, writer
from egoengine.visual import composite, render


def first_frames(source, destination, count, fps):
    info = video_info(source)
    cap = cv2.VideoCapture(str(source))
    out = writer(destination, info["width"], info["height"], fps)
    try:
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Source video has fewer than {count} frames: {source}")
            out.write(frame)
    finally:
        cap.release()
        out.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=PROJECT_ROOT / "outputs/gpu_smoke_reproduction/smoke_rl.npz",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/visual_smoke")
    args = parser.parse_args()
    source = get_case("mustard_1", args.data_root)
    output = args.output.resolve()
    fixture = output / "inputs"
    count = 5
    info = dict(source.info)
    info.update(note="Five-frame implementation check; no task-quality claim")
    processed = fixture / info["processed_dir"]
    visual = fixture / info["visual_dir"]
    visual.mkdir(parents=True, exist_ok=True)
    assets = fixture / "assets"
    if not assets.exists():
        assets.symlink_to(source.root / "assets", target_is_directory=True)
    shutil.copytree(source.processed, processed, dirs_exist_ok=True)
    reference = source.reference()
    full_count = len(reference["qpos"])
    np.savez(
        processed / "0/trajectory_kinematic.npz",
        **{
            key: value[:count] if value.ndim and len(value) == full_count else value
            for key, value in reference.items()
        },
    )
    retarget = json.loads((source.visual / "retarget.json").read_text())
    poses, k = source.camera()
    warmup = int(retarget.get("warm_up_steps", 0))
    if len(retarget["traj"]) == full_count + warmup:
        retarget["traj"] = retarget["traj"][warmup:]
    retarget["traj"] = retarget["traj"][:count]
    retarget["warm_up_steps"] = 0
    retarget["camera_traj"] = {"target2camera": poses[:count].tolist(), "cam_K": k.tolist()}
    info["camera_convention"] = "camera_to_world"
    (visual / "retarget.json").write_text(json.dumps(retarget) + "\n")
    np.savetxt(visual / "egocentric_intrinsic.txt", k)
    for name in ["cropped_video.mp4", "removed_w_mask_5.mp4"]:
        first_frames(source.visual / name, visual / name, count, 1 / source.ref_dt)
    (fixture / "manifest.json").write_text(json.dumps({"cases": [info]}, indent=2) + "\n")
    case = get_case(source.name, fixture)
    trajectory = args.trajectory.resolve()
    render(case, output / "render")
    video = composite(case, output, render_dir=output / "render")
    dataset = export_dataset(
        case,
        video,
        output / "dataset.hdf5",
        action_path=trajectory,
        visual_source="original_retarget_render",
    )
    with h5py.File(dataset) as f:
        demo = f["data/demo_0"]
        shapes = {
            name: list(demo[name].shape)
            for name in ["actions", "obs/qpos", "obs/qvel", "obs/agentview_image", "obs/rby1_joint_commands"]
        }
        for name in shapes:
            if shapes[name][0] != count or not np.isfinite(demo[name][:]).all():
                raise ValueError(f"Invalid paired dataset: {name}")
        np.testing.assert_allclose(demo["timestamps"][:], np.arange(count) * case.ref_dt)
        if demo.attrs["action_source"] != "adaptive_physics":
            raise ValueError("Expected newly generated physics states")
    report = {
        "scope": "Five-frame original RB-Y1 visual replay paired with refined EEF actions; no task-quality claim",
        "frames": count,
        "shapes": shapes,
        "action_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "hdf5_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "visual_source": "original_retarget_render",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
