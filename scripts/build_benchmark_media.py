"""Check refined EEF data paired with original RB-Y1 visual replay."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def tile(frame):
    return ImageOps.pad(Image.fromarray(frame), (320, 320), color="white",
                        method=Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports, hashes = [], {}
    for folder in sorted(args.paired.iterdir()):
        if not (folder / "dataset.hdf5").is_file():
            continue
        run = args.runs / f"codesign__{folder.name}__seed{args.seed}"
        report = json.loads((run / "report.json").read_text())
        if not report["validated_complete"]:
            raise ValueError(f"Cannot present incomplete trajectory as paired success: {folder.name}")
        metadata = json.loads((folder / "render/render.json").read_text())
        if (metadata["renderer"] != "rby1" or metadata["action_source"] != "original_retarget"
                or metadata.get("backend") != "original_robosuite_replay"):
            raise ValueError("Expected original RB-Y1 configuration replay for the visual branch")
        with h5py.File(folder / "dataset.hdf5") as f:
            demo = f["data/demo_0"]
            count, dt = int(demo.attrs["num_samples"]), float(demo.attrs["ref_dt"])
            names = ["actions", "obs/qpos", "obs/qvel", "obs/agentview_image", "timestamps"]
            shapes = {name: list(demo[name].shape) for name in names}
            if any(s[0] != count for s in shapes.values()):
                raise ValueError("Paired arrays have different lengths")
            if not all(np.isfinite(demo[name][:]).all() for name in names):
                raise ValueError("Nonfinite paired data")
            if demo.attrs["action_source"] != "adaptive_physics":
                raise ValueError("Expected actual optimized action states")
            np.testing.assert_allclose(demo["timestamps"][:], np.arange(count) * dt)
        scene = read_frames(folder / "render/scene.mp4")
        visual = read_frames(folder / "visual.mp4")
        masks = read_frames(folder / "render/visibility_mask.mp4")
        if any(len(frames) != count for frames in (scene, visual, masks)):
            raise ValueError("A generated video has a different frame count")
        images = []
        for i in range(0, count, 3):
            image = Image.new("RGB", (640, 342), "white")
            draw = ImageDraw.Draw(image)
            draw.text((8, 5), "RB-Y1: original action replay", fill="black")
            draw.text((328, 5), "Composited visual observation", fill="black")
            image.paste(tile(scene[i]), (0, 22))
            image.paste(tile(visual[i]), (320, 22))
            images.append(image)
        gif = args.output / f"{folder.name}.gif"
        images[0].save(gif, save_all=True, append_images=images[1:],
                       duration=int(round(dt * 3000)), loop=0)
        hashes[gif.name] = hashlib.sha256(gif.read_bytes()).hexdigest()
        stages = Image.new("RGB", (640, 686), "white")
        draw = ImageDraw.Draw(stages)
        for column, i in enumerate(np.linspace(0, count - 1, 4).round().astype(int)):
            x, y = (column % 2) * 320, (column // 2) * 343
            draw.text((x + 8, y + 5), f"Frame {i} / {count - 1} | {i * dt:.2f}s", fill="black")
            stages.paste(tile(visual[i]), (x, y + 23))
        stage_path = args.output / f"{folder.name}_stages.png"
        stages.save(stage_path)
        hashes[stage_path.name] = hashlib.sha256(stage_path.read_bytes()).hexdigest()
        reports.append({"case": folder.name, "frames": count, "shapes": shapes,
                        "seed": args.seed, "renderer": "rby1",
                        "backend": "original_robosuite_replay", "gif": gif.name,
                        "stages": stage_path.name,
                        "frames_with_visible_hand": sum(bool(np.any(m > 127)) for m in masks),
                        "action_sha256": hashlib.sha256((run / "mixed.npz").read_bytes()).hexdigest(),
                        "dataset_sha256": hashlib.sha256((folder / "dataset.hdf5").read_bytes()).hexdigest(),
                        "video_sha256": hashlib.sha256((folder / "visual.mp4").read_bytes()).hexdigest()})
    (args.output / "manifest.json").write_text(
        json.dumps({"scope": "Refined EEF actions + independent original RB-Y1 visual replay",
                    "cases": reports, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps({"checked_paired_cases": len(reports),
                      "total_frames": sum(r["frames"] for r in reports)}))


if __name__ == "__main__":
    main()
