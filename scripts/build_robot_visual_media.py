"""Build documentation panels from actual configured-robot outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from egoengine.data import PROJECT_ROOT, get_case
from egoengine.video import read_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/franka_xhand")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/media")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    for name in ("mustard_1", "mustard_2", "mustard_3"):
        case = get_case(name)
        folder = args.root / name
        metadata = json.loads((folder / "visual.json").read_text())
        if metadata.get("robot_name") != "FrankaXHand":
            raise ValueError(f"Expected a newly composited Franka visual: {folder}")
        paths = (
            case.visual / "cropped_video.mp4",
            folder / "render/foreground.mp4",
            folder / "visual.mp4",
        )
        frames = []
        count = len(case.reference()["qpos"])
        for index in np.linspace(0, count - 1, 18).astype(int):
            panel = Image.new("RGB", (672, 256), "#f2f4f5")
            draw = ImageDraw.Draw(panel)
            for column, (label, path) in enumerate(
                zip(("Source RGB", "Franka + original XHand", "Original compositor"), paths)
            ):
                picture = Image.fromarray(read_frame(path, int(index))).resize((224, 224))
                panel.paste(picture, (column * 224, 32))
                draw.text((column * 224 + 7, 8), label, fill="#17232d", font=font)
            frames.append(panel)
        samples = Image.new("RGB", (336, 128 * len(frames)))
        for index, frame in enumerate(frames):
            samples.paste(frame.resize((336, 128)), (0, index * 128))
        palette = samples.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
        gif_frames = [
            frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
        ]
        gif_frames[0].save(
            args.output / f"franka_xhand_{name}.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=200,
            loop=0,
            optimize=True,
        )
        frames[len(frames) // 2].save(args.output / f"franka_xhand_{name}.png")


if __name__ == "__main__":
    main()
