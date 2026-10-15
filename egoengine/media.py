"""Rebuild documentation figures and GIFs from actual prepared videos."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .data import PROJECT_ROOT, cases
from .video import read_frame, video_info
from .visual import composite, render


def font(size):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def panel(frames, labels, columns=3, title="", subtitle="", cell=240):
    rows = (len(frames) + columns - 1) // columns
    margin, gap, header, label = 18, 12, 64, 30
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + columns * cell + (columns - 1) * gap,
            header + margin + rows * (cell + label + gap),
        ),
        "#f2f5f9",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), title, fill="#172a44", font=font(21))
    draw.text((margin, 38), subtitle, fill="#4a5b70", font=font(12))
    for index, (frame, name) in enumerate(zip(frames, labels)):
        x = margin + (index % columns) * (cell + gap)
        y = header + (index // columns) * (cell + label + gap)
        draw.rounded_rectangle((x, y, x + cell, y + cell + label), radius=8, fill="white")
        im = Image.fromarray(frame)
        im = ImageOps.pad(im, (cell, cell), color="#101820", method=Image.Resampling.LANCZOS)
        canvas.paste(im, (x, y + label))
        draw.text((x + 8, y + 7), name, fill="#172a44", font=font(12))
    return canvas


def pipeline_svg(output):
    nodes = [
        (
            30,
            110,
            200,
            88,
            "Prepared inputs",
            "Scene · poses · calibration",
            "Background already removed",
            "#e6eefb",
        ),
        (
            285,
            35,
            235,
            84,
            "Action branch",
            "Replay → MPC → RL",
            "Plan 2 chunks · commit 1",
            "#e8f4ef",
        ),
        (
            575,
            35,
            235,
            84,
            "Aligned state export",
            "qpos · qvel · controls",
            "Reference-frame timestamps",
            "#e8f4ef",
        ),
        (
            575,
            180,
            235,
            84,
            "Visual branch",
            "Original RB-Y1 actions + mask",
            "Occlusion from scene geometry",
            "#fff1df",
        ),
        (
            865,
            180,
            235,
            84,
            "Composite",
            "Robot + removed background",
            "Retained visual compositor",
            "#fff1df",
        ),
        (
            1155,
            110,
            230,
            88,
            "Paired dataset",
            "RGB · state · action · time",
            "HDF5 + audit JSON",
            "#e9e7fa",
        ),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1420" height="330" viewBox="0 0 1420 330">',
        '<rect width="1420" height="330" fill="#f7f9fc"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#58708c"/></marker></defs>',
        '<g fill="none" stroke="#58708c" stroke-width="2" marker-end="url(#arrow)">',
        '<path d="M230 140 L255 140 L255 77 L285 77"/><path d="M520 77 H575"/>',
        '<path d="M230 174 L270 174 L270 222 H575"/>',
        '<path d="M810 222 H865"/><path d="M1100 222 H1124 V166 H1155"/>',
        '<path d="M810 77 H1124 V138 H1155"/></g>',
    ]
    for x, y, w, h, title, l1, l2, color in nodes:
        parts += [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#c8d4e3"/>',
            f'<text x="{x + 14}" y="{y + 25}" font-family="sans-serif" font-size="17" font-weight="600" fill="#172a44">{title}</text>',
            f'<text x="{x + 14}" y="{y + 47}" font-family="sans-serif" font-size="13" fill="#304966">{l1}</text>',
            f'<text x="{x + 14}" y="{y + 66}" font-family="sans-serif" font-size="12" fill="#4a5b70">{l2}</text>',
        ]
    parts += [
        '<text x="30" y="310" font-family="sans-serif" font-size="13" fill="#4a5b70">Prepared inputs feed both branches independently. Visual: original RB-Y1 replay. Action: floating-wrist refinement.</text>',
        "</svg>",
    ]
    (output / "pipeline.svg").write_text("\n".join(parts))


def build_media(data_root, output, render_scenes=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pipeline_svg(output)
    records = []
    thumbs = []
    for case in cases(data_root):
        scene_dir = PROJECT_ROOT / "outputs/documentation_media" / case.name / "render"
        if render_scenes:
            render(case, scene_dir, renderer_type="rby1")
        replay_visible = (
            json.loads((scene_dir / "render.json").read_text())["frames_with_visible_robot"]
            if render_scenes else None
        )
        if case.info["dataset"] == "Aria":
            visual = composite(
                case,
                PROJECT_ROOT / "outputs/documentation_media" / case.name,
                scene_dir if render_scenes else None,
            )
            paths = [
                case.visual / "cropped_video.mp4",
                case.visual / "removed_w_mask_5.mp4",
                scene_dir / "foreground.mp4" if render_scenes else case.visual / "mask.mp4",
                scene_dir / "visibility_mask.mp4"
                if render_scenes
                else case.visual / "mask_bin.mp4",
                visual,
            ]
            labels = [
                "Prepared RGB",
                "Removed background",
                "New RB-Y1 robot layer" if render_scenes else "Prepared robot layer",
                "Visibility mask",
                "New compositor output",
            ]
            if render_scenes:
                paths.append(scene_dir / "scene.mp4")
                labels.append("New full reference scene")
            columns = 3
            subtitle = (
                "Matched reference frames · full RB-Y1 render + compositor re-run"
                if render_scenes
                else "Matched reference frames · historical foreground + compositor re-run"
            )
            aligned = True
        else:
            paths = [
                case.visual / name
                for name in (
                    "color.mp4",
                    "historical_ref.mp4",
                    "historical_rl.mp4",
                    "historical_inpainted_video.mp4",
                )
            ]
            labels = [
                "Prepared RGB",
                "Historical reference",
                "Historical RL",
                "Historical visual output",
            ]
            if render_scenes:
                paths.extend([scene_dir / "scene.mp4", scene_dir / "visibility_mask.mp4"])
                labels.extend(["New full RB-Y1 reference", "New visibility mask"])
            columns = 3 if render_scenes else 2
            subtitle = (
                "Historical clips + new reference render · preview clocks differ · background missing"
                if render_scenes
                else "Historical clips · preview clocks differ · background missing"
            )
            aligned = False
        if render_scenes and not replay_visible:
            labels[-2] = "Original retarget: robot out of view"
            subtitle = "Original retarget: robot out of view; empty mask; historical clips retained"
        infos = [video_info(p) for p in paths]
        count = min(info["frames"] for info in infos) if aligned else 36
        positions = np.linspace(0, 1, 36)
        frames = []
        for position in positions:
            indices = [
                min(
                    info["frames"] - 1,
                    int(round(position * (count - 1 if aligned else info["frames"] - 1))),
                )
                for info in infos
            ]
            frames.append(
                panel(
                    [read_frame(path, index) for path, index in zip(paths, indices)],
                    labels,
                    columns,
                    f"{case.info['dataset']} / {case.name}",
                    subtitle,
                )
            )
        still = frames[len(frames) // 2]
        still.save(output / f"{case.name}_stages.png")
        palette = frames[len(frames) // 2].quantize(colors=128)
        gif = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        gif_path = output / f"{case.name}.gif"
        gif[0].save(
            gif_path, save_all=True, append_images=gif[1:], duration=100, loop=0, optimize=True
        )
        thumbs.append((np.asarray(still), case.name))
        records.append(
            dict(
                case=case.name,
                outputs=[gif_path.name, f"{case.name}_stages.png"],
                sources=[
                    str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
                    for p in paths
                ],
                labels=labels,
                alignment="same reference frame"
                if aligned
                else "independent normalized clip progress; not a paired timeline",
                gif_frames=36,
                gif_display_fps=10,
                output_scope=(
                    "new full RB-Y1 reference render and compositor"
                    if render_scenes
                    else "CPU compositor re-run"
                )
                if aligned
                else "retained historical output with reference-scene inspection; no new optimization claim",
            )
        )
        print(f"[media] {case.name}: stage panel and 36-frame GIF", flush=True)
    gallery = panel(
        [x[0] for x in thumbs],
        [x[1] for x in thumbs],
        columns=3,
        title="EgoEngine / six prepared examples",
        subtitle="TACO: three interactions · Aria: three Mustard demonstrations",
        cell=360,
    )
    gallery.save(output / "examples.png")
    artifact_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.suffix in (".gif", ".png", ".svg")
    }
    (output / "manifest.json").write_text(
        json.dumps(
            dict(cases=records, render_scenes=render_scenes, sha256=artifact_hashes), indent=2
        )
        + "\n"
    )
    return records
