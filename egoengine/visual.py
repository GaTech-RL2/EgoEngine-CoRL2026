"""Calibrated prepared-scene rendering and the retained visual compositor."""

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .data import aligned_trajectory
from .video import retime, video_info, writer


def camera_quaternion(world_from_cv_camera):
    rotation = np.asarray(world_from_cv_camera)[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def calibrated_model(scene_path, k, source_size, width, height, xml_override=None):
    """Use physical sensor intrinsics, retaining the prepared scene's world frame."""
    scene_path = Path(scene_path)
    root = ET.fromstring(xml_override) if xml_override else ET.parse(scene_path).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for mesh in root.findall("./asset/mesh"):
        if "file" in mesh.attrib:
            mesh.set("file", str((scene_path.parent / meshdir / mesh.get("file")).resolve()))
    for texture in root.findall("./asset/texture"):
        if "file" in texture.attrib:
            texture.set("file", str((scene_path.parent / texture.get("file")).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_ = visual.find("global")
    if global_ is None:
        global_ = ET.SubElement(visual, "global")
    global_.set("offwidth", str(width))
    global_.set("offheight", str(height))
    source_w, source_h = source_size
    fx, fy, cx, cy = k[0, 0] / source_w, k[1, 1] / source_h, k[0, 2] / source_w, k[1, 2] / source_h
    ET.SubElement(
        root.find("worldbody"),
        "camera",
        name="egoengine_calibrated",
        mode="fixed",
        sensorsize="1 1",
        resolution=f"{width} {height}",
        focal=f"{fx} {fy}",
        principal=f"{cx - 0.5} {0.5 - cy}",
        ipd="0",
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def robot_geoms(model):
    roots = {
        int(model.jnt_bodyid[j])
        for j in range(model.njnt)
        if model.joint(j).name
        in ("R_forearm_tx_link_joint", "L_forearm_tx_link_joint", "right_arm_0", "left_arm_0")
    }
    geoms = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and body not in roots:
            body = int(model.body_parentid[body])
        if body in roots:
            geoms.append(geom)
    if not geoms:
        raise ValueError("No robot arm geometry found")
    return np.asarray(geoms)


def render(case, output, action_path=None, width=None, height=None, renderer_type="rby1",
           robot_config=None):
    if robot_config is not None:
        if renderer_type == "xhand":
            raise ValueError("Choose either a visual robot configuration or the xhand debug renderer")
        from .robot_visual import render_robot

        return render_robot(case, output, robot_config, width, height)
    if renderer_type == "rby1":
        from .legacy_visual import render_original

        return render_original(case, output, width, height)
    # Explicit debug renderer for the floating-wrist action simulator only.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = aligned_trajectory(case, action_path)
    poses, k = case.camera(world_frame="action")
    source_video = case.visual / (
        "cropped_video.mp4" if case.info["dataset"] == "Aria" else "color.mp4"
    )
    size = video_info(source_video)
    width = width or size["width"]
    if height is None:
        height = max(2, int(round(width * size["height"] / size["width"] / 2)) * 2)
    model = calibrated_model(case.scene, k, (size["width"], size["height"]), width, height)
    data = mujoco.MjData(model)
    cam = model.camera("egoengine_calibrated").id
    ids = robot_geoms(model)
    paths = {key: output / name for key, name in [
        ("scene", "scene.mp4"), ("foreground", "foreground.mp4"),
        ("mask", "visibility_mask.mp4"),
    ]}
    writers = {key: writer(path, width, height, 1 / case.ref_dt) for key, path in paths.items()}
    pixels = []
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for index in range(len(poses)):
                data.qpos[:] = trajectory["qpos"][index]
                data.qvel[:] = trajectory["qvel"][index]
                model.cam_pos[cam] = poses[index, :3, 3]
                model.cam_quat[cam] = camera_quaternion(poses[index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=cam)
                seg = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                mask = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:, :, 0], ids)
                pixels.append(int(mask.sum()))
                foreground = rgb.copy()
                foreground[~mask] = 0
                writers["scene"].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                writers["foreground"].write(cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR))
                writers["mask"].write(np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2))
    finally:
        for out in writers.values():
            out.release()
    metadata = dict(
        case=case.name, frames=len(poses), fps=1 / case.ref_dt, renderer="xhand",
        backend="action_scene_debug", action_source=trajectory["source"],
        action_sha256=hashlib.sha256(Path(action_path).read_bytes()).hexdigest() if action_path else None,
        camera_convention="camera-to-world OpenCV; converted to MuJoCo axes",
        world_frame="prepared action scene", frames_with_visible_robot=sum(p > 0 for p in pixels),
        render_paths={key: path.name for key, path in paths.items()},
    )
    (output / "render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def composite(case, output, render_dir=None):
    """Reject missing/misaligned stages before running the original compositor."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(render_dir) if render_dir else case.visual
    names = (
        ("foreground.mp4", "visibility_mask.mp4") if render_dir else ("mask.mp4", "mask_bin.mp4")
    )
    paths = [source_dir / names[0], source_dir / names[1], case.visual / "removed_w_mask_5.mp4"]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Prepared visual stage missing: " + ", ".join(missing))
    n = len(case.reference()["qpos"])
    render_metadata = None
    if render_dir:
        render_metadata = json.loads((source_dir / "render.json").read_text())
        if render_metadata["case"] != case.name or render_metadata["frames"] != n:
            raise ValueError("Rendered case/time contract differs from the selected reference")
    for path in paths:
        if video_info(path, decode=True)["frames"] != n:
            raise ValueError(f"Visual/action frame mismatch: {path}; expected {n}")
    from visual import compositor

    parameters = (
        dict(edge_blur=1, edge_erode=1, rim_blend=0.05, rim_noise=0.5, fg_noise=0.1)
        if case.info["dataset"] == "Aria" else {}
    )

    with tempfile.TemporaryDirectory() as tmp:
        ok = compositor.composite_episode(
            str(source_dir),
            str(case.visual),
            out_dir=tmp,
            out_name="result.mp4",
            src_name=names[0],
            mask_name=names[1],
            bg_name="removed_w_mask_5.mp4",
            clone_mode="mixed",
            **parameters,
        )
        if not ok:
            raise RuntimeError("Visual compositor failed")
        retime(Path(tmp) / "result.mp4", output / "visual.mp4", 1 / case.ref_dt, n)
    (output / "visual.json").write_text(
        json.dumps(
            dict(
                case=case.name,
                frames=n,
                fps=1 / case.ref_dt,
                source=(
                    "original_retarget_render"
                    if render_metadata and render_metadata.get("retarget_sha256")
                    else "new_action_render"
                    if render_metadata and render_metadata.get("action_sha256")
                    else "new_reference_render"
                    if render_dir
                    else "prepared_historical_foreground"
                ),
                action_sha256=render_metadata.get("action_sha256") if render_metadata else None,
                retarget_sha256=render_metadata.get("retarget_sha256") if render_metadata else None,
                render_directory=os.path.relpath(source_dir.resolve(), output.resolve())
                if render_dir
                else None,
                backend=render_metadata.get("backend") if render_metadata else "prepared_historical",
                robot_name=render_metadata.get("robot_name") if render_metadata else None,
                robot_config_sha256=render_metadata.get("robot_config_sha256") if render_metadata else None,
                commands_file=render_metadata.get("commands_file") if render_metadata else None,
                compositor_parameters=parameters,
                renderer=render_metadata.get("renderer")
                if render_metadata
                else "prepared_full_robot",
                video_sha256=hashlib.sha256((output / "visual.mp4").read_bytes()).hexdigest(),
                compositor="retained visual/compositor.py; mixed clone mode",
            ),
            indent=2,
        )
        + "\n"
    )
    return output / "visual.mp4"
