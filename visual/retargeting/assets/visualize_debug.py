#!/usr/bin/env python3
"""
MuJoCo viewer for:
  - MJCF XML (.xml) files
  - URDF (.urdf) files

Features:
  - Keeps "action" = all zeros (data.ctrl[:] = 0) every step
  - Adds good lighting for visibility (headlight + a world light for MJCF)
  - For URDF: tries direct load; if it fails due to invalid inertia eigenvalues,
    it retries by injecting MuJoCo URDF-extension directives IN MEMORY (no file edits):
      * compiler inertiafromgeom="true"
      * compiler balanceinertia="true"
      * compiler meshdir="<urdf_dir>"
      * adds a world light
    This preserves your on-disk URDF exactly.

Usage:
  python mujoco_view.py /path/to/model.xml
  python mujoco_view.py /path/to/model.urdf

Options:
  --no-gravity     Disable gravity
  --dt 0.01        Viewer loop sleep
"""

import argparse
import os
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer


def _ensure_white_room(worldbody: ET.Element, group: str = "1"):
    """
    Create a simple white room: floor + 4 walls + ceiling, and multiple white lights.
    Non-colliding (contype/conaffinity = 0) so it won't interfere with the robot.
    Uses group=1 so it remains visible even if group 0 is hidden in the viewer.
    Idempotent via unique geom/light names.
    """
    def _has_child(tag: str, name: str) -> bool:
        for c in list(worldbody):
            if c.tag == tag and c.get("name") == name:
                return True
        return False

    def _add_geom(name: str, **attrs):
        if _has_child("geom", name):
            return
        g = ET.SubElement(worldbody, "geom")
        g.set("name", name)
        for k, v in attrs.items():
            g.set(k, v)
        # make non-colliding by default, unless caller overrides
        g.set("contype", attrs.get("contype", "0"))
        g.set("conaffinity", attrs.get("conaffinity", "0"))
        g.set("group", attrs.get("group", group))

    def _add_light(name: str, **attrs):
        if _has_child("light", name):
            return
        l = ET.SubElement(worldbody, "light")
        l.set("name", name)
        for k, v in attrs.items():
            l.set(k, v)

    # Room dimensions
    # Centered at origin, floor at z=0. Robot typically sits near z>=0.
    half_w = 8.0     # half-width in x
    half_d = 8.0     # half-depth in y
    half_h = 4.0     # half-height in z (ceiling at z=2*half_h)
    wall_th = 0.05

    white_rgba = "0.1 0.1 0.1 1"

    # Floor (plane)
    _add_geom(
        "__white_floor",
        type="plane",
        pos="0 0 0",
        size=f"{half_w} {half_d} 0.1",
        rgba=white_rgba,
    )

    # Ceiling (thin box)
    _add_geom(
        "__white_ceiling",
        type="box",
        pos=f"0 0 {2.0 * half_h}",
        size=f"{half_w} {half_d} {wall_th}",
        rgba=white_rgba,
    )

    # Walls: +Y, -Y, +X, -X (thin boxes)
    _add_geom(
        "__white_wall_posy",
        type="box",
        pos=f"0 {half_d} {half_h}",
        size=f"{half_w} {wall_th} {half_h}",
        rgba=white_rgba,
    )
    _add_geom(
        "__white_wall_negy",
        type="box",
        pos=f"0 {-half_d} {half_h}",
        size=f"{half_w} {wall_th} {half_h}",
        rgba=white_rgba,
    )
    _add_geom(
        "__white_wall_posx",
        type="box",
        pos=f"{half_w} 0 {half_h}",
        size=f"{wall_th} {half_d} {half_h}",
        rgba=white_rgba,
    )
    _add_geom(
        "__white_wall_negx",
        type="box",
        pos=f"{-half_w} 0 {half_h}",
        size=f"{wall_th} {half_d} {half_h}",
        rgba=white_rgba,
    )

    # Lots of white light: a strong directional overhead + several point fills
    # Keep castshadow=false to avoid darkening.
    _add_light(
        "__white_light_overhead_dir",
        pos=f"0 0 {2.0 * half_h - 0.2}",
        dir="0 0 -1",
        directional="true",
        castshadow="false",
        diffuse="1 1 1",
        ambient="0.9 0.9 0.9",
        specular="0.3 0.3 0.3",
    )

    # Corner fills
    _add_light(
        "__white_light_corner_1",
        pos=f"{half_w - 0.5} {half_d - 0.5} {2.0 * half_h - 0.5}",
        castshadow="false",
        diffuse="1 1 1",
        ambient="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
    )
    _add_light(
        "__white_light_corner_2",
        pos=f"{-(half_w - 0.5)} {half_d - 0.5} {2.0 * half_h - 0.5}",
        castshadow="false",
        diffuse="1 1 1",
        ambient="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
    )
    _add_light(
        "__white_light_corner_3",
        pos=f"{half_w - 0.5} {-(half_d - 0.5)} {2.0 * half_h - 0.5}",
        castshadow="false",
        diffuse="1 1 1",
        ambient="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
    )
    _add_light(
        "__white_light_corner_4",
        pos=f"{-(half_w - 0.5)} {-(half_d - 0.5)} {2.0 * half_h - 0.5}",
        castshadow="false",
        diffuse="1 1 1",
        ambient="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
    )

    # Front fill (helps make the robot “flat-lit”)
    _add_light(
        "__white_light_front_fill",
        pos="0 -6 2",
        castshadow="false",
        diffuse="1 1 1",
        ambient="0.85 0.85 0.85",
        specular="0.15 0.15 0.15",
    )


def _add_mjcf_light_if_missing(mjcf_text: str) -> str:
    """Ensure there is at least one light in <worldbody>."""
    root = ET.fromstring(mjcf_text)
    if root.tag != "mujoco":
        return mjcf_text  # not MJCF; don't touch
    _force_white_skybox(root)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # White room + filled white light
    _ensure_white_room(worldbody, group="1")

    has_light = any(child.tag == "light" for child in list(worldbody))
    if not has_light:
        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")
        light.set("dir", "0 0 -1")
        light.set("directional", "true")
        light.set("castshadow", "true")

    return ET.tostring(root, encoding="unicode")


def _inject_urdf_mujoco_extensions(urdf_text: str, meshdir: str) -> str:
    """
    Inject MuJoCo URDF extensions as <mujoco> child under <robot>.
    Does NOT edit the URDF file on disk; only modifies the in-memory XML string.
    """
    root = ET.fromstring(urdf_text)
    if root.tag != "robot":
        raise ValueError(f"Expected URDF root <robot>, got <{root.tag}>")

    # Find or create <mujoco> extension node FIRST
    mujoco_elem = root.find("mujoco")
    if mujoco_elem is None:
        mujoco_elem = ET.SubElement(root, "mujoco")

    # Force white background (skybox) under this <mujoco> node
    _force_white_skybox(mujoco_elem)

    compiler = mujoco_elem.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_elem, "compiler")

    # Key settings to make "bad inertias" loadable in MuJoCo without changing the URDF file:
    compiler.set("inertiafromgeom", "true")   # override URDF inertial tensors using geoms
    compiler.set("balanceinertia", "true")    # stabilize/repair borderline inertias
    compiler.set("discardvisual", "false")    # keep visuals for color
    compiler.set("meshdir", meshdir)          # resolve relative mesh paths robustly

    # Add lighting via extension worldbody (optional but harmless)
    worldbody = mujoco_elem.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(mujoco_elem, "worldbody")

    # White room + filled white light
    _ensure_white_room(worldbody, group="1")

    has_light = any(child.tag == "light" for child in list(worldbody))
    if not has_light:
        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")
        light.set("dir", "0 0 -1")
        light.set("directional", "true")
        light.set("castshadow", "false")  # keep background/scene from looking darker

    return ET.tostring(root, encoding="unicode")


def load_mjcf_xml(path: Path) -> mujoco.MjModel:
    """Load MJCF (.xml). Adds a world light if missing (in-memory only)."""
    txt = path.read_text(encoding="utf-8")
    txt2 = _add_mjcf_light_if_missing(txt)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(path.parent))  # help relative includes/mesh paths
        return mujoco.MjModel.from_xml_string(txt2)
    finally:
        os.chdir(old_cwd)


def load_urdf(path: Path) -> mujoco.MjModel:
    """
    Always load URDF via in-memory injected MuJoCo extensions so skybox/light changes
    actually take effect (otherwise from_xml_path returns before injection).
    """
    urdf_text = path.read_text(encoding="utf-8")
    injected = _inject_urdf_mujoco_extensions(urdf_text, meshdir=str(path.parent))

    # quick sanity check (optional but useful)
    print("[MuJoCo] injected has skybox:", 'type="skybox"' in injected)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(path.parent))  # help mesh paths resolve
        return mujoco.MjModel.from_xml_string(injected)
    finally:
        os.chdir(old_cwd)


def _force_white_skybox(root: ET.Element):
    """Make MuJoCo background pure white by overriding the skybox texture."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Reuse an existing skybox texture if present, otherwise create one.
    tex = None
    for t in asset.findall("texture"):
        if t.get("type") == "skybox":
            tex = t
            break
    if tex is None:
        tex = ET.SubElement(asset, "texture")
        tex.set("type", "skybox")

    tex.set("builtin", "flat")
    tex.set("rgb1", "1 1 1")
    tex.set("rgb2", "1 1 1")
    tex.set("width", "1")
    tex.set("height", "1")


def configure_viewer_visibility(viewer):
    """Try to turn on headlight and shadows for better visibility."""
    try:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_HEADLIGHT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SHADOW] = 0
    except Exception:
        pass

    try:
        viewer.opt.geomgroup[0] = 0
    except Exception:
        # Some viewer/binding versions may not expose geomgroup; ignore safely.
        pass


def run_viewer(model: mujoco.MjModel, dt: float, no_gravity: bool):
    if no_gravity:
        model.opt.gravity[:] = 0.0

    data = mujoco.MjData(model)
    set_all_joints_to_zero(model, data)

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_visibility(viewer)

        while viewer.is_running():
            # Keep action at all zeros
            set_all_joints_to_zero(model, data)
            if model.nu > 0:
                data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


def set_all_joints_to_zero(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Force all generalized coordinates (qpos) to zero and velocities to zero.
    This pins the robot at the zero configuration each frame.
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    # If you have mocap bodies, also zero them to avoid constraints fighting you
    if model.nmocap > 0:
        data.mocap_pos[:] = 0.0
        data.mocap_quat[:] = 0.0
        data.mocap_quat[:, 0] = 1.0  # identity quaternion
    mujoco.mj_forward(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        default="/home/aloha/EgoEngine_Workspace/egoengine/robosuite/models/assets/robots/rby1/robot_xhand_patched_torso.xml",
        help="Path to .xml (MJCF) or .urdf",
    )
    ap.add_argument("--dt", type=float, default=0.01, help="Viewer loop sleep (seconds)")
    ap.add_argument("--no-gravity", action="store_true", help="Disable gravity")
    args = ap.parse_args()

    p = Path(args.model_path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File does not exist: {p}")

    suffix = p.suffix.lower()
    print(f"[MuJoCo] Loading: {p}")

    if suffix == ".xml":
        model = load_mjcf_xml(p)
    elif suffix == ".urdf":
        model = load_urdf(p)
    else:
        raise SystemExit("Unsupported file type. Provide a .xml (MJCF) or .urdf.")

    run_viewer(model, dt=args.dt, no_gravity=args.no_gravity)


if __name__ == "__main__":
    main()
