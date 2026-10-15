#!/usr/bin/env python3
"""
urdf2mjcf_xhand.py  –  Works with your xhand URDFs out‑of‑the‑box.

Usage:
    python urdf2mjcf_xhand.py /path/to/xhand_right.urdf
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

TARGET_SUBDIR = "meshes/x_hands"
KP_DEFAULT    = 2.0

# ──────────────────────────────────────────────────────────────────
# NEW ── 0. Sanitise inertias: if any diagonal < 1e‑8, set the whole
#         tensor and mass to zero so MuJoCo assigns safe defaults.
# ──────────────────────────────────────────────────────────────────
def zero_tiny_inertias(urdf_path: Path):
    tree = ET.parse(urdf_path)
    changed = False
    for inertial in tree.findall(".//inertial"):
        inertia = inertial.find("inertia")
        if inertia is None:
            continue
        diags = [float(inertia.get(k, "0")) for k in ("ixx", "iyy", "izz")]
        if min(diags) < 1e-8:
            # Zero‑out all inertia terms and mass
            for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                inertia.set(k, "0")
            mass_tag = inertial.find("mass")
            if mass_tag is not None:
                mass_tag.set("value", "0")
            changed = True
    if changed:
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

# ----------------------------------------------------------------------
# 1) Convert every <mesh filename="…"> to an absolute on‑disk path
# ----------------------------------------------------------------------
def make_mesh_paths_absolute(urdf_path: Path) -> Path:
    pkg_root  = urdf_path.parent.parent          # …/xhand_right
    mesh_root = pkg_root / "meshes"              # …/xhand_right/meshes
    abs_urdf  = urdf_path.with_name(urdf_path.stem + "_abs.urdf")

    tree = ET.parse(urdf_path)
    for mesh in tree.findall(".//mesh[@filename]"):
        fname = mesh.get("filename")

        if fname.startswith("package://"):
            # package://xhand_right/meshes/foo.STL -> meshes/foo.STL
            rel = "/".join(fname.split("/")[2:])
            mesh.set("filename", str(pkg_root / rel))

        elif fname.startswith("meshes/"):
            mesh.set("filename", str(pkg_root / fname))

        elif not Path(fname).is_absolute():
            mesh.set("filename", str(mesh_root / fname))
        # (absolute paths are left untouched)

    tree.write(abs_urdf, encoding="utf-8", xml_declaration=True)
    return abs_urdf


# ----------------------------------------------------------------------
# 2) Helpers for header / actuators / mesh‑path rewrite in the output MJCF
# ----------------------------------------------------------------------
import xml.etree.ElementTree as ET

def ensure_header(root: ET.Element):
    if root.find("statistic") is None:
        ET.SubElement(root, "statistic", center="0.3 0 0.4", extent="1")

    comp = root.find("compiler") or ET.SubElement(root, "compiler")
    comp.set("angle", "radian"); comp.set("meshdir", "./"); comp.set("autolimits", "true")

    if root.find("visual") is None:
        vis = ET.SubElement(root, "visual")
        ET.SubElement(vis, "headlight",
                      diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
        ET.SubElement(vis, "rgba",  haze="0.15 0.25 0.35 1")
        ET.SubElement(vis, "global", azimuth="120", elevation="20")

    d = root.find("default") or ET.SubElement(root, "default")
    if d.find("joint") is None:
        ET.SubElement(d, "joint", damping="1")
    if d.find("muscle") is None:
        ET.SubElement(d, "muscle", ctrllimited="true", ctrlrange="0 100")
    if d.find("motor") is None:
        ET.SubElement(d, "motor",  ctrllimited="true", ctrlrange="-3 3", gear="1")
    if d.find("position") is None:
        ET.SubElement(d, "position", ctrllimited="true", kp="10")


def add_actuators(root: ET.Element):
    parent = root.find("actuator") or ET.SubElement(root, "actuator")
    for joint in root.findall(".//joint[@type]"):
        if joint.get("type") not in ("hinge", "slide"):
            continue
        lo, hi = ("-3.1416", "3.1416") if joint.get("type") == "hinge" else ("-1.0", "1.0")
        if joint.get("range"):
            lo, hi = joint.get("range").split()
        ET.SubElement(
            parent, "position",
            name=f"{joint.get('name')}_act",
            joint=joint.get("name"),
            kp=str(KP_DEFAULT),
            gear="1",
            ctrlrange=f"{lo} {hi}",
        )


def rewrite_mesh_refs(root: ET.Element):
    for mesh in root.findall(".//mesh[@file]"):
        mesh.set("file", f"{TARGET_SUBDIR}/{Path(mesh.get('file')).name}")


def convert(urdf_file: Path):
    if not urdf_file.is_file():
        print(f"[skip] {urdf_file} not found"); return

    # 1. Patch mesh paths
    patched = make_mesh_paths_absolute(urdf_file)

    # ── NEW: zero tiny inertias before compiling
    zero_tiny_inertias(patched)
    # ────────────────────────────────────────────

    # 2. Compile with MuJoCo
    mjcf_path = urdf_file.with_suffix(".xml")
    mujoco.mj_saveLastXML(
        str(mjcf_path),
        mujoco.MjModel.from_xml_path(str(patched))
    )

    # 3. Post‑process MJCF
    tree = ET.parse(mjcf_path); root = tree.getroot()
    ensure_header(root); add_actuators(root); rewrite_mesh_refs(root)
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)

    patched.unlink()
    print(f"✓  {urdf_file.name} → {mjcf_path.name}")



# ----------------------------------------------------------------------
# 4) CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python urdf2mjcf_xhand.py file1.urdf [file2.urdf …]")
        sys.exit(1)
    for urdf in sys.argv[1:]:
        convert(Path(urdf))
