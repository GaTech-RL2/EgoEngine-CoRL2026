# sync_xhand_left.py
"""
Step‑wise updater that synchronises MuJoCo x_left_hand.xml with a reference URDF:

    1. Copy inertial blocks (mass, principal moments, CoM) from URDF <link>       
       into matching <body> in the MuJoCo XML. Off‑diagonal terms are ignored.

    2. Align actuator force limits with URDF <joint limit effort>. Also refresh
       ctrlrange if the numeric limit in the URDF differs from the XML.

    3. Set joint damping uniformly to a very small value (0.001) so the hand is
       effectively passive like the hardware.

    4. Enable self‑collision: every mesh geom gets contype = conaffinity = 1 and
       any <contact><exclude …> pairs are deleted.

Run once, the script rewrites the MJCF in place while saving a .bak.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# ---------------------------------------------------------------------------
# Configure paths -----------------------------------------------------------
# ---------------------------------------------------------------------------
URDF_PATH = Path(
    "/home/rl2-aloha/Downloads/xhand_left/urdf/xhand_left.urdf"
)
MJCF_PATH = Path(
    "/mnt/data2/dexmimic/workspace/egoengine/robosuite/models/assets/grippers/x_left_hand.xml"
)

BAK_SUFFIX = ".bak"
DAMPING_VALUE = "0.001"  # almost zero damping

# ---------------------------------------------------------------------------
# Helper functions ----------------------------------------------------------
# ---------------------------------------------------------------------------

def et_parse(path: Path) -> ET.ElementTree:
    if not path.exists():
        raise FileNotFoundError(path)
    return ET.parse(path)


def copy_inertial(link_urdf: ET.Element, body_mj: ET.Element) -> None:
    """Copy <inertial> from URDF link into MuJoCo body (replace / insert)."""
    inertial_urdf = link_urdf.find("inertial")
    if inertial_urdf is None:
        return  # nothing to copy

    origin = inertial_urdf.find("origin")
    mass_tag = inertial_urdf.find("mass")
    inertia_tag = inertial_urdf.find("inertia")
    if mass_tag is None or inertia_tag is None:
        return

    xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    mass_val = mass_tag.get("value")
    # principal moments (off-diagonals ignored)
    ixx = inertia_tag.get("ixx", "1e-6")
    iyy = inertia_tag.get("iyy", "1e-6")
    izz = inertia_tag.get("izz", "1e-6")

    # Replace existing or create new
    inertial_mj = body_mj.find("inertial")
    if inertial_mj is None:
        inertial_mj = ET.SubElement(body_mj, "inertial")
    inertial_mj.set("pos", xyz)
    inertial_mj.set("mass", mass_val)
    inertial_mj.set("diaginertia", f"{ixx} {iyy} {izz}")


def map_links_to_bodies(mjroot: ET.Element) -> dict[str, ET.Element]:
    """Return dict of body name → body element (only leafs under worldbody)."""
    bodies = {}

    def recurse(body: ET.Element):
        name = body.get("name")
        if name:
            bodies[name] = body
        for child in body.findall("body"):
            recurse(child)

    world = mjroot.find("worldbody")
    if world is None:
        raise RuntimeError("No <worldbody> in MJCF")
    for top in world.findall("body"):
        recurse(top)
    return bodies


def sync_inertials(tree_urdf: ET.ElementTree, tree_mj: ET.ElementTree):
    link2body = map_links_to_bodies(tree_mj.getroot())
    for link in tree_urdf.findall("link"):
        name = link.get("name")
        body = link2body.get(name)
        if body is not None:
            copy_inertial(link, body)


def sync_effort_limits(tree_urdf: ET.ElementTree, tree_mj: ET.ElementTree):
    # Build map joint_name -> effort, upper, lower
    effort_map = {}
    for joint in tree_urdf.findall("joint"):
        lim = joint.find("limit")
        if lim is None:
            continue
        eff = lim.get("effort")
        lower = lim.get("lower")
        upper = lim.get("upper")
        if eff:
            effort_map[joint.get("name")] = (float(eff), lower, upper)

    # Iterate actuators that reference these joints
    for act in tree_mj.findall(".//actuator/position"):
        joint_name = act.get("joint")
        if joint_name in effort_map:
            effort, lower, upper = effort_map[joint_name]
            act.set("forcerange", f"-{effort} {effort}")
            if lower and upper:
                act.set("ctrlrange", f"{lower} {upper}")


def set_joint_damping(tree_mj: ET.ElementTree, damping=DAMPING_VALUE):
    for joint in tree_mj.findall(".//joint"):
        joint.set("damping", damping)


def enable_self_collision(tree_mj: ET.ElementTree):
    # 1) set contype / conaffinity = 1 for all collision geoms
    for geom in tree_mj.findall(".//geom"):
        geom.set("contype", "1")
        geom.set("conaffinity", "1")

    # 2) remove any <contact><exclude> pairs
    contact = tree_mj.find("contact")
    if contact is not None:
        for ex in list(contact.findall("exclude")):
            contact.remove(ex)


# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    urdf = et_parse(URDF_PATH)
    mjcf = et_parse(MJCF_PATH)

    # Backup
    shutil.copy2(MJCF_PATH, MJCF_PATH.with_suffix(BAK_SUFFIX))

    # 1 Inertia
    sync_inertials(urdf, mjcf)
    print("[1] inertial blocks copied")

    # 2 Effort & range
    sync_effort_limits(urdf, mjcf)
    print("[2] actuator forcerange aligned")

    # 3 Damping unify
    set_joint_damping(mjcf)
    print("[3] joint damping set to", DAMPING_VALUE)

    # 4 Self collision
    enable_self_collision(mjcf)
    print("[4] self‑collision enabled (contype=conaffinity=1)")

    mjcf.write(MJCF_PATH, encoding="utf-8", xml_declaration=True)
    print("✅  sync_xhand_left.py finished; original file backed up with .bak suffix.")


if __name__ == "__main__":
    main()
