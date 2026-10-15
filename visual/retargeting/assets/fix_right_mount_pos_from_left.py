#!/usr/bin/env python3
import argparse
import copy
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, List

LEFT_MOUNT_BODY = "left"         # in your xml, left mount body is named "left"
RIGHT_MOUNT_BODY = "right_mount" # right mount body name
RIGHT_EEF_BODY = "right_eef"     # xhand root on right

LEFT_PREFIX = "LEFT_MOUNT/"
RIGHT_PREFIX = "RIGHT_MOUNT/"

# Hard mapping for the known mount mesh basenames
MESH_BASENAME_MAP = {
    "ee_mount_l": "ee_mount_r",
    "ee_mount_l_collision_0": "ee_mount_r_collision_0",
}

def find_body(root: ET.Element, name: str) -> Optional[ET.Element]:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None

def find_asset_mesh_names(root: ET.Element) -> set:
    names = set()
    asset = root.find("asset")
    if asset is None:
        return names
    for m in asset.findall("mesh"):
        nm = m.get("name")
        if nm:
            names.add(nm)
    return names

def parse_vec3(s: str) -> List[float]:
    parts = [p for p in s.strip().split() if p]
    if len(parts) != 3:
        raise ValueError(f"Expected vec3, got {s!r}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]

def fmt_vec3(v: List[float]) -> str:
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}".rstrip("0").rstrip(".")

def get_pos(b: ET.Element) -> List[float]:
    return parse_vec3(b.get("pos", "0 0 0"))

def set_pos(b: ET.Element, v: List[float]) -> None:
    b.set("pos", fmt_vec3(v))

def direct_geoms(b: ET.Element) -> List[ET.Element]:
    return [c for c in list(b) if c.tag == "geom"]

def clear_direct_geoms(b: ET.Element) -> None:
    for g in direct_geoms(b):
        b.remove(g)

def remap_left_mount_mesh_to_right(mesh_name: str) -> str:
    """
    Converts LEFT_MOUNT/ee_mount_l* -> RIGHT_MOUNT/ee_mount_r*
    If not prefixed, returns mesh_name unchanged.
    """
    if not mesh_name.startswith(LEFT_PREFIX):
        return mesh_name

    base = mesh_name[len(LEFT_PREFIX):]

    # exact mapping first (most reliable)
    if base in MESH_BASENAME_MAP:
        base2 = MESH_BASENAME_MAP[base]
        return RIGHT_PREFIX + base2

    # fallback: replace trailing _l with _r in tokens like "..._l" or "..._l_collision_0"
    # but only if it looks like a left/right mount name
    base2 = re.sub(r"(^|_)l($|_)", r"\1r\2", base)

    return RIGHT_PREFIX + base2

def copy_left_mount_structure_to_right(left_mount: ET.Element, right_mount: ET.Element) -> Tuple[List[str], List[str]]:
    """
    Copy right_mount attributes (quat + optional others) and direct geoms from left_mount,
    but remap mesh names to RIGHT mount meshes.
    Returns (new_meshes, missing_meshes_in_asset)
    """
    # copy quat to match left structure
    if "quat" in left_mount.attrib:
        right_mount.set("quat", left_mount.get("quat"))
    else:
        right_mount.attrib.pop("quat", None)

    # DO NOT blindly copy 'name' (keep right_mount name)
    # Optionally also mirror 'pos' if you want exact same as left:
    # right_mount.set("pos", left_mount.get("pos", "0 0 0"))

    # replace direct geoms
    lgeoms = direct_geoms(left_mount)
    if len(lgeoms) == 0:
        raise RuntimeError("Left mount has no direct <geom> children to copy.")

    clear_direct_geoms(right_mount)

    new_meshes = []
    for lg in lgeoms:
        rg = copy.deepcopy(lg)
        mesh = rg.get("mesh")
        if mesh:
            rg.set("mesh", remap_left_mount_mesh_to_right(mesh))
            new_meshes.append(rg.get("mesh"))
        right_mount.append(rg)

    # Validate meshes exist in asset
    asset_meshes = find_asset_mesh_names(left_mount)  # (root is not passed here; will validate outside)
    return new_meshes, []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to robot_xhand_new.xml")
    ap.add_argument("--out", default=None, help="Output path. Default: overwrite input (with .bak backup)")
    ap.add_argument("--dz_mount", type=float, default=0.0, help="Add to right_mount pos.z (meters)")
    ap.add_argument("--dz_eef", type=float, default=0.0, help="Add to right_eef pos.z (meters)")
    ap.add_argument("--copy_left_pos", action="store_true", help="Also set right_mount.pos = left_mount.pos before dz_mount")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    out_path = Path(args.out) if args.out else xml_path

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    left_mount = find_body(root, LEFT_MOUNT_BODY)
    right_mount = find_body(root, RIGHT_MOUNT_BODY)
    right_eef = find_body(root, RIGHT_EEF_BODY)

    if left_mount is None:
        print(f"[ERROR] Could not find left mount body name={LEFT_MOUNT_BODY!r}", file=sys.stderr)
        sys.exit(2)
    if right_mount is None:
        print(f"[ERROR] Could not find right mount body name={RIGHT_MOUNT_BODY!r}", file=sys.stderr)
        sys.exit(2)
    if right_eef is None:
        print(f"[ERROR] Could not find right eef body name={RIGHT_EEF_BODY!r}", file=sys.stderr)
        sys.exit(2)

    # (1) Make right_mount structure match left mount (but with RIGHT meshes)
    # We will rebuild the direct geoms to mirror left, remapping meshes properly.
    lgeoms = direct_geoms(left_mount)
    if len(lgeoms) != 2:
        # You said mount has the same structure; typically exactly 2 geoms (visual + collision).
        # If it's not 2, we still copy whatever exists, but warn.
        print(f"[WARN] Left mount has {len(lgeoms)} direct geoms (expected 2). Copying all.", file=sys.stderr)

    # Copy quat from left
    if "quat" in left_mount.attrib:
        right_mount.set("quat", left_mount.get("quat"))
    else:
        right_mount.attrib.pop("quat", None)

    # Optionally copy left pos
    if args.copy_left_pos:
        right_mount.set("pos", left_mount.get("pos", "0 0 0"))

    # Replace geoms with remapped meshes
    clear_direct_geoms(right_mount)

    new_meshes = []
    for lg in lgeoms:
        rg = copy.deepcopy(lg)
        mesh = rg.get("mesh")
        if not mesh:
            raise RuntimeError("A left mount geom is missing a mesh attribute; cannot mirror safely.")
        new_mesh = remap_left_mount_mesh_to_right(mesh)
        rg.set("mesh", new_mesh)
        new_meshes.append(new_mesh)
        right_mount.append(rg)

    # (2) Z tweaks (pos-only, per your requirement)
    rm_pos = get_pos(right_mount)
    rm_pos[2] += args.dz_mount
    set_pos(right_mount, rm_pos)

    re_pos = get_pos(right_eef)
    re_pos[2] += args.dz_eef
    set_pos(right_eef, re_pos)

    # Validate new meshes exist
    asset_meshes = find_asset_mesh_names(root)
    missing = [m for m in new_meshes if m not in asset_meshes]
    if missing:
        print("[ERROR] After rewrite, these meshes are still missing in <asset>:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("This means your <asset><mesh name=...> entries do not include them.", file=sys.stderr)
        sys.exit(3)

    # Write with backup if overwriting
    if out_path == xml_path:
        bak = xml_path.with_suffix(xml_path.suffix + ".bak")
        xml_path.replace(bak)
        print(f"[OK] Backup written: {bak}")

    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] right_mount pos.z now: {get_pos(right_mount)[2]:.6f}")
    print(f"[OK] right_eef pos.z now:  {get_pos(right_eef)[2]:.6f}")
    print("[OK] right_mount meshes now:")
    for m in new_meshes:
        print(f"  - {m}")

if __name__ == "__main__":
    main()
