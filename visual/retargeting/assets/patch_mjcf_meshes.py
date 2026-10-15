#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -------------------------
# Sorting helpers
# -------------------------
_num_re = re.compile(r"(\d+)")


def natural_key(s: str):
    # split into str/int for natural sort
    parts = _num_re.split(s)
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def sort_visual(files: List[str]) -> List[str]:
    return sorted(files, key=natural_key)


def sort_collision(files: List[str]) -> List[str]:
    # prefer sorting by collision index if exists
    def key(fn: str):
        m = re.search(r"_collision_(\d+)\.obj$", fn, flags=re.IGNORECASE)
        if m:
            return (0, int(m.group(1)), natural_key(fn))
        return (1, 10**9, natural_key(fn))
    return sorted(files, key=key)


# -------------------------
# Filesystem scan
# -------------------------
def list_obj_files(dir_path: Path) -> Tuple[List[str], List[str]]:
    """
    Return (visual_objs, collision_objs) as filenames (not full paths).
      - visual: *.obj excluding *_collision_*.obj and files starting with '_'
      - collision: *_collision_*.obj
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    objs = [p.name for p in dir_path.glob("*.obj") if p.is_file()]
    objs = [n for n in objs if not n.startswith("_")]  # ignore temp vhacd outputs

    collision = [n for n in objs if "_collision_" in n]
    visual = [n for n in objs if "_collision_" not in n]

    return sort_visual(visual), sort_collision(collision)


# -------------------------
# MJCF text surgery helpers
# -------------------------
def find_body_block(text: str, body_name: str) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) spanning full <body ...name="X"...>...</body>.
    Robust even if the <body ...> tag has newlines.
    """
    needle = f'name="{body_name}"'
    i = 0
    while True:
        s = text.find("<body", i)
        if s == -1:
            break
        gt = text.find(">", s)
        if gt == -1:
            break
        tag = text[s : gt + 1]  # may include newlines

        if needle in tag:
            start = s
            j = gt + 1
            depth = 1
            while depth > 0:
                next_open = text.find("<body", j)
                next_close = text.find("</body>", j)
                if next_close == -1:
                    raise ValueError(f'Unclosed body "{body_name}"')

                if next_open != -1 and next_open < next_close:
                    gt2 = text.find(">", next_open)
                    if gt2 == -1:
                        raise ValueError("Malformed <body ...> tag (missing '>').")
                    depth += 1
                    j = gt2 + 1
                else:
                    depth -= 1
                    j = next_close + len("</body>")
            return start, j

        i = gt + 1

    raise ValueError(f'Cannot find <body name="{body_name}">')


def extract_attr_values(lines: List[str], attr: str) -> List[str]:
    out: List[str] = []
    for ln in lines:
        m = re.search(rf'\b{re.escape(attr)}="([^"]+)"', ln)
        if m:
            out.append(m.group(1))
    return out


def cycle(values: List[str], n: int) -> List[str]:
    if not values:
        return [""] * n
    return [values[i % len(values)] for i in range(n)]


# -------------------------
# <asset> <mesh ...> patch (robust, supports multi-line <mesh .../>)
# -------------------------
def replace_asset_mesh_lines_for_subdir(text: str, subdir: str, obj_files: List[str]) -> str:
    """
    Replace all <mesh ... file="...assets/<subdir>/..."> tags with new ones from obj_files.

    Robust to tags that span multiple lines:
      <mesh
        name="..."
        file="./assets/SUBDIR/xxx.obj"
      />

    Keeps insertion location:
      - if existing tags found: replace at first removed tag
      - else: insert right after <asset ...> start tag (or beginning)
    """
    lines = text.splitlines(True)

    target_re = re.compile(rf'assets/{re.escape(subdir)}/', re.IGNORECASE)

    new_lines: List[str] = []
    removed_any = False
    insert_at: Optional[int] = None
    indent = "        "

    i = 0
    while i < len(lines):
        ln = lines[i]
        if "<mesh" in ln:
            tag_lines = [ln]
            j = i + 1

            # accumulate until we see '/>' (self-closing) OR '</mesh>'
            if "/>" not in ln and "</mesh>" not in ln:
                while j < len(lines) and "/>" not in lines[j] and "</mesh>" not in lines[j]:
                    tag_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    tag_lines.append(lines[j])
                    j += 1

            tag_text = "".join(tag_lines)

            if target_re.search(tag_text):
                # remove it
                if insert_at is None:
                    insert_at = len(new_lines)
                    m = re.match(r"^(\s*)<mesh\b", tag_lines[0])
                    if m:
                        indent = m.group(1)
                removed_any = True
                i = j
                continue

            # keep it
            new_lines.extend(tag_lines)
            i = j
            continue

        new_lines.append(ln)
        i += 1

    # if none removed, choose insertion after <asset ...>
    if insert_at is None:
        insert_at = None
        for k, ln in enumerate(new_lines):
            if re.search(r"^\s*<asset\b", ln):
                insert_at = k + 1
                break
        if insert_at is None:
            insert_at = 0

    # ensure deterministic order: visuals first then collisions if mixed
    obj_files_sorted = sort_visual([f for f in obj_files if "_collision_" not in f]) + sort_collision(
        [f for f in obj_files if "_collision_" in f]
    )

    def mk(fn: str) -> str:
        stem = Path(fn).stem
        return f'{indent}<mesh name="{stem}" file="./assets/{subdir}/{fn}" />\n'

    injected = [mk(fn) for fn in obj_files_sorted]
    new_lines[insert_at:insert_at] = injected

    return "".join(new_lines)


# -------------------------
# Body patch: only top-level mesh geoms
# -------------------------
def patch_body_mesh_geoms_top_level(
    text: str,
    body_name: str,
    mesh_prefixes: List[str],
    new_visual_mesh_names: List[str],
    new_collision_mesh_names: List[str],
    material_override: Optional[Dict[str, str]] = None,
) -> str:
    """
    In the given body, ONLY modify top-level <geom ... mesh="..."> lines that match mesh_prefixes.
    - Removes old mesh-based geoms referencing those prefixes.
    - Inserts new mesh-based visual + collision geoms.
    - Does NOT touch primitive collisions (capsule/box/etc without mesh="...").
    """
    material_override = material_override or {}

    s, e = find_body_block(text, body_name)
    blk = text[s:e]
    blk_lines = blk.splitlines(True)

    # Find first nested child <body ...> within this block to avoid touching descendants
    first_child = len(blk_lines)
    for i, ln in enumerate(blk_lines):
        if i == 0:
            continue
        if "<body" in ln:
            first_child = i
            break

    def mesh_matches_line(ln: str) -> bool:
        if "<geom" not in ln or 'mesh="' not in ln:
            return False
        m = re.search(r'\bmesh="([^"]+)"', ln)
        if not m:
            return False
        mesh_name = m.group(1)
        return any(mesh_name.startswith(p) for p in mesh_prefixes)

    rm_idx: List[int] = []
    old_geom: List[str] = []
    for i in range(0, first_child):
        ln = blk_lines[i]
        if mesh_matches_line(ln):
            rm_idx.append(i)
            old_geom.append(ln)

    old_vis = [ln for ln in old_geom if 'class="visual"' in ln]
    old_col = [ln for ln in old_geom if 'class="collision"' in ln]
    old_mats = extract_attr_values(old_vis, "material")
    old_rgbas = extract_attr_values(old_col, "rgba")

    indent = " " * 12
    if old_geom:
        m = re.match(r"^(\s*)<geom", old_geom[0])
        if m:
            indent = m.group(1)

    # remove old mesh geoms
    for i in reversed(rm_idx):
        blk_lines.pop(i)

    # insert after last top-level <joint>, else after <inertial>, else after open tag
    insert_at = 1
    for i in range(0, min(first_child, len(blk_lines))):
        if re.search(r"^\s*<joint\b", blk_lines[i]):
            insert_at = i + 1
    if insert_at == 1:
        for i in range(0, min(first_child, len(blk_lines))):
            if re.search(r"^\s*<inertial\b", blk_lines[i]):
                insert_at = i + 1
                break

    mats = cycle(old_mats, len(new_visual_mesh_names))
    rgbas = cycle(old_rgbas, len(new_collision_mesh_names))

    new_lines: List[str] = []
    for i, mesh_name in enumerate(new_visual_mesh_names):
        mat = material_override.get(mesh_name, mats[i] if i < len(mats) else "")
        if mat:
            new_lines.append(f'{indent}<geom class="visual" material="{mat}" mesh="{mesh_name}" />\n')
        else:
            new_lines.append(f'{indent}<geom class="visual" mesh="{mesh_name}" />\n')

    for mesh_name, rgba in zip(new_collision_mesh_names, rgbas):
        if rgba:
            new_lines.append(f'{indent}<geom class="collision" rgba="{rgba}" mesh="{mesh_name}" />\n')
        else:
            new_lines.append(f'{indent}<geom class="collision" mesh="{mesh_name}" />\n')

    blk_lines[insert_at:insert_at] = new_lines
    blk_new = "".join(blk_lines)
    return text[:s] + blk_new + text[e:]


def infer_link11_white_black(vis_names: List[str]) -> Dict[str, str]:
    """
    Infer LINK_11 material groups from *_1 / *_2 or WY_1 / WY_2.
    Group 1 uses white material "254,254,255"; group 2 uses black "25,25,25".
    """
    m: Dict[str, str] = {}

    def is_one(n: str) -> bool:
        u = n.upper()
        return ("WY_1" in u) or bool(re.search(r"(?:_)?1$", n))

    def is_two(n: str) -> bool:
        u = n.upper()
        return ("WY_2" in u) or bool(re.search(r"(?:_)?2$", n))

    ones = [n for n in vis_names if is_one(n)]
    twos = [n for n in vis_names if is_two(n)]

    if ones and twos:
        for n in ones:
            m[n] = "254,254,255"
        for n in twos:
            m[n] = "25,25,25"
        return m

    if len(vis_names) == 2:
        m[vis_names[0]] = "254,254,255"
        m[vis_names[1]] = "25,25,25"
    return m


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", type=str, required=True, help="robot_xhand_new.xml (MJCF)")
    ap.add_argument("--assets_root", type=str, required=True, help=".../rby1_new/assets")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    mjcf_path = Path(args.mjcf).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve()

    txt = mjcf_path.read_text(encoding="utf-8")

    # Only arm folders
    link11_dir = assets_root / "LINK_11"
    link12_dir = assets_root / "LINK_12"
    link13_dir = assets_root / "LINK_13"

    l11_vis_f, l11_col_f = list_obj_files(link11_dir)
    l12_vis_f, l12_col_f = list_obj_files(link12_dir)
    l13_vis_f, l13_col_f = list_obj_files(link13_dir)

    if not l11_vis_f:
        raise RuntimeError(f"No visual OBJ in {link11_dir}")
    if not l12_vis_f:
        raise RuntimeError(f"No visual OBJ in {link12_dir}")
    if not l13_vis_f:
        raise RuntimeError(f"No visual OBJ in {link13_dir}")

    # Patch <asset> mesh declarations for LINK_11/12/13 only
    txt2 = txt
    txt2 = replace_asset_mesh_lines_for_subdir(txt2, "LINK_11", l11_vis_f + l11_col_f)
    txt2 = replace_asset_mesh_lines_for_subdir(txt2, "LINK_12", l12_vis_f + l12_col_f)
    txt2 = replace_asset_mesh_lines_for_subdir(txt2, "LINK_13", l13_vis_f + l13_col_f)

    # Patch bodies: link_right_arm_4/5/6 only
    l11_vis_names = [Path(f).stem for f in l11_vis_f]
    l11_col_names = [Path(f).stem for f in l11_col_f]
    l12_vis_names = [Path(f).stem for f in l12_vis_f]
    l12_col_names = [Path(f).stem for f in l12_col_f]
    l13_vis_names = [Path(f).stem for f in l13_vis_f]
    l13_col_names = [Path(f).stem for f in l13_col_f]

    link11_mat_map = infer_link11_white_black(l11_vis_names)

    txt3 = txt2
    txt3 = patch_body_mesh_geoms_top_level(
        txt3,
        body_name="link_right_arm_4",
        mesh_prefixes=["LINK_11"],
        new_visual_mesh_names=l11_vis_names,
        new_collision_mesh_names=l11_col_names,
        material_override=link11_mat_map,
    )
    txt3 = patch_body_mesh_geoms_top_level(
        txt3,
        body_name="link_right_arm_5",
        mesh_prefixes=["LINK_12"],
        new_visual_mesh_names=l12_vis_names,
        new_collision_mesh_names=l12_col_names,
        material_override=None,
    )
    txt3 = patch_body_mesh_geoms_top_level(
        txt3,
        body_name="link_right_arm_6",
        mesh_prefixes=["LINK_13"],
        new_visual_mesh_names=l13_vis_names,
        new_collision_mesh_names=l13_col_names,
        material_override=None,
    )

    if args.backup:
        bak = mjcf_path.with_suffix(mjcf_path.suffix + ".bak")
        bak.write_text(txt, encoding="utf-8")
        print(f"[backup] {bak}")

    mjcf_path.write_text(txt3, encoding="utf-8")

    print(f"[OK] Patched MJCF (ARM ONLY): {mjcf_path}")
    print(f"  LINK_11 visuals: {len(l11_vis_names)}, collisions: {len(l11_col_names)}")
    print(f"  LINK_12 visuals: {len(l12_vis_names)}, collisions: {len(l12_col_names)}")
    print(f"  LINK_13 visuals: {len(l13_vis_names)}, collisions: {len(l13_col_names)}")
    if link11_mat_map:
        print(f"  LINK_11 material override (white/black): {link11_mat_map}")


if __name__ == "__main__":
    main()
