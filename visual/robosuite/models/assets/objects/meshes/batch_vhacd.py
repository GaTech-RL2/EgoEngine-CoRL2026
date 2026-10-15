#!/usr/bin/env python3
"""
patch_convex_meshes.py
──────────────────────
Replace collision meshes in every *_cm.xml so that:

    - visual geom  ->  meshes/XXX_cm.obj   (unchanged)
    - collision geom -> meshes/XXX_convex.obj

Run inside the directory that contains all *_cm.xml files.

Usage:
    python patch_convex_meshes.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys, re

# Helper: pretty-print back with indentation
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

# Process every *_cm.xml
for xml_path in sorted(Path(".").glob("*_cm.xml")):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ----- 1. locate the original mesh -----
    asset = root.find("asset")
    if asset is None:
        print(f"[skip] {xml_path}: no <asset>", file=sys.stderr); continue

    mesh_cm = None
    for m in asset.findall("mesh"):
        if re.search(r"_cm\.obj$", m.get("file", "")):
            mesh_cm = m
            break
    if mesh_cm is None:
        print(f"[skip] {xml_path}: no *_cm.obj mesh", file=sys.stderr); continue

    base = Path(mesh_cm.get("file")).stem.replace("_cm", "")   # e.g. "001"
    convex_file = f"meshes/{base}_convex.obj"
    convex_mesh_name = f"{base}_convex_mesh"

    # ----- 2. append the convex mesh if not already there -----
    already = any(m.get("name") == convex_mesh_name for m in asset.findall("mesh"))
    if not already:
        new_mesh = ET.Element("mesh")
        new_mesh.attrib.update(
            file=convex_file,
            name=convex_mesh_name,
            scale=mesh_cm.get("scale", "1 1 1")
        )
        asset.append(new_mesh)

    # ----- 3. switch one collision geom to the convex mesh -----
    switched = False
    for geom in root.findall(".//geom"):
        # choose the first geom that looks like “collision” (contype/affinity non-zero)
        if geom.get("contype", "0") != "0" or geom.get("conaffinity", "0") != "0":
            geom.set("mesh", convex_mesh_name)
            switched = True
            break

    if not switched:
        print(f"[warn] {xml_path}: no geom with contype/conaffinity=1; nothing changed",
              file=sys.stderr)

    # ----- 4. save back -----
    indent(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[ok]  {xml_path.name}  -> uses {convex_file}")

print("✓ All files processed.")
