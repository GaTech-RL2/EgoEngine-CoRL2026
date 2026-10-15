#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import os

# ─── set this to where you keep x_left_hand.xml & x_right_hand.xml ───
base_dir = "/mnt/data2/dexmimic/workspace/egoengine/robosuite/models/assets/grippers"

# our two colors
BLACK = "0.055 0.055 0.063 1"
WHITE = "0.941 0.941 0.941 1"

def fix_file(path):
    tree = ET.parse(path)
    root = tree.getroot()

    wb = root.find("worldbody")
    if wb is None:
        print(f"⚠️ no <worldbody> in {os.path.basename(path)}")
        return

    n = 0
    # find every geom under worldbody
    for geom in wb.findall(".//geom"):
        mesh = geom.get("mesh")
        if mesh is None:
            continue

        # pick white for any "*back*" mesh, else black
        color = WHITE if "back" in mesh else BLACK

        # force it into group 1 (which Robosuite renders)
        geom.set("group", "1")

        # overwrite or add rgba
        geom.set("rgba", color)
        n += 1

    tree.write(path, xml_declaration=True, encoding="utf-8")
    print(f"✔️  {os.path.basename(path)} → updated {n} geoms")

if __name__ == "__main__":
    for fn in ("x_left_hand.xml", "x_right_hand.xml"):
        full = os.path.join(base_dir, fn)
        if not os.path.exists(full):
            print(f"❌  file not found: {full}")
        else:
            fix_file(full)
