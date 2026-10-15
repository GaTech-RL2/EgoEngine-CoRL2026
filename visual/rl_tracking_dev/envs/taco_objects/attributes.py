import os
import xml.etree.ElementTree as ET
from glob import glob

XML_DIR = "./"

# new_friction = "20.0 5.0 5.0"
# new_solimp = "0.95 0.95 0.001"
# new_solref = "0.01 2"


new_friction = "0.95 0.3 0.1"
new_solimp="0.998 0.998 0.001"
new_solref = "0.001 1"
new_margin = "0.001"
new_density = "50"


new_friction = "1.2 0.5 0.1"
new_margin = "0.001"
new_density = "25"


condim = "4"


contype = "4"
conaffinity = "7"

xml_files = sorted(glob(os.path.join(XML_DIR, "*_cm.xml")))

BALLAST_ATTRS = {
    "type": "box",
    "name": "obj_ballast",
    "size": "0.015 0.015 0.010",
    "pos":  "0 0 -0.010",
    "density": "0",
    "rgba": "0 0 0 0",
    "contype": "0",
    "conaffinity": "0",
    "group": "0"
}

def add_ballast_to_body(root_or_tree, body_name="object", attrs=None):
    """
    Add (or update) a non-colliding ballast geom under a named body.
    root_or_tree: ET.Element or ET.ElementTree
    body_name:    target <body name="...">
    attrs:        dict to override BALLAST_ATTRS (optional)
    """
    root = root_or_tree.getroot() if isinstance(root_or_tree, ET.ElementTree) else root_or_tree
    attrs = {**BALLAST_ATTRS, **(attrs or {})}

    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"Body '{body_name}' not found")

    # If a geom with the same name exists, update it; else create one.
    geom = body.find(f"./geom[@name='{attrs['name']}']")
    if geom is None:
        ET.SubElement(body, "geom", attrs)
    else:
        geom.attrib.update(attrs)
    return root


for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    modified = False
    for geom in root.findall(".//geom"):
        # Set or replace attributes
        mesh_name = geom.get("mesh")
        if not mesh_name:
            continue
        is_collision = False
        if "convex" in mesh_name:
            is_collision = True

        geom.set("density", new_density)
        geom.set("friction", new_friction)
        geom.set("solimp", new_solimp)
        geom.set("solref", new_solref)
        geom.set("margin", new_margin)
        geom.attrib.pop("condim", None)

        if is_collision:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

            # geom.set("contype", contype)
            # geom.set("conaffinity", conaffinity)
            geom.set("group", "1")
        else:
            geom.attrib.pop("density", None)
            geom.attrib.pop("friction", None)
            geom.attrib.pop("solimp", None)
            geom.attrib.pop("solref", None)
            geom.attrib.pop("margin", None)

            geom.set("density", new_density)
            geom.set("friction", new_friction)
            geom.set("solimp", new_solimp)
            geom.set("solref", new_solref)
            geom.set("contype", contype)
            geom.set("conaffinity", conaffinity)
            geom.set("group", "0")
            geom.set("condim", condim)


        modified = True

    add_ballast_to_body(root, attrs=BALLAST_ATTRS)

    if modified:
        tree.write(xml_file)
        print(f"Modified: {os.path.basename(xml_file)}")
    else:
        print(f"Skipped: {os.path.basename(xml_file)} (no <geom> tags found)")