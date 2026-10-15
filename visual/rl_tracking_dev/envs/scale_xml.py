from glob import glob
import xml.etree.ElementTree as ET

all_xmls = glob("./taco_objects/*.xml")

for fpath in all_xmls:
    tree = ET.parse(fpath)
    root = tree.getroot()

    # Loop through all <mesh> elements and update the scale
    for mesh in root.findall(".//mesh"):
        if "scale" in mesh.attrib:
            mesh.set("scale", "1 1 1")

    # Save the modified XML
    tree.write(fpath)