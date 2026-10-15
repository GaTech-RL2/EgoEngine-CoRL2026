import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def fix_collision_geoms(input_path, output_path):
    """
    Parses the input MuJoCo XML, sets contype/conaffinity to '1' 
    on all mesh <geom> entries, and writes a prettified output XML.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Iterate over all geom elements
    for geom in root.findall('.//geom'):
        if 'mesh' in geom.attrib:
            # Enable collision by setting contype and conaffinity to 1
            geom.set('contype', '1')
            geom.set('conaffinity', '1')

    # Write out the prettified XML
    pretty_xml = prettify(root)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    print(f"Fixed XML written to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enable collision on mesh geoms in a MuJoCo XML.")
    parser.add_argument("input_xml", help="Path to your original x_left_hand.xml")
    parser.add_argument("output_xml", help="Path to save the corrected XML (e.g., x_left_hand_fixed.xml)")
    args = parser.parse_args()

    fix_collision_geoms(args.input_xml, args.output_xml)