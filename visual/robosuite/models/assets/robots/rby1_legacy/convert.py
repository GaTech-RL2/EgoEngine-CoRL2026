import xml.etree.ElementTree as ET

def convert_actuators(xml_path, output_path):
    """
    Convert general actuators to motors with joint ranges as control limits.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()


    joint_limits = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        range_attr = joint.get("range")
        if name and range_attr:
            joint_limits[name] = range_attr


    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError("No actuator element found")


    for general in list(actuator.findall("general")):
        act_name = general.get("name")
        joint_name = general.get("joint")

        ctrlrange = joint_limits.get(joint_name, "-1 1")


        motor = ET.Element("motor", {
            "name": act_name,
            "joint": joint_name,
            "ctrllimited": "true",
            "ctrlrange": ctrlrange
        })


        idx = list(actuator).index(general)
        actuator.insert(idx, motor)
        actuator.remove(general)


    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Converted XML written to: {output_path}")


if __name__ == "__main__":
    convert_actuators("/mnt/data2/dexmimic/workspace/egoengine/robosuite/models/assets/robots/rby1/robot.xml", "/mnt/data2/dexmimic/workspace/egoengine/robosuite/models/assets/robots/rby1/robot.xml")
