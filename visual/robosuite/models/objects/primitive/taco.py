from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

class TACOObject(MujocoXMLObject):
    """
    Household object imported from the **TACO** dataset.

    Args:
        xml_id (str):   File stem of the XML inside `assets/TACO_object/`.
                        Example: "001_cm" loads "TACO_object/001_cm.xml".
        name (str):     MuJoCo name. Defaults to "taco_<xml_id>".
        joints:         Same as in MujocoXMLObject; by default a single free joint.
        obj_type (str): {"all", "visual", "collision"} — pass-through.
        duplicate_collision_geoms (bool): Whether to auto-duplicate visual geoms
                                          for collisions if they are missing.
    """

    def __init__(
        self,
        xml_id: str,
        name: str | None = None,
        joints="default",
        obj_type: str = "all",
        duplicate_collision_geoms: bool = True,
    ):
        # default name → taco_001_cm, etc.
        name = name or f"taco_{xml_id}"

        # default → one 6-DoF free joint with mild damping
        if joints == "default":
            joints = [dict(type="free", damping="0.0005")]

        super().__init__(
            fname=xml_path_completion(f"objects/{xml_id}.xml"),
            name=name,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )