import numpy as np
import collections
from copy import copy

from robosuite.models.objects import MujocoObject
from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_multiply, mat2quat, convert_quat

from robosuite.utils.placement_samplers import ObjectPositionSampler

class UniformRandomPoseSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        obj_poses,
        mujoco_objects=None,
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.obj_poses = obj_poses

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(base_offset)

        # Sample pos and quat for all objects assigned to this sampler
        for obj, obj_pose in zip(self.mujoco_objects, self.obj_poses):
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False


            object_x = obj_pose[0, 3] + base_offset[0]
            object_y = obj_pose[1, 3] + base_offset[1]
            object_z = self.z_offset + base_offset[2]
            if on_top:
                object_z -= bottom_offset[-1]

            location_valid = True
            if self.ensure_valid_placement:
                for (x, y, z), _, other_obj in placed_objects.values():
                    if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                    ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                        location_valid = False

            if location_valid:
                # random rotation
                obj_rot = obj_pose[:3, :3]
                quat = convert_quat(mat2quat(obj_rot), "wxyz")

                # location is valid, put the object down
                pos = (object_x, object_y, object_z)
                placed_objects[obj.name] = (pos, quat, obj)
                success = True

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects
