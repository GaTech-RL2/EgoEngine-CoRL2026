import os
import pickle
from glob import glob
import numpy as np
_alias_map = [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
]
for _name, _type in _alias_map:
    if not hasattr(np, _name):
        setattr(np, _name, _type)
import torch
import trimesh
from trimesh import Trimesh
from os.path import join, basename, dirname

# Make sure you have hand_pose_loader.py in the same directory.
from hand_pose_loader import mano_params_to_hand_info

OBJ_SCALE = 0.01


def create_obj_urdf(obj_path, scale=(1, 1, 1)):
    """Create a minimal URDF file for a given .obj mesh."""
    obj_path = os.path.abspath(obj_path)
    mesh_dir = dirname(obj_path)
    mesh_name = basename(obj_path)

    urdf_content = f"""<?xml version="1.0"?>
        <robot name="dynamic_obj">
        <link name="base">
            <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_name}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_name}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
            </collision>
        </link>
        </robot>
        """
    urdf_name = mesh_name.replace(".obj", ".urdf")
    urdf_path = join(mesh_dir, urdf_name)
    with open(urdf_path, "w") as f:
        f.write(urdf_content)
    return urdf_path


def _add_pose(pose, og_sz=0.005, ax_len=0.02):
    """Visualize a 4×4 pose matrix as a coordinate frame."""
    frame = trimesh.creation.axis(origin_size=og_sz, axis_length=ax_len)
    frame.apply_transform(pose)
    return frame


def load_and_visualize_taco_data(data_dir, mesh_dir, frame_to_show=60, device="cuda:0"):
    """
    Load TACO data and visualize it in an interactive trimesh window.
    `data_dir` should point to Hand_Poses/<triplet>/<sequence>.
    """
    # Derive the sibling Object_Poses directory automatically
    object_pose_dir = data_dir.replace(os.sep + "Hand_Poses" + os.sep,
                                       os.sep + "Object_Poses" + os.sep)
    if not os.path.isdir(object_pose_dir):
        raise FileNotFoundError(
            f"Object pose directory not found: {object_pose_dir}"
        )

    # --- 1. Load object pose data ---
    target_pose_path = glob(join(object_pose_dir, "target_*.npy"))[0]
    tool_pose_path = glob(join(object_pose_dir, "tool_*.npy"))[0]

    target_obj_id = basename(target_pose_path).split(".")[0].split("_")[1]
    tool_obj_id = basename(tool_pose_path).split(".")[0].split("_")[1]

    # Load object models and scale them (cm → m)
    target_model = trimesh.load(join(mesh_dir, f"{target_obj_id}_cm.obj"))
    target_model.vertices *= OBJ_SCALE
    tool_model = trimesh.load(join(mesh_dir, f"{tool_obj_id}_cm.obj"))
    tool_model.vertices *= OBJ_SCALE

    # Load object pose sequences
    target_poses = np.load(target_pose_path)
    tool_poses = np.load(tool_pose_path)
    n_frames = tool_poses.shape[0]

    # --- 2. Load hand data ---
    print("Loading hand shapes and poses...")
    right_hand_shape_path = join(data_dir, "right_hand_shape.pkl")
    left_hand_shape_path = join(data_dir, "left_hand_shape.pkl")
    right_hand_pose_path = join(data_dir, "right_hand.pkl")
    left_hand_pose_path = join(data_dir, "left_hand.pkl")

    # Load MANO shape parameters
    with open(right_hand_shape_path, "rb") as f:
        right_hand_beta = pickle.load(f)["hand_shape"].reshape(10).detach().cpu().numpy()
    with open(left_hand_shape_path, "rb") as f:
        left_hand_beta = pickle.load(f)["hand_shape"].reshape(10).detach().cpu().numpy()

    # Get per‑frame hand vertices
    right_hand_vertices, right_hand_joints, right_hand_frames, hand_faces = \
        mano_params_to_hand_info(
            right_hand_pose_path,
            mano_beta=right_hand_beta,
            side="right",
            return_faces=True,
            return_frames=True,
            device=device
        )

    left_hand_vertices, left_hand_joints, left_hand_frames = \
        mano_params_to_hand_info(
            left_hand_pose_path,
            mano_beta=left_hand_beta,
            side="left",
            return_frames=True,
            device=device
        )

    # --- 3. Assemble and visualize the scene ---
    if frame_to_show >= n_frames:
        print(f"Warning: frame_to_show ({frame_to_show}) is out of bounds. Using last frame ({n_frames - 1}).")
        frame_to_show = n_frames - 1

    print(f"Visualizing frame {frame_to_show}...")
    scene = trimesh.Scene()

    # Hand meshes
    rh_mesh = Trimesh(vertices=right_hand_vertices[frame_to_show], faces=hand_faces)
    lh_mesh = Trimesh(vertices=left_hand_vertices[frame_to_show], faces=hand_faces)

    # Object meshes
    tool_pose = tool_poses[frame_to_show]
    target_pose = target_poses[frame_to_show]

    tool_mesh = tool_model.copy()
    tool_mesh.apply_transform(tool_pose)

    target_mesh = target_model.copy()
    target_mesh.apply_transform(target_pose)

    # Add geometries to the scene
    scene.add_geometry([rh_mesh, lh_mesh, tool_mesh, target_mesh])

    # Coordinate frames for debugging
    scene.add_geometry([_add_pose(tool_pose, ax_len=0.1), _add_pose(target_pose, ax_len=0.1)])
    for pose in np.concatenate([right_hand_frames[frame_to_show],
                                left_hand_frames[frame_to_show]]):
        scene.add_geometry(_add_pose(pose, ax_len=0.03, og_sz=0.002))

    # Show the interactive scene
    scene.show()


if __name__ == '__main__':
    # Paths
    base_dir = "/mnt/data2/dexmimic/datasets/egoengine/TACO"
    mesh_dir = join(base_dir, "Object_Models/object_models_released")

    triplet = "(stir, spoon, bowl)"
    sequence = "20231105_019"

    demo_dir = join(base_dir, "Hand_Poses", triplet, sequence)

    # Frame to display
    frame_to_show = 75

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    load_and_visualize_taco_data(
        data_dir=demo_dir,
        mesh_dir=mesh_dir,
        frame_to_show=frame_to_show,
        device=device
    )
