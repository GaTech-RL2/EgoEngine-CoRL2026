import os
import trimesh

# Cylinder parameters (radius=0.03m, height=0.03m)
radius = 0.03
height = 0.03

# Target folder for RIGHT_MOUNT assets
output_dir = '/mnt/data2/dexmimic/workspace/egoengine/robosuite/models/assets/robots/rby1/assets/RIGHT_MOUNT'
os.makedirs(output_dir, exist_ok=True)

# Create cylinder mesh
cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=64)

# Export to OBJ (and accompanying MTL)
obj_path = os.path.join(output_dir, 'ee_mount_r.obj')
cylinder.export(obj_path)

print(f'Generated cylindrical mount mesh at: {obj_path}')
