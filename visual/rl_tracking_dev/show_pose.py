import trimesh
import numpy as np
from urdfpy import URDF


scene = trimesh.Scene()

# desk_urdf = "/mnt/data2/dexmimic/workspace/egoengine/retargeting/assets/desk/desk.urdf"
# robot = URDF.load(desk_urdf)

# scene = robot.scene()




aria_cam_pose = np.array([
    [-0.01579835, -0.37887728,  0.92531198,  0.23750265],
    [-0.99969959,  0.02332757, -0.00751672, -0.06064347],
    [-0.01873737, -0.9251529,  -0.37913203,  1.47158396],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
], dtype=np.float32)


taco_pose = np.array([[ 0.07808541, -0.94811852,  0.3081783 , -0.48817091],
       [-0.99694496, -0.07483678,  0.02236616, -0.05263493],
       [ 0.00185732, -0.30898329, -0.95106568,  1.48029928],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
taco_pose[0, 3] += 0.7


taco_axis = trimesh.creation.axis(
    origin_size=0.02,   # sphere at origin
    axis_length=0.1,     # length of X,Y,Z axes
    origin_color=(255, 255, 0, 255)
)
scene.add_geometry(taco_axis, transform=taco_pose)

aria_axis = trimesh.creation.axis(
    origin_size=0.02,   # sphere at origin
    axis_length=0.1,     # length of X,Y,Z axes
    origin_color=(255, 0, 0, 255)
)

scene.add_geometry(aria_axis, transform=aria_cam_pose)
# Add world axis at origin (optional)

ori_axis = trimesh.creation.axis(
    origin_size=0.1,   # sphere at origin
    axis_length=0.4,     # length of X,Y,Z axes
    origin_color=(128, 128, 128, 255)
)
scene.add_geometry(ori_axis, )


scene.show()