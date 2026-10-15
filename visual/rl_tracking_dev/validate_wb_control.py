
import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
from robosuite.controllers import load_composite_controller_config
from custom_tasks import *

import json
import mujoco


controller_config = load_composite_controller_config(controller="RBY1")
robot_name = "RBY1"

# controller_config = load_composite_controller_config(controller="basic")
controller_config = load_composite_controller_config(controller="RBY1_TWO_ARMS")
robot_name = "RBY1ArmsOnly"

# controller_config = load_composite_controller_config(controller="RBY1")
# robot_name = "GR1FixedLowerBody"


print(controller_config)

# create environment instance
env = suite.make(
    target_obj_info={},
    tool_obj_info={},
    env_name="CustomTask", # try with other tasks like "Stack" and "Door"
    robots=robot_name,  # try with other robots like "Sawyer" and "Jaco"
    # gripper_types=["InspireLeftHand", "InspireRightHand"],
    controller_configs=controller_config,
    has_renderer=False,
    camera_names="frontview",
    has_offscreen_renderer=True,
    use_camera_obs=True,
    table_offset=(0, 0, 0.4)
)



robot = env.robots[0]
robot.print_action_info()

robot_model = robot.robot_model
joint_names = [mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(robot_model.njnt)]
print(joint_names, '?')
print(len(joint_names))

# reset the environment
obs = env.reset()
a_spec = env.action_spec
o_spec = env.observation_spec()
# print(a_spec, o_spec)

init_qpos = obs["robot0_joint_pos"]
action = np.random.randn(*env.action_spec[0].shape) * 0.1
action[:len(init_qpos)] += init_qpos

reached = []


def plot_tracking_with_image(target, reached, n_per_row, image, scale=0.5):
    reached = np.array(reached)
    m = reached.shape[1]
    n_rows = int(np.ceil((m + 1) / n_per_row))  # +1 for image

    fig, axs = plt.subplots(n_rows, n_per_row, figsize=(4 * n_per_row * scale, 3 * n_rows * scale))
    axs = np.atleast_2d(axs)
    axs = axs.reshape(n_rows, n_per_row)

    # Plot tracking curves
    for ii in range(m):
        ax = axs[ii // n_per_row, ii % n_per_row]
        ax.plot(reached[:, ii])
        ax.axhline(y=target[ii], color='r', linestyle='--')
        ax.set_title(f'Dimension {ii}')

    # Plot image in the next cell
    img_cell_idx = m  # image goes after all curves
    img_ax = axs[img_cell_idx // n_per_row, img_cell_idx % n_per_row]
    img_ax.imshow(image)
    img_ax.axis('off')
    img_ax.set_title("Appended Image")

    # Hide any remaining empty subplots
    for jj in range(m + 1, n_rows * n_per_row):
        axs[jj // n_per_row, jj % n_per_row].axis('off')

    plt.tight_layout()
    plt.show()




for i in range(1000):

    obs, reward, done, info = env.step(action)  # take action in the environment

    print(obs.keys())
    print(obs["robot0_joint_pos"].shape)
    reached.append(obs["robot0_joint_pos"])

    plot_tracking_with_image(action, reached, 5, obs["frontview_image"][::-1])

