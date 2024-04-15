import argparse

from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()


simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path


from models.manipulator_robot import ManipulatorRobot
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.types import ArticulationAction


from controllers.simple_controller import SimpleMoveController

from omni.isaac.core.objects import DynamicCuboid


my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

aloha_asset_path = "/home/zhang/.local/share/ov/pkg/isaac_sim-2022.2.1/aloha-isaac/assets/ALOHA.usd"


# Initialize the ManipulatorRobot for the left manipulator
my_aloha = my_world.scene.add(
    ManipulatorRobot(
        prim_path="/World/my_aloha",
        name="my_aloha",
        end_effector_prim_name= "fl_link6",
        gripper_dof_names= ["fl_joint7", "fl_joint8"],
        create_robot=True,
        usd_path=aloha_asset_path,
        position=np.array([0, 0.0, 0.005]),
    )
)



my_world.scene.add(
        DynamicCuboid(
        prim_path= "/World/random_cube",
        name = "fancy_cube",
        position = np.array([1,1,1]),
        scale = np.array([0.05,0.05,0.05]),
        color = np.array([0,0,1.0])
    )
)


my_world.scene.add_default_ground_plane()
my_world.reset()

# print(my_aloha.dof_indices)
# print(my_aloha.dof_names)


# Initialize the controller
joint_index = [14]
simple_controller = SimpleMoveController(
    name="simple_move_controller",
    joint_index=joint_index,
    amplitude=1.0,
    frequency=0.5
)

i = 0
# Main simulation loop
while simulation_app.is_running():
    dt = 1.0 / 120.0  # Assuming a simulation step of 1/60 seconds
    actions = simple_controller.forward(dt)
    my_aloha.apply_joint_actions(actions)

    gripper_positions = my_aloha.gripper.get_joint_positions()

    i += 1
    gripper_positions = my_aloha.gripper.get_joint_positions()
    if i < 500:
            #close the gripper slowly
        my_aloha.gripper.apply_action(
            ArticulationAction(joint_positions=[gripper_positions[0] + 0.1, gripper_positions[1] - 0.1]))
    if i > 500:
            #open the gripper slowly
        my_aloha.gripper.apply_action(
            ArticulationAction(joint_positions=[gripper_positions[0] - 0.1, gripper_positions[1] + 0.1]))
    if i == 1000:
        i = 0

simulation_app.close()
