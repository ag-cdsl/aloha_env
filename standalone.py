from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

my_world = World(stage_units_in_meters=1.0)

asset_path = "aloha-isaac/assets/ALOHA.usd"

add_reference_to_stage(usd_path=asset_path, prim_path="/World/aloha")

gripper = ParallelGripper(
    end_effector_prim_path="/World/aloha/fl_link6",
    joint_prim_names=["fl_joint7", "fl_joint8"],
    joint_opened_positions=np.array([0,0]),
    joint_closed_positions=np.array([1,1]),
    action_deltas=np.array([1,1]),
)

my_aloha = my_world.scene.add(SingleManipulator(
    prim_path="/World/aloha", name="aloha_robot",
    end_effector_prim_name="fl_link6",
    gripper=gripper
))

my_world.scene.add_default_ground_plane()

my_world.reset()

i = 0

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        gripper_positions = my_aloha.gripper.get_joint_positions()
        if i < 500:
            #close the gripper slowly
            my_aloha.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.1, gripper_positions[1] - 0.1]))
        if i > 500:
            #open the gripper slowly
            my_aloha.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.1, gripper_positions[1] + 0.1]))
        if i == 1000:
            i = 0

simulation_app.close()