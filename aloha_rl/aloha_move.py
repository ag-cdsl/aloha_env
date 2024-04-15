
ALOHA_ASSET_PATH = "~/.local/share/ov/pkg/isaac_sim-2022.1.1/standalone_examples/aloha_env/aloha_rl/ALOHA.usd"


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
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
# from omni.isaac.wheeled_robots.robots import WheeledRobot
from wheeled_robot import WheeledRobot

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

my_aloha = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/aloha",
        name="my_aloha",
        wheel_dof_names=["left_wheel", "right_wheel"],
        create_robot=True,
        usd_path=ALOHA_ASSET_PATH,
        position=np.array([0, 0.0, 0.005]),
    )
)
my_world.scene.add_default_ground_plane()
my_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
my_world.reset()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        if i >= 0 and i < 1000:
            # forward
            my_aloha.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            print(my_aloha.get_linear_velocity())
        elif i >= 1000 and i < 1300:
            # rotate
            my_aloha.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
            print(my_aloha.get_angular_velocity())
        elif i >= 1300 and i < 2000:
            # forward
            my_aloha.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
        elif i == 2000:
            i = 0
        i += 1
    if args.test is True:
        break


simulation_app.close()
