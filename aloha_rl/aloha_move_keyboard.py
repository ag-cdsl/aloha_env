# # Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto. Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.
# #
# import argparse

# from omni.isaac.kit import SimulationApp
# import os
# parser = argparse.ArgumentParser()
# parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
# args, unknown = parser.parse_known_args()


# simulation_app = SimulationApp({"headless": False,})

# import carb
# import numpy as np
# from omni.isaac.core import World
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
# # from omni.isaac.wheeled_robots.robots import WheeledRobot
# from wheeled_robot import WheeledRobot
# from omni.isaac.core.utils.prims import create_prim

# my_world = World(stage_units_in_meters=1.0)
# assets_root_path = get_assets_root_path()
# if assets_root_path is None:
#     carb.log_error("Could not find Isaac Sim assets folder")
# # aloha_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
# aloha_asset_path = "/home/ladanova_sv/.local/share/ov/pkg/isaac_sim-2022.1.1/standalone_examples/aloha_env/aloha_rl/ALOHA_with_sensor_02.usd"
# # create_prim(
# #                 prim_path=f"/World/aloha",
# #                 translation=(0,0,0),
# #                 usd_path=aloha_asset_path
# #             )
# my_aloha = my_world.scene.add(
#     WheeledRobot(
#         prim_path="/World/aloha",
#         name="my_aloha",
#         wheel_dof_names=["left_wheel", "right_wheel"],
#         create_robot=True,
#         usd_path=aloha_asset_path,
#         position=np.array([0, 0.0, 0.005]),
#     )
# )
# # print(my_aloha)
# my_world.scene.add_default_ground_plane()
# base_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
# my_world.reset()

# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
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
# aloha_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
aloha_asset_path = "/home/ladanova_sv/.local/share/ov/pkg/isaac_sim-2022.1.1/standalone_examples/aloha_env/aloha_rl/ALOHA_with_sensor_02.usd"
my_aloha = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/aloha",
        name="my_aloha",
        wheel_dof_names=["left_wheel", "right_wheel"],
        create_robot=True,
        usd_path=aloha_asset_path,
        position=np.array([0, 0.0, 0.005]),
    )
)
my_world.scene.add_default_ground_plane()
base_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
my_world.reset()


#____________________________________________START_BLOCK_KEYBOARD
action = np.array([0.0, 0.0])
import omni

def keyboard_events(event):
    global action
    if (event.type == carb.input.KeyboardEventType.KEY_PRESS
        or event.type == carb.input.KeyboardEventType.KEY_REPEAT):
        if event.input == carb.input.KeyboardInput.W:  #NUMPAD_8
            base_controller.initialVelocity = 0.3
            base_controller.initialAccel = 0.2
            action = action*0.2 + 0.8*np.array([20, 0.0])
        if event.input == carb.input.KeyboardInput.S: # NUMPAD_2
            base_controller.initialVelocity = 0.3
            base_controller.initialAccel = 0.2
            action = action*0.2 + 0.8*np.array([-20, 0.0])
        if event.input == carb.input.KeyboardInput.A: #NUMPAD_4
            base_controller.initialVelocity = 0.5
            base_controller.initialAccel = 0.2
            action =action*0.5 + 1.5*np.array([0.0, np.pi / 1])
        if event.input == carb.input.KeyboardInput.D: # NUMPAD_6
            base_controller.initialVelocity = 0.5
            base_controller.initialAccel = 0.2
            action = action*0.5 + 1.5*np.array([0.0, -np.pi / 1])
    if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        action = np.array([0.0, 0.0])


app_window = omni.appwindow.get_default_app_window()
keyboard = app_window.get_keyboard()
input = carb.input.acquire_input_interface()

keyboard_sub_id = input.subscribe_to_keyboard_events(keyboard, keyboard_events)
#______________________________________________END_BLOCK_KEYBOARD


i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        my_aloha.apply_wheel_actions(base_controller.forward(command=action))#move
    simulation_app.update()
    if args.test is True:
        break


simulation_app.close()
#____________________________________________START_BLOCK_KEYBOARD
action = np.array([0.0, 0.0])
import omni

def keyboard_events(event):
    global action
    if (event.type == carb.input.KeyboardEventType.KEY_PRESS
        or event.type == carb.input.KeyboardEventType.KEY_REPEAT):
        if event.input == carb.input.KeyboardInput.W:  #NUMPAD_8
            base_controller.initialVelocity = 0.3
            base_controller.initialAccel = 0.2
            action = action*0.2 + 0.8*np.array([20, 0.0])
        if event.input == carb.input.KeyboardInput.S: # NUMPAD_2
            base_controller.initialVelocity = 0.3
            base_controller.initialAccel = 0.2
            action = action*0.2 + 0.8*np.array([-20, 0.0])
        if event.input == carb.input.KeyboardInput.A: #NUMPAD_4
            base_controller.initialVelocity = 0.5
            base_controller.initialAccel = 0.2
            action =action*0.5 + 1.5*np.array([0.0, np.pi / 1])
        if event.input == carb.input.KeyboardInput.D: # NUMPAD_6
            base_controller.initialVelocity = 0.5
            base_controller.initialAccel = 0.2
            action = action*0.5 + 1.5*np.array([0.0, -np.pi / 1])
    if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        action = np.array([0.0, 0.0])


app_window = omni.appwindow.get_default_app_window()
keyboard = app_window.get_keyboard()
input = carb.input.acquire_input_interface()

keyboard_sub_id = input.subscribe_to_keyboard_events(keyboard, keyboard_events)
#______________________________________________END_BLOCK_KEYBOARD


i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        my_aloha.apply_wheel_actions(base_controller.forward(command=action))#move
    simulation_app.update()
    if args.test is True:
        break


simulation_app.close()
