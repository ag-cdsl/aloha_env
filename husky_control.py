
from omni.isaac.kit import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Set display options to show default grid
}


simulation_app = SimulationApp(launch_config=CONFIG)

from omni.isaac.core.utils.extensions import enable_extension

# Enable WebSocket Livestream extension
# Default URL: http://localhost:8211/streaming/client/


# Livestream settings
simulation_app.set_setting("/app/window/drawMouse", True)
simulation_app.set_setting("/app/livestream/proto", "ws")
simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
simulation_app.set_setting("app/livestream/websocket/encoder_selection", 'OPENH264')
simulation_app.set_setting("/ngx/enabled", False)


enable_extension("omni.services.streamclient.websocket")
enable_extension('omni.isaac.robot_composer')




from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core import World
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils import nucleus, stage
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.objects import DynamicCuboid


from typing import Optional
from pxr import UsdPhysics, Usd, Sdf
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import get_stage_units

from pxr import UsdPhysics
from datetime import datetime

import os
import numpy as np


import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.string import find_unique_string_name

from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.motion_generation import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.articulations import ArticulationSubset
from pxr import UsdPhysics, Usd, Sdf
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import get_stage_units

from omni.isaac.core.prims.xform_prim import XFormPrim
import omni.isaac.core.utils.prims as prims_utils

import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.articulations import Articulation

from controllers.pick_controller import MyPickController
from controllers.put_controller import MyPutController
from rmpflow.rmpflow import RMPFlowController

from tasks.pick_place import PickPlace

from configs.main_config import MainConfig
from controllers.husky_controller import HuskyController
import asyncio
#from configs.pickplace_config import PickPlaceConfig

#from omni.isaac.core.loggers import DataLogger
from loggers.data_logger import DataLogger

############################ INIT ############################

config = MainConfig()

tasks = []
num_of_tasks = 2
husky_controllers = []

object_names = []
objects = []

my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()


############################ SETUP SCENE #############################
'''
    Add multiple tasks
'''
for i in range(num_of_tasks):
    print(i)
    my_world.add_task(PickPlace(name="task_" + str(i), world = my_world, config = config, offset=np.array([0, (i * 30), 0])))

my_world.reset()

############################ SETUP TASKS #############################
'''
    Add info about tasks
'''

for i in range(num_of_tasks):
    task = my_world.get_task(name="task_" + str(i))
    tasks.append(task)

    controller = HuskyController(task, my_world, simulation_app)
    husky_controllers.append(controller)    



############################ START SIMULATION ############################ 


while simulation_app.is_running() and not simulation_app.is_exiting():

    my_world.step(render=True)

    if my_world.is_playing():
        
        
        for i in range(num_of_tasks):

            current_observations = tasks[i].get_observations()

            target_position = current_observations[tasks[i]._object.name]['target_position']
            object_position = current_observations[tasks[i]._object.name]['position']
            
            # stage = get_current_stage()
            # object_prim = stage.GetPrimAtPath(self.object_prim_path)
            # object_tr_matrix = get_world_transform_matrix(object_prim)
            # object_position = object_tr_matrix.ExtractTranslation()

            goal_position = tuple(object_position)

            if current_observations[tasks[i].name + "_event"] == 0:
                husky_controllers[i].move_to_location(goal_position = object_position, min_distance = 1.0)
            elif current_observations[tasks[i].name + "_event"] == 1:
                object_position = current_observations[tasks[i]._object.name]['position']
                husky_controllers[i].pickup_object()
            elif current_observations[tasks[i].name + "_event"] == 2:
                husky_controllers[i].move_to_location(goal_position = target_position, min_distance = 1.0)
            elif current_observations[tasks[i].name + "_event"] == 3:
                husky_controllers[i].put_object(object_position)
            elif current_observations[tasks[i].name + "_event"] == 4:
                print('TRAJECTORY DONE!!!')

            # log data 
            data = tasks[i].data_frame_logging_func()
            tasks[i]._data_logger.add_data(data)

    simulation_app.update()
simulation_app.close() 
