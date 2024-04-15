# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import numpy as np

from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.wheeled_robots.robots import WheeledRobot
from pxr import UsdPhysics
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.simulation_context import SimulationContext
import numpy as np
from typing import Optional
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from abc import ABCMeta, abstractmethod
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.kit.viewport.window import ViewportWindow
from omni.kit.widget.viewport import ViewportWidget
from omni.isaac.sensor import Camera
from PIL import Image
import omni.isaac.core.utils.numpy.rotations as rot_utils
# from omni.isaac.core.loggers import DataLogger
from loggers.data_logger import DataLogger
import carb
import uuid

# from src.config import Config
# import omni.isaac.core.tasks as tasks

from dataclasses import asdict, dataclass
from datetime import datetime

## add some configs for task 
from omni.isaac.core.utils.string import find_unique_string_name
from omni.usd import get_world_transform_matrix
import omni.replicator.core as rep
from omni.isaac.core import World

import omni                                                     # Provides the core omniverse apis
import asyncio                                                  # Used to run sample asynchronously to not block rendering thread
from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor
from pxr import UsdGeom, Gf, UsdPhysics, Semantics              # pxr usd imports used to create cube

import omni.kit.commands
from pxr import Gf
import omni.replicator.core as rep
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface


from configs.main_config import MainConfig
#from configs.pickplace_config import PickPlaceConfig

from omni.isaac.core.tasks import BaseTask
from robots.husky import HuskyRobot



class PickPlace(BaseTask):
    def __init__(self, 
        name: str,
        world: World,
        config: MainConfig, 
        offset: Optional[np.ndarray] = None,
        ) -> None:
        """
        Args:
            name (str): needs to be unique if added to the World.
            offset (Optional[np.ndarray], optional): offset applied to all assets of the task.

        """

        BaseTask.__init__(self, name=name, offset=offset)

        self._world = world
        self._scene = None
        self._name = name
        self._task_event = 0
        self._offset = offset
        self._task_objects = dict()
        self._robots = dict()
        self.cameras = dict()
        self.depth_annotators = dict()
        
        self._lidar_sensor_interface = None

        self._config = config

        self._husky = None
        self._object = None # object to pick and place
        self._object_initial_position = None
        self._object_initial_orientation = asdict(config).get('_object_initial_orientation', None)
        self._target_position = asdict(config).get('target_position', None)
        self._object_size = asdict(config).get('_object_size', None)

        if self._object_size is None:
            self._object_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if self._object_initial_position is None:
            self._object_initial_position = np.array([0.3, 0.3, 0.3]) / get_stage_units()
        if self._object_initial_orientation is None:
            self._object_initial_orientation = np.array([1, 0, 0, 0])
        if self._target_position is None:
            self._target_position = np.array([-0.3, -0.3, 0]) / get_stage_units()
            self._target_position[2] = self._object_size[2] / 2.0

        self._target_position = self._target_position + self._offset
        return



    def set_up_scene(self, scene: Scene) -> None:
        """Adding assets to the stage as well as adding the encapsulated objects such as XFormPrim..etc
           to the task_objects happens here.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene

        ###################### ADD ROOM ######################

        env_prim_path = find_unique_string_name(
            initial_name=self._config.env_prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x))
        env_name = find_unique_string_name(
            initial_name=self._config.env_name, is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        env_prim = get_prim_at_path(env_prim_path)

        if not env_prim.IsValid():
            add_reference_to_stage(usd_path=self._config.env_usd_path, prim_path=env_prim_path) 
        scene.add(XFormPrim(prim_path=env_prim_path, name=env_name))

        self._env = scene.get_object(env_name)
        self._task_objects[self._env.name] = self._env

        ################ ADD OBJECT TO MANIPULATE ################
        object_name = find_unique_string_name(
            initial_name=self._config.object_name, is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        object_prim_path = find_unique_string_name(
            initial_name=self._config.object_prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x))

        object_prim = get_prim_at_path(object_prim_path)

        if not object_prim.IsValid():
            add_reference_to_stage(usd_path=self._config.object_usd_path, prim_path=object_prim_path)
        scene.add(XFormPrim(prim_path=object_prim_path, name=object_name, position = self._config.object_init_position, scale = self._config.object_scale)) # fix bug with scale for isaac sim 2022 only scale =np.array([0.01, 0.01, 0.01])

        self._object = scene.get_object(object_name)
        self._task_objects[self._object.name] = self._object
        ##########################################################


        self.set_robot()

        # move objects to their position + offset
        self._move_task_objects_to_their_frame() 

        if self._config.log_camera_data:
            self.set_up_camera()
        if self._config.log_lidar_data:
            self.set_up_lidar()

        self.setup_logger()

        return

    def set_up_camera(self) -> None:
        """Setup camera sensors based on config paths

        Args:
            ...
        """
        for camera_relative_path in self._config.cameras:
            camera_path = os.path.join(self._husky.husky_prim_path, camera_relative_path)
            camera_name = camera_relative_path.split('/')[-1]

            print(camera_relative_path)
            print(camera_path)
    
            camera = Camera(
                prim_path=camera_path,
                name = camera_name
                )

            camera.initialize()
            camera.add_motion_vectors_to_frame()

            self.depth_annotators[camera_name] = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            self.depth_annotators[camera_name].attach([camera._render_product_path])

            self.cameras[camera_name] = camera

        return

    def set_up_lidar(self):
        """ Setup lidar sensor based on config paths
            check docs for more information: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.range_sensor/docs/index.html
        """
        timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
        lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR

        #  = self._config.lidar_prim_relative_path
        #lidar_path = os.path.join(self._husky.husky_prim_path, self._config.lidar_relative_path)

        result, prim = omni.kit.commands.execute(
                    "RangeSensorCreateLidar",
                    path=self._config.lidar_relative_path,
                    parent=self._husky.husky_prim_path,
                    min_range = self._config.min_range,
                    max_range = self._config.max_range,
                    draw_points = self._config.draw_points,
                    draw_lines = self._config.draw_lines,
                    horizontal_fov = self._config.horizontal_fov,
                    vertical_fov = self._config.vertical_fov,
                    horizontal_resolution = self._config.horizontal_resolution,
                    vertical_resolution = self._config.vertical_resolution,
                    rotation_rate = self._config.rotation_rate,
                    high_lod = self._config.high_lod,
                    yaw_offset = self._config.yaw_offset,
                    enable_semantics = self._config.enable_semantics
                )
        UsdGeom.XformCommonAPI(prim).SetTranslate(self._config.lidar_pos)
        self._lidar_sensor_interface = acquire_lidar_sensor_interface()

        

        return 

    def setup_logger(self) -> None: 
        """Setup data logger based on config 

        Args:
            ...
        """
        self._data_logger = DataLogger(config = self._config)

        return 
    
    def data_frame_logging_func(self):

        lidar_path = os.path.join(self._husky.husky_prim_path, self._config.lidar_relative_path)
        print(lidar_path)
        
        data = {
            "current_time_step": self._world.current_time_step_index,
            "current_time" : self._world.current_time,
            "husky_joint_positions": self._scene.get_object(self._husky.name).get_joint_positions(),
            "husky_applied_joint_positions": self._scene.get_object(self._husky.name).get_applied_action().joint_positions,
            "ur_5_joint_positions": self._scene.get_object(self._ur5.name).get_joint_positions(),
            "ur_5_applied_joint_positions": self._scene.get_object(self._ur5.name).get_applied_action().joint_positions,
            "target_position": self._scene.get_object(self._object.name).get_world_pose()[0],
            "point_cloud": self._lidar_sensor_interface.get_point_cloud_data(lidar_path)
        }        

        for camera_relative_path in self._config.cameras:
            camera_path = os.path.join(self._husky.husky_prim_path, camera_relative_path)
            camera_name = camera_relative_path.split('/')[-1]

            depth = self.depth_annotators[camera_name].get_data()
            rgb = self.cameras[camera_name].get_rgba()[:, :, :3] # you can log rbga if you want

            data['_'.join(['rbg_image', camera_name])] = rgb
            data['_'.join(['depth_image', camera_name])] = depth

            # depth_image = Image.fromarray(depth).convert("L")
            # depth_image.save(depth_path, format="PNG")

            # image = Image.fromarray(rgb)
            # image.save(img_path, format="PNG")

        return data


    @abstractmethod
    def set_robot(self) -> None:
        """Husky and ur5 setup
        """
        
        # for multiple simulation add names like husky_0, husky_1 and etc...
        husky_name = find_unique_string_name(
            initial_name="husky", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        ur_5_name = find_unique_string_name(
            initial_name="ur5", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )

        self._husky = HuskyRobot(self._config, husky_name, ur_5_name, np.array(self._config.husky_init_pose) + self._offset)
        self._ur5 = self._husky.manipulator

        self._robots[husky_name] = self._husky
        self._robots[ur_5_name] = self._ur5

        stage = get_current_stage()
        
        prim_trans_point_path = os.path.join(self._husky.ur_5_prim_path, self._config.trans_pint_relative_path)

        prim_trans_point = stage.GetPrimAtPath(prim_trans_point_path)
        self._robots["trans_point"] = prim_trans_point


        self.scene.add(self._husky)
        self.scene.add(self._husky.manipulator)


        return

    def _move_task_objects_to_their_frame(self):

        """_summary_
        """

        # if self._task_path:
        # TODO: assumption all task objects are under the same parent
        # Specifying a task  path has many limitations atm
        # XFormPrim(prim_path=self._task_path, position=self._offset)
        # for object_name, task_object in self._task_objects.items():
        #     new_prim_path = self._task_path + "/" + task_object.prim_path.split("/")[-1]
        #     task_object.change_prim_path(new_prim_path)
        #     current_position, current_orientation = task_object.get_world_pose()
        for object_name, task_object in self._task_objects.items():
            current_position, current_orientation = task_object.get_world_pose()

            #task_object.set_local_pose(translation=current_position + self._offset, orientation=current_orientation)
            task_object.set_world_pose(position=current_position + self._offset)
            task_object.set_default_state(position=current_position + self._offset)
        return

    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """Returns current observations from the objects needed for the behavioral layer.

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        # stage = get_current_stage()
        # object_prim = stage.GetPrimAtPath(self.object_prim_path)
        # object_tr_matrix = get_world_transform_matrix(object_prim)
        # object_position = object_tr_matrix.ExtractTranslation()

        object_position, object_orientation = self._object.get_local_pose()
        husky_position, husky_orientation = self._husky.get_local_pose()


        ur_5_joints_state = self._husky.manipulator.get_joints_state()
        ur_5_end_effector_position, _ = self._husky.manipulator.end_effector.get_local_pose()
        ur5_position, ur5_orientation = self._husky.manipulator.get_local_pose()


        observations = {
            self.name + '_event': self._task_event,
            self._object.name: {
                "position": object_position,
                "orientation": object_orientation,
                "target_position": self._target_position,
            }, 
            self._husky.name: {
                "husky_position":husky_position,
                "husky_orientation":husky_orientation,
            },
            self._husky.manipulator.name: {
                "joint_positions": ur_5_joints_state.positions,
                "end_effector_position": ur_5_end_effector_position,
                "ur5_position":ur5_position,
                "ur5_orientation":ur5_orientation,
            },
        }
        return observations


    def get_params(self) -> dict:
        """Gets the parameters of the task.
           This is defined differently for each task in order to access the task's objects and values.
           Note that this is different from get_observations. 
           Things like the robot name, block name..etc can be defined here for faster retrieval. 
           should have the form of params_representation["param_name"] = {"value": param_value, "modifiable": bool}
    
        Raises:
            NotImplementedError: [description]

        Returns:
            dict: defined parameters of the task.
        """
        params_representation = dict()
        position, orientation = self._cube.get_local_pose()
        params_representation["cube_position"] = {"value": position, "modifiable": True}
        params_representation["cube_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["target_position"] = {"value": self._target_position, "modifiable": True}
        params_representation["cube_name"] = {"value": self._cube.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation


    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Returns True of the task is done.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world.
        """
        # self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._task_event = 0
        return

    def get_description(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return ""

    def cleanup(self) -> None:
        """Called before calling a reset() on the world to removed temporarly objects that were added during
           simulation for instance.
        """
        return

    def set_params(
        self,
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        ) -> None:
        """
            Changes the modifiable paramateres of the task
        """
        if target_position is not None:
            self._target_position = target_position
        if object_position is not None or object_orientation is not None:
            self._object.set_local_pose(translation=object_position, orientation=object_orientation)
        return

