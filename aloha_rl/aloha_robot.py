from configs.main_config import MainConfig
#from configs.pickplace_config import PickPlaceConfig
from omni.isaac.robot_composer import RobotComposer
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.wheeled_robots.robots import WheeledRobot

import numpy as np

from typing import List

enable_extension('omni.isaac.robot_composer')

import os


class HuskyRobot(Robot):
    def __init__(self, config: MainConfig, husky_name: str , ur_5_name: str, husky_init_pos: List) -> None:
        self._config = config
        self.husky_init_pos = husky_init_pos

        # for parallels simulations  
        self.husky_prim_path: str = find_unique_string_name(
            initial_name=self._config.husky_stage_path, is_unique_fn=lambda x: not is_prim_path_valid(x))
        usd_path: str = os.path.abspath(self._config.husky_usd_path) #'assets/husky/husky.usd'
        prim = get_prim_at_path(self.husky_prim_path)
        
        if not prim.IsValid():
            #add_reference_to_stage(usd_path=usd_path, prim_path=self.husky_prim_path)
            prim = define_prim(self.husky_prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)
        #super().__init__(prim_path=self.husky_prim_path, name=husky_name, position=[0, 0, 0], articulation_controller=None)
        Robot.__init__(self, self.husky_prim_path, husky_name, position=husky_init_pos, articulation_controller=None)
        
        self._load_ur5(ur_5_name)
        self._make_ur5_bind()


    def _load_ur5(self, ur_5_name: str):


        self.ur_5_prim_path = find_unique_string_name(
            initial_name=self._config.ur5_stage_path, is_unique_fn=lambda x: not is_prim_path_valid(x))


        add_reference_to_stage(usd_path=os.path.abspath(self._config.ur5_usd_path), prim_path=self.ur_5_prim_path)

        #end_effector_prim_path = find_unique_string_name(
        #    initial_name=self._config.end_effector_stage_path, is_unique_fn=lambda x: not is_prim_path_valid(x))

        self.gripper = ParallelGripper(
            end_effector_prim_path=self._config.end_effector_stage_path, #.replace('ur5', ur_5_name),
            joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
            joint_opened_positions=np.array([0.0, 0.0]),
            joint_closed_positions=np.array(self._config.gripper_joint_closed_positions),
            #action_deltas=np.array([-0.05, 0.05]),
        )


        self.manipulator = SingleManipulator(
            prim_path=self.ur_5_prim_path,
            name=ur_5_name,
            end_effector_prim_name="ur5_ee_link",
            gripper=self.gripper,
            translation=np.array(self._config.ur5_relative_pose) + self.husky_init_pos,
            # orientation = np.array([1, 0, 0, 0.]),[ 0, 0, 0.7071068, 0.7071068 ]
        )

        default_pose = np.zeros(12)
        default_pose[:6] = list(self._config.joints_default_positions) # (3.1415927, -2.871760, 2.799204, -3.072348, -1.581982, -0.000120)
        self.manipulator.set_joints_default_state(default_pose)
 
    def _make_ur5_bind(self):
        composer = RobotComposer()
        composer.compose(self.husky_prim_path, 
                        self.ur_5_prim_path, 
                        '/put_ur5', 
                        '/ur5_base_link', 
                        single_robot=False)
