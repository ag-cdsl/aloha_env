# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers import ParallelGripper

# from src.config import Config
from controllers.pick_place_controller import MPickPlaceController

from rmpflow.rmpflow import RMPFlowController
from configs.main_config import MainConfig


class PickPlaceController(MPickPlaceController):
    def __init__(
        self, name: str, gripper: ParallelGripper, robot_articulation: Articulation, config: MainConfig
    ) -> None:
        # manipulators_controllers.PickPlaceController.__init__(
        MPickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=list(config.events_dt),
            end_effector_initial_height=config.end_effector_initial_height,
            #ur5_init_pose=list(config.ur5_init_pose),
        )

        return

    def pick_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase of puck up. Otherwise False.
        """
        if self._event > 4.5:
            return True
        else:
            return False