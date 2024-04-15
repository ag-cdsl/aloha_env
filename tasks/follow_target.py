from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np


# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "aloha_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        #TODO: change this to the robot usd file.
        asset_path = "aloha-isaac/assets/ALOHA.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/aloha")

        gripper = ParallelGripper(
            end_effector_prim_path="/World/aloha/fl_link6",
            joint_prim_names=["fl_joint7", "fl_joint8"],
            joint_opened_positions=np.array([0,0]),
            joint_closed_positions=np.array([1,1]),
            action_deltas=np.array([1,1]),
        )

        manipulator = SingleManipulator(
                        prim_path="/World/aloha", name="aloha_robot",
                        end_effector_prim_name="fl_link6",
                        gripper=gripper
                    )

        return manipulator