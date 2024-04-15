import re
from typing import Optional, List, Tuple, Any

import carb
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path


# Assuming ParallelGripper class or similar is available for use
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper


class ManipulatorRobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: str = "manipulator_robot",
        left: bool = True,
        usd_path: Optional[str] = None,
        create_robot: bool = False,
        end_effector_prim_name: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path=prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name

        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/aloha-isaac/assets/ALOHA.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/fl_link8"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["fl_joint7", "fl_joint8"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/fl_link8"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["fl_joint7", "fl_joint8"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.05, 0.05]) / get_stage_units()
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )
        return

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    @property
    # def dof_indices(self) -> List[int]:
    #     return self._dof_indices
    
    # @property
    # def dof_names(self) -> List[str]:
    #     return self._dof_names

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper
    
    def apply_action(self, control_actions: ArticulationAction) -> None:
        return super().apply_action(control_actions)

    def apply_joint_actions(self, actions: ArticulationAction) -> None:
        if actions.joint_positions is not None:
            # Check if the input is a dictionary with indices, then convert to full list
            if isinstance(actions.joint_positions, dict):
                joint_positions = np.zeros(self.num_dof)
                for index, position in actions.joint_positions.items():
                    joint_positions[index] = position
            else:
                joint_positions = actions.joint_positions

            if len(joint_positions) != self.num_dof:
                raise Exception("Length of joint_positions must be the same as the number of DOFs")

            joint_actions = ArticulationAction(joint_positions=joint_positions)
        else:
            raise Exception("Joint positions not specified")

        # Apply the action using the Robot class method
        self.apply_action(joint_actions)

    # def print_details(self):
    #     print(f"Robot Name: {self.name}")
    #     print(f"DOF Names: {self._dof_names}")
    #     print("DOF Indices:")
    #     for index in self._dof_indices:
    #         print(f" - {index}")

    #     print("Primitive details:")
    #     if hasattr(self, 'manipulator_prim'):
    #         print(f" - Type: {self.manipulator_prim.GetTypeName()}")
    #         print(f" - Name: {self.manipulator_prim.GetName()}")
    #     else:
    #         print("No primitive details stored.")



    def post_reset(self) -> None:
        """[summary]
        """
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return
