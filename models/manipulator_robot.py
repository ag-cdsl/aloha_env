import re
from typing import Optional, List, Tuple, Any

import carb
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path, get_prim_children
from omni.isaac.core.utils.types import ArticulationAction


class ManipulatorRobot(Robot):
    def __init__(
        self,
        prim_path: str,
        robot_path: Optional[str] = None,
        dof_names: Optional[List[str]] = None,
        dof_indices: Optional[List[int]] = None,
        name: str = "manipulator_robot",
        usd_path: Optional[str] = None,
        create_robot: bool = False,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            if create_robot:
                prim = define_prim(prim_path, "Xform")
                if usd_path:
                    prim.GetReferences().AddReference(usd_path)
                else: 
                    carb.log_error("no valid usd path defined to create new robot")
            else:
                carb.log_error(f"no prim at path {prim_path}")
                return

        if robot_path is not None:
            robot_path = "/" + robot_path
            # regex: remove all prefixing "/", need at least one prefix "/" to work
            robot_path = re.sub("^([^\/]*)\/*", "", "/" + robot_path)
            prim_path = prim_path + "/" + robot_path

        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )

        self._dof_names = dof_names
        self._dof_indices = dof_indices
        
        return 


    @property
    def dof_indices(self) -> List[int]:
        return self._dof_indices
    
    @property
    def dof_names(self) -> List[str]:
        return self._dof_names
    

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view=physics_sim_view)
        if self._dof_names is not None:

            self._dof_indices = [
                self.get_dof_index(self._dof_names[i])
                for i in range(len(self._dof_names))
            ]
        elif self._dof_indices is None:
            carb.log_error("DOF Indices missing")

        self._num_dof = len(self._dof_indices)
    


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

    

    def print_details(self):
        print(f"Robot Name: {self.name}")
        print(f"DOF Names: {self._dof_names}")
        print("DOF Indices:")
        for index in self._dof_indices:
            print(f" - {index}")

        print("Primitive details:")
        if hasattr(self, 'manipulator_prim'):
            print(f" - Type: {self.manipulator_prim.GetTypeName()}")
            print(f" - Name: {self.manipulator_prim.GetName()}")
        else:
            print("No primitive details stored.")


    def post_reset(self) -> None:
        super().post_reset()
        # Setup the control mode or any other configurations for the robot's DOFs as necessary
