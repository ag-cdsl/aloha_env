from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.articulations import Articulation
from typing import Optional


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        #TODO: change the config path
        self._kinematics = LulaKinematicsSolver(robot_description_path="aloha-isaac/rmpflow/robot_descriptor.yaml",
                                                urdf_path="aloha-isaac/assets/ALOHA.urdf")
        if end_effector_frame_name is None:
            end_effector_frame_name = "fl_link6"
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return