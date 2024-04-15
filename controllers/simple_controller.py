from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
from typing import List



class SimpleMoveController(BaseController):
    def __init__(self, name: str, joint_index: int, amplitude: float = 1.0, frequency: float = 0.5):
        super(SimpleMoveController, self).__init__(name=name)
        self.joint_index = joint_index
        self.amplitude = amplitude
        self.frequency = frequency
        self.time = 0.0

    def forward(self, dt: float):
        self.time += dt
        joint_position = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time)
        
        # We assume the joint_position applies only to the specified joint
        actions = ArticulationAction(joint_positions={self.joint_index: joint_position})
        return actions

    def reset(self):
        self.time = 0.0
