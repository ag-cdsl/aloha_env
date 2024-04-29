import torch
import numpy as np
from gym import spaces
from typing import Optional

from .base import AlohaTask

from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.articulations import ArticulationView


class ReachTask(AlohaTask):

    def __init__(self, 
                 name: str, 
                 n_envs: int = 1, 
                 offset: Optional[np.ndarray] = None):
        super().__init__(name, n_envs, offset)

        self.target_position = np.array([6,0,0]) # take as input later
        self.previous_distance = np.linalg.norm(self.target_position)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(16,), dtype=np.float32)  # Updated observation space


    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        
        for scene_id in range(self.num_envs):
            scene_prim_path = f"/World/scene_{scene_id}"

            # adding target location
            tloc = scene.add(
                VisualCuboid(
                    prim_path=f"{scene_prim_path}/target_location",
                    name=f"target_location_{scene_id}",
                    translation=self.target_position,
                    size=1,
                    color=np.array([0, 1.0, 0]),  # green color
                )
            )

        self.tlocs = ArticulationView(
            prim_paths_expr=f"/World/scene_*/target_location",
            name="tloc_view"
        )
    

    def reset(self, env_ids=None):
        self.robots.set_joint_positions(self.default_robot_joint_positions)
        
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        for i in range(self.num_envs):
            articulation = dc.get_articulation(f"/World/scene_{i}/aloha")
            root_body = dc.get_articulation_root_body(articulation)
            dc.wake_up_articulation(articulation)
            tf = _dynamic_control.Transform()
            tf.p = (0,3*i,0)
            dc.set_rigid_body_pose(root_body, tf)
  

    def get_observations(self) -> dict:
        """
        0-2: platform position
        3-6: platform orientation
        7-9: platform linear velocity
        10-12: platform angular velocity
        13-15: target location positions
        """
        robot_local_positions, robot_local_orientations = self.robots.get_local_poses()
        dof_linvels = self.robots.get_linear_velocities()
        dof_angvels = self.robots.get_angular_velocities()

        tloc_pos, tloc_quat = self.tlocs.get_local_poses()
        
        self.obs = torch.cat(
            [   
                robot_local_positions,
                robot_local_orientations,
                dof_linvels,
                dof_angvels,
                tloc_pos,
            ],
            axis=-1
        )
        return self.obs
    
    def calculate_metrics(self) -> dict:
        robot_position = self.obs[:, :3]
        tloc_pos = self.obs[:, 13:16]
        dist = np.linalg.norm(tloc_pos - robot_position)
        rewards = self.previous_distance - dist
        self.previous_distance = dist
        return torch.as_tensor(rewards)

 
 