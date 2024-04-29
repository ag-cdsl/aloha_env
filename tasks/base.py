import math
import torch
import numpy as np
from gym import spaces
from typing import Optional

from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

from utils.json_parser import asset_path


class AlohaTask(BaseTask):
    """ Defines a task space containing Aloha Robot and it's configurations (action space, DOF),
        Any class inheriting this class would have to implement their own observation space and reward mechanism.
        REFERENCE: https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_new_rl_example.html 
    """
    def __init__(self, 
        name: str,
        n_envs: int = 1,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        self.num_envs = n_envs
        self.env_spacing = 1.5
        
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(51,), dtype=np.float32)
        
        # wheels
        self._wheel_dof_names = ["left_wheel", "right_wheel"]
        self._num_wheel_dof = len(self._wheel_dof_names)
        self._wheel_dof_indices: list[int]
        self.max_velocity = 0.5
        self.max_angular_velocity = math.pi * 0.5
        self.aloha_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
        
        # gripper_1
        self._gripper1_dof_names = ["fl_joint7", "fl_joint8" ]
        self._num_gripper1_dof = len(self._gripper1_dof_names)
        self._gripper1_dof_indices: list[int]
        
        # gripper_2
        self._gripper2_dof_names = ["fr_joint7", "fr_joint8" ]
        self._num_gripper2_dof = len(self._gripper2_dof_names)
        self._gripper2_dof_indices: list[int]
        
        n_arm_dofs = 6
        
        # arm_1
        self._arm1_dof_names = [f"fl_joint{i}" for i in range(1, n_arm_dofs + 1)]
        self._num_arm1_dof = len(self._arm1_dof_names)
        self._arm1_dof_indices: list[int]
        
        # arm_2
        self._arm2_dof_names = [f"fr_joint{i}" for i in range(1, n_arm_dofs + 1)]
        self._num_arm2_dof = len(self._arm2_dof_names)
        self._arm2_dof_indices: list[int]
        
        BaseTask.__init__(self, name=name, offset=offset)


    def set_up_scene(self, scene):
        """Setup the scene with only the robot as default behavior."""
        # Implement robot setup logic here. This part can be overridden by derived classes to customize the scene.
        table_height = 0.7
        cube_size = 0.05
        self.cube_default_translation = np.array([1.5, -0.2, table_height + cube_size / 2])
        
        for scene_id in range(self.num_envs):
            scene_prim_path = f"/World/scene_{scene_id}"
            create_prim(
                prim_path=scene_prim_path,
                position=(0, 0 + 3 * scene_id, 0)
            )
            
            # adding robot
            create_prim(
                prim_path=f"{scene_prim_path}/aloha",
                translation= np.array([0,0,0]),
                usd_path=asset_path()
            )


        self.robots = ArticulationView(
            prim_paths_expr=f"/World/scene_*/aloha",
            name="aloha_view"
        )
        scene.add_default_ground_plane()
        scene.add(self.robots)
   
    
    def post_reset(self) -> None:
        self._wheel_dof_indices = [
            self.robots.get_dof_index(self._wheel_dof_names[i]) for i in range(self._num_wheel_dof)
        ]
        self._gripper1_dof_indices = [
            self.robots.get_dof_index(self._gripper1_dof_names[i]) for i in range(self._num_gripper1_dof)
        ]
        self._gripper2_dof_indices = [
            self.robots.get_dof_index(self._gripper2_dof_names[i]) for i in range(self._num_gripper2_dof)
        ]
        self._arm1_dof_indices = [
            self.robots.get_dof_index(self._arm1_dof_names[i]) for i in range(self._num_arm1_dof)
        ]
        self._arm2_dof_indices = [
            self.robots.get_dof_index(self._arm2_dof_names[i]) for i in range(self._num_arm2_dof)
        ]
        self.default_robot_joint_positions = self.robots.get_joint_positions()


    def pre_physics_step(self, actions):
        """Apply actions to the environment."""
        
        actions = torch.as_tensor(actions, dtype=torch.float32)
        # Ensure actions are at least 2-dimensional
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)  # Add batch dimension if it's not present


        # control wheel pairs
        # -------------------
        wheel_vels = actions[:, :2]

        raw_forward = wheel_vels[:, 0]   
        raw_angular = wheel_vels[:, 1]

        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        angular_velocity = raw_angular * self.max_angular_velocity

        command = np.array([forward_velocity.numpy()[0], angular_velocity.numpy()[0]])
        action_vels = self.aloha_controller.forward(command=command)
        action = ArticulationAction(joint_velocities=torch.from_numpy(action_vels.joint_velocities).float(), joint_indices=torch.Tensor(self._wheel_dof_indices))
        self.robots.apply_action(action)

        
        # control gripper1
        # -------------------
        cmd = actions[:, 2] # gripper: 1 to open, -1 to close
        cmd = torch.clip(cmd, min=-1.0, max=1.0) / 50
        jpos = torch.stack([cmd, cmd], axis=-1)
        self.robots.set_joint_position_targets(jpos, joint_indices=self._gripper1_dof_indices)
        
        
        # control arm1
        # -------------------
        jvels = actions[:, 3:9]
        amp = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32) * 1.3
        jvels = torch.clip(jvels, min=-1.0, max=1.0) * amp
        self.robots.set_joint_position_targets(jvels, joint_indices=self._arm1_dof_indices)

        # control gripper2
        # -------------------
        cmd = actions[:, 9] # gripper: 1 to open, -1 to close
        cmd = torch.clip(cmd, min=-1.0, max=1.0) / 50
        jpos = torch.stack([cmd, cmd], axis=-1)
        self.robots.set_joint_position_targets(jpos, joint_indices=self._gripper2_dof_indices)
        
        # control arm2
        # -------------------
        jvels = actions[:, 10:16]
        amp = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)
        jvels = torch.clip(jvels, min=-1.0, max=1.0) * amp * 1.3
        self.robots.set_joint_position_targets(jvels, joint_indices=self._arm2_dof_indices)


    def reset(self, env_ids=None):
        """Reset task environments. This method can be overridden in derived classes."""
        raise NotImplementedError
    
    
    def is_done(self) -> bool:
        dones = torch.tensor([False] * self.num_envs, dtype=bool)
        return dones
