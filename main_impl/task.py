import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces

from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import VisualCuboid, DynamicCuboid, FixedCuboid

from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
from omni.isaac.core.articulations import ArticulationView


ALOHA_ASSET_PATH = (
    Path.home()
    / ".local/share/ov/pkg/isaac_sim-2022.1.1/standalone_examples/aloha_env/aloha_rl/ALOHA.usd"
).as_posix()


class AlohaTask(BaseTask):
    def __init__(self, 
        name: str,
        n_envs: int = 1,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        self.num_envs = n_envs
        self.env_spacing = 1.5
        
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)
        
        # wheels
        self._wheel_dof_names = ["left_wheel", "right_wheel"]
        self._num_wheel_dof = len(self._wheel_dof_names)
        self._wheel_dof_indices: list[int]
        self.max_velocity = 0.5
        self.max_angular_velocity = math.pi * 0.5
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        self.wheels_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
        
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

    def set_up_scene(self, scene: Scene) -> None:
        self.all_cubes = []
        
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
                translation=(0,0,0),
                usd_path=ALOHA_ASSET_PATH
            )
            
            # adding table
            table = scene.add(
                FixedCuboid(
                    prim_path=f"{scene_prim_path}/table",
                    name=f"table_{scene_id}",
                    translation=np.array([1.5, 0.0, table_height / 2]),
                    size=table_height,
                    color=np.array([0, 0, 1.0]),
                )
            )
            
            # adding cube
            cube = scene.add(
                DynamicCuboid(
                    prim_path=f"{scene_prim_path}/cube",
                    name=f"visual_cube_{scene_id}",
                    translation=self.cube_default_translation,
                    size=cube_size,
                    color=np.array([1.0, 0, 0]),
                )
            )
            self.all_cubes.append(cube)
            
            # adding target location
            tloc = scene.add(
                VisualCuboid(
                    prim_path=f"{scene_prim_path}/target_location",
                    name=f"target_location_{scene_id}",
                    translation=np.array([1.5, 0.2, table_height+0.1]),
                    size=0.2,
                    color=np.array([0, 1.0, 0]),
                )
            )
        
        self.robots = ArticulationView(
            prim_paths_expr=f"/World/scene_*/aloha",
            name="aloha_view"
        )
        self.cubes = ArticulationView(
            prim_paths_expr=f"/World/scene_*/cube",
            name="cube_view"
        )
        self.tables = ArticulationView(
            prim_paths_expr=f"/World/scene_*/table",
            name="table_view"
        )
        self.tlocs = ArticulationView(
            prim_paths_expr=f"/World/scene_*/target_location",
            name="tloc_view"
        )
        
        scene.add_default_ground_plane()
        scene.add(self.robots)
    
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
            
            t = self.cube_default_translation.copy()
            t[1] += 3 * i
            self.all_cubes[i].set_local_pose(translation=t)
        
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
    
    def get_observations(self) -> dict:
        """
        0-2: platform position
        3-6: platform orientation
        7-9: platform linear velocity
        10-12: platform angular velocity
        13-14: gripper_1 joint positions
        15-20: arm_1 joint positions
        21-26: arm_1 joint velocities
        27-28: gripper_2 joint positions
        29-34: arm_2 joint positions
        35-40: arm_2 joint velocities
        41-43: cube positions
        44-47: cube orientations
        48-50: target location positions
        """
        robot_local_positions, robot_local_orientations = self.robots.get_local_poses()
        dof_linvels = self.robots.get_linear_velocities()
        dof_angvels = self.robots.get_angular_velocities()
        
        grip_1_jpos = self.robots.get_joint_positions(joint_indices=self._gripper1_dof_indices)
        grip_2_jpos = self.robots.get_joint_positions(joint_indices=self._gripper2_dof_indices)
        arm_1_jpos = self.robots.get_joint_positions(joint_indices=self._arm1_dof_indices)
        arm_1_jvel = self.robots.get_joint_velocities(joint_indices=self._arm1_dof_indices)
        arm_2_jpos = self.robots.get_joint_positions(joint_indices=self._arm2_dof_indices)
        arm_2_jvel = self.robots.get_joint_velocities(joint_indices=self._arm2_dof_indices)
        
        cube_pos, cube_quat = self.cubes.get_local_poses()
        tloc_pos, tloc_quat = self.tlocs.get_local_poses()
        
        self.obs = torch.cat(
            [   
                robot_local_positions,
                robot_local_orientations,
                dof_linvels,
                dof_angvels,
                grip_1_jpos,
                arm_1_jpos,
                arm_1_jvel,
                grip_2_jpos,
                arm_2_jpos,
                arm_2_jvel,
                cube_pos,
                cube_quat,
                tloc_pos,
            ],
            axis=-1
        )
        return self.obs
    
    def calculate_metrics(self) -> dict:
        tloc_pos = self.obs[:, 48:51]
        cube_pos = self.obs[:, 41:44]
        dist = np.linalg.norm(tloc_pos - cube_pos)
        rewards = -dist
        return torch.as_tensor(rewards)

    def is_done(self) -> bool:
        dones = torch.tensor([False] * self.num_envs, dtype=bool)
        return dones

    def pre_physics_step(self, actions):
        """
        0-1: wheel velocities
        2: gripper_1 control (1 to open, 0 to close)
        3-8: 6 arm_1 joint position refs
        9: gripper_2 control
        10-15: 6 arm_2 joint position refs
        """
        actions = torch.as_tensor(actions, dtype=torch.float32)
        # control wheel pairs
        # -------------------
        
        # # forward velocity, angular velocity on [-1, 1]
        # raw_forward = actions[:, 0]
        # raw_angular = actions[:, 1]

        # # to always drive forward we transform to [0,1]. we also scale by our max velocity
        # forward = (raw_forward + 1.0) / 2.0
        # forward_velocity = forward * self.max_velocity

        # # we scale the angular, but leave it on [-1,1] so the robot can remain an ambiturner
        # angular_velocity = raw_angular * self.max_angular_velocity
        
        # wheel_vels = np.stack([
        #     self.wheels_controller.forward(command=[v, w]).joint_velocities
        #     for v, w in zip(forward_velocity, angular_velocity)
        # ])
        wheel_vels = actions[:, :2]
        wheel_vels = torch.as_tensor(wheel_vels).to(torch.float32)
        wheel_vels = torch.clip(wheel_vels, min=-1.0, max=1.0) * 3
        
        self.robots.set_joint_velocities(wheel_vels, joint_indices=self._wheel_dof_indices)
        
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
