# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import typing

import numpy as np
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

try:
    from omni.usd.utils import get_world_transform_matrix
except:  # noqa E722
    from omni.usd import get_world_transform_matrix


class MPickPlaceController:
    """
    A simple pick and place state machine for tutorials

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).

    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller
        that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional):
        end effector initial picking height to start from (more info in phases above).
        If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional):
        Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        ur5_init_pose: list = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self.flag = False
        self._ur5_init_pose = ur5_init_pose
        self._trans_height = 0.8
        self.ppose = None
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        prim_trans_point=None,
        object_position=None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional):
            offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional):
            end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        # print(f"Now event: {self._event}")
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        elif self._event == 4:
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            # print("event 4 go go go")

        elif self._event in [4.5, 9.5]:
            # if self._event == 9.5:
            #     input()

            translation_trans_point = get_world_transform_matrix(prim_trans_point)
            trans_x = translation_trans_point[-1][0]
            trans_y = translation_trans_point[-1][1]
            # print(f"trans_x, trans_y: {trans_x, trans_y}\n")

            if self._event == 4.5:
                interpolated_xy = self._get_interpolated_xy(
                    trans_x, trans_y, self._current_target_x, self._current_target_y
                )
            else:
                interpolated_xy = self._get_interpolated_xy(
                    trans_x, trans_y, placing_position[0], placing_position[1]
                )

            trans_height = self._trans_height  # setting
            target_height = self._get_target_hs(trans_height)

            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if self._event == 9.5:
                print(f"position_target of event 9.5: {position_target}")
                print(f"trans position: {trans_x} {trans_y}\n")

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )
        elif self._event == 5:
            # input()

            translation_trans_point = get_world_transform_matrix(prim_trans_point)
            trans_x = translation_trans_point[-1][0]
            trans_y = translation_trans_point[-1][1]

            # bullshit: fix it!!!!
            # trans_x += 0.2
            # trans_y += 0.2

            #trans_x, trans_y, trans_z = object_position
            print(f'trans_pint_pos: {translation_trans_point}')
            print(f'placing_pos:{placing_position}')

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], trans_x, trans_y
            )
            print(f'interpolated_xy:{interpolated_xy}')
            # print(f"trans_xy: {trans_x, trans_y}")

            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            print(f'position_target:{position_target}')
            # print(f"x: {position_target[0]} y: {position_target[1]}, z: {position_target[2]}\n")
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            # print(f"target_joint_positions: {target_joint_positions}")

        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            # print(f"interpolated_xy: {interpolated_xy}\n")
            target_height = self._get_target_hs(placing_position[2])
            # print(f"target_height: {target_height}\n")
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            # print(f"position_target: {position_target}\n")
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )

        if self._event == 9.5:
            self._t += self._events_dt[-2]
        else:
            self._t += self._events_dt[int(self._event)]
        if self._t >= 1.0:
            if (self._event == 4) or (self._event == 4.5) or (self._event == 9) or (self._event == 9.5):
                self._event += 0.5
            else:
                self._event += 1
            self._t = 0
        # print(f"t = {self._t}\n")
        # print(f" self._events_dt: { self._events_dt}\n")
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        # print(f"alpha={alpha}")
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 4.5:
            return 0
        elif self._event in [4.5, 9.5]:
            return self._mix_sin(self._t)

        elif self._event == 5:
            return self._mix_sin(self._t)
            # mix_sin_ = self._mix_sin(self._t)
            # print(f"self._mix_sin(self._t): {mix_sin_}\n")
            # return 1
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 4.5:
            # h = self._h1
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._trans_height, a)
        elif self._event == 5:
            # h = self._h1
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._trans_height, self._h1, a)
            # h = self._trans_height
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        elif self._event == 9.5:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._trans_height, a)
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional):
            end effector initial picking height to start from.
            If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):
            Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return