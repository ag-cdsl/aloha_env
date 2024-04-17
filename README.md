### Aloha env


See `run_isaac.py` for example usage.


Action components:

- 0-1: wheels' velocities
- 2: gripper_1 control (continuous, 1 to open, -1 to close)
- 3-8: 6 arm_1 joint position refs
- 9: gripper_2 control
- 10-15: 6 arm_2 joint position refs


Observation components:

- 0-2: platform position
- 3-6: platform orientation
- 7-9: platform linear velocity
- 10-12: platform angular velocity
- 13-14: gripper_1 joint positions
- 15-20: arm_1 joint positions
- 21-26: arm_1 joint velocities
- 27-28: gripper_2 joint positions
- 29-34: arm_2 joint positions
- 35-40: arm_2 joint velocities
- 41-43: cube position
- 44-47: cube orientation
- 48-50: target location position
