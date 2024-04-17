import math
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces

# import omni.isaac.core.tasks as tasks

ALOHA_ASSET_PATH = (
    Path.home()
    / ".local/share/ov/pkg/isaac_sim-2022.1.1/standalone_examples/aloha_env/aloha_rl/ALOHA.usd"
).as_posix()


class AlohaEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=3,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=512,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        from .task import AlohaTask
        self.task = AlohaTask("AlohaGenericTask")
        self._my_world.add_task(self.task)
        
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
        self.manip_controller = None
        
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gymnasium.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        self.max_velocity = 0.5
        self.max_angular_velocity = math.pi*0.5
        self.reset_counter = 0

    def get_dt(self):
        return self._dt

    def step(self, action):
        # action forward velocity , angular velocity on [-1, 1]
        raw_forward = action[0]
        raw_angular = action[1]

        # we want to force the jetbot to always drive forward
        # so we transform to [0,1].  we also scale by our max velocity
        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        # we scale the angular, but leave it on [-1,1] so the
        # jetbot can remain an ambiturner.
        angular_velocity = raw_angular * self.max_angular_velocity

        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            self.task.robot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            # self.manip_controller.apply_action(np.array([1, 1]))
            self._my_world.step(render=True)

        observations = self.task.get_observations()
        info = {}
        done = False
        truncated = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
            truncated = True
        reward, done = self.compute_reward_and_done()
        return observations, reward, done, truncated, info

    def compute_reward_and_done(self):
        target_loc_pos, _ = self.task.target_location.get_world_pose()
        cube_pos, _ = self.task.goal.get_world_pose()
        
        dist = np.linalg.norm(target_loc_pos - cube_pos)
        r = -dist
        done = dist <= 0.01
        return r, done
    
    def reset(self, seed=None):
        self._my_world.reset()
        # self.jetbot_controller.reset()
        self.manip_controller = self.task.robot.manipulator.get_articulation_controller()
        
        self.reset_counter = 0
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 1.50 * math.sqrt(np.random.rand()) + 0.20
        # self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05]))
        observations = self.task.get_observations()
        return observations, {}

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
