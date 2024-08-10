import gym
from gym import spaces
import numpy as np
import math
import carb

# для удаленного просмотра
config = {
    "renderer": "RayTracedLighting",
    "headless": True,
    #headless: False,
    "multi_gpu": False, 
    #"active_gpu": gpu_to_use,
    "enable":"omni.kit.livestream.native"
}

class AlphaBaseEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=4,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=4096,
        seed=3,
        headless=False,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        #self.headless = headless
        #self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._simulation_app = SimulationApp(config)
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        # from omni.isaac.wheeled_robots.robots import WheeledRobot
        from .wheeled_robot import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        #jetbot_asset_path = "/isaac-sim/standalone_examples/base_aloha_env/Aloha/assets/aloha/ALOHA.usd"
        jetbot_asset_path = "/isaac-sim/standalone_examples/base_aloha_env/Aloha/assets/aloha/ALOHA_with_sensor_02.usd"
        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel", "right_wheel"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.3, wheel_base=0.5)
        
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([6.0,0.0,0.0]),
                size=0.5,
                color=np.array([0, 1.0, 0]),
            )
        )
        self.obstacle_1 = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_2",
                name="obstacle_1",
                position=np.array([4.0,0.0,0.0]),
                size=1,
                color=np.array([1, 0, 0]),
            )
        )
        self.obstacle_2 = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_3",
                name="obstacle_2",
                position=np.array([4.0,0.0,0.0]),
                size=1,
                color=np.array([1, 0, 0]),
            )
        )
        self.obstacle_3 = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_4",
                name="obstacle_3",
                position=np.array([4.0,0.0,0.0]),
                size=1,
                color=np.array([1, 0, 0]),
            )
        )
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(25,), dtype=np.float32)

        self.max_velocity = 2
        self.max_angular_velocity = math.pi*0.4
        self.reset_counter = 0
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.jetbot.get_world_pose()
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
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)



        observations = self.get_observations()
        info = {}
        done = False
        truncated = True
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
            truncated = True
        goal_world_position, _ = self.goal.get_world_pose()
        obstacle_1_world_position, _ = self.obstacle_1.get_world_pose()
        obstacle_2_world_position, _ = self.obstacle_2.get_world_pose()
        obstacle_3_world_position, _ = self.obstacle_3.get_world_pose()
        current_jetbot_position, _ = self.jetbot.get_world_pose()

        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)

        current_dist_to_obst_1 = np.linalg.norm(current_jetbot_position - obstacle_1_world_position)
        current_dist_to_obst_2 = np.linalg.norm(current_jetbot_position - obstacle_2_world_position)
        current_dist_to_obst_3 = np.linalg.norm(current_jetbot_position - obstacle_3_world_position)

        punish_obst = - 2/(1+current_dist_to_obst_1) - 2/(1+current_dist_to_obst_2) - 2/(1+current_dist_to_obst_3)

        dt = self._my_world.current_time_step_index
        print("dt")
        print(dt)
        print("dist")
        print(current_dist_to_goal)
        print(current_dist_to_obst_1)
        print(current_dist_to_obst_2)
        print(current_dist_to_obst_3)
        print("vel")
        print(forward_velocity)
        print(angular_velocity)

        if (abs(forward_velocity)>0.7):
            reward_v = -0.3*abs(forward_velocity)
        else:
            reward_v = 0.5*abs(forward_velocity)#less

        if (abs(angular_velocity)>0.8):
            punish_w = -1.5*abs(angular_velocity)
        else:
            punish_w = 0

        print("data")
        print()
        reward_dir = (previous_dist_to_goal - current_dist_to_goal)*2500
        print(reward_dir)
        reward_goal = (5/(0.5+current_dist_to_goal)-1.5)*4
        print(reward_goal)
        reward = reward_goal + reward_dir + punish_obst + reward_v + punish_w
        print(reward)
        if (np.linalg.norm(current_jetbot_position - obstacle_1_world_position) <= 1.1):
            reward = -1000
            done = True
        if (np.linalg.norm(current_jetbot_position - obstacle_2_world_position) <= 1.1):
            reward = -1000
            done = True
        if (np.linalg.norm(current_jetbot_position - obstacle_3_world_position) <= 1.1):
            reward = -1000
            done = True


        if dt >= 4096:
            reward = reward - 500  

        if current_dist_to_goal < 0.5:
            reward = 3000 - float(dt)/3
            done = True

        print(reward)
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        self.reset_counter = 0
        # Generate a random angle alpha between 0 and 2*pi
        alpha = 2 * math.pi * np.random.rand()
        # Set the radius of the circle to 2.5 units
        radius = 4
        # Calculate the x and y coordinates based on the radius and angle
        x = radius * math.cos(alpha)
        y = radius * math.sin(alpha)
        # Set the goal's position based on these coordinates
        self.goal.set_world_pose(np.array([x, y, 0.05]))

        alpha = 2 * math.pi * np.random.rand()
        # Set the radius of the circle to 2.5 units
        radius = 1
        # Calculate the x and y coordinates based on the radius and angle
        x = radius * math.cos(alpha)
        y = radius * math.sin(alpha)
        # Set the goal's position based on these coordinates
        self.obstacle_1.set_world_pose(np.array([x, y, 0.05]))

        alpha = 2 * math.pi * np.random.rand()
        # Set the radius of the circle to 2.5 units
        radius = 1.5
        # Calculate the x and y coordinates based on the radius and angle
        x = radius * math.cos(alpha)
        y = radius * math.sin(alpha)
        # Set the goal's position based on these coordinates
        self.obstacle_1.set_world_pose(np.array([x, y, 0.05]))

        alpha = 2 * math.pi * np.random.rand()
        # Set the radius of the circle to 2.5 units
        radius = 2
        # Calculate the x and y coordinates based on the radius and angle
        x = radius * math.cos(alpha)
        y = radius * math.sin(alpha)
        # Set the goal's position based on these coordinates
        self.obstacle_1.set_world_pose(np.array([x, y, 0.05]))


        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        goal_world_position, _ = self.goal.get_world_pose()
        obstacle_1_world_position, _ = self.obstacle_1.get_world_pose()
        obstacle_2_world_position, _ = self.obstacle_2.get_world_pose()
        obstacle_3_world_position, _ = self.obstacle_3.get_world_pose()
        return np.concatenate(
            [
                jetbot_world_position,
                jetbot_world_orientation,
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                goal_world_position,
                obstacle_1_world_position,
                obstacle_2_world_position,
                obstacle_3_world_position,
            ]
        )

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
