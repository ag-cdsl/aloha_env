import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import carb
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import clip
import torchvision.transforms as T
from typing import Optional
from scipy.special import expit

config = {
    "renderer": "RayTracedLighting",
    "headless": True,
    #"headless": False,
    "multi_gpu": False, 
    #"active_gpu": gpu_to_use,
    "enable":"omni.kit.livestream.native"
}

def euler_from_quaternion(vec):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = vec[0], vec[1], vec[2], vec[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def get_quaternion_from_euler(roll,yaw=0, pitch=0):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])


class AlphaBaseEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=4,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1024,
        seed=10,
        headless=False,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp(config)
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from .wheeled_robot import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        jetbot_asset_path = "/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/assets/aloha/ALOHA_with_sensor_02.usd"
        
        create_prim(
                    prim_path=f"/room",
                    translation=(0, 0.22, 0),
                    usd_path="/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/assets/scenes/sber_kitchen/sber_kitchen_12_1.usd",
                )

        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel", "right_wheel"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([1.5, 0.2, 0.0]),
                orientation=get_quaternion_from_euler(np.pi/2),
            )
        )
        from pxr import PhysicsSchemaTools, UsdUtils, PhysxSchema, UsdPhysics
        from pxr import Usd
        from omni.physx import get_physx_simulation_interface
        import omni.usd
        self.my_stage = omni.usd.get_context().get_stage()
        self.my_prim = self.my_stage.GetPrimAtPath("/jetbot")

        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.my_prim)
        contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([10.0,0.0,0.0]),
                size=0.1,
                color=np.array([0, 1.0, 0]),
            )
        )
        self.helper = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/helper",
                name="visual_cube_help",
                position=np.array([10.0,0.0,0.0]),
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        )

        self.render_products = []
        from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
        from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
        import omni.replicator.core as rep
        self.image_resolution = 250
        self.camera_width = self.image_resolution
        self.camera_height = self.image_resolution
        camera_paths = "/jetbot/fl_link4/visuals/realsense/husky_rear_left"

        render_product = rep.create.render_product(camera_paths, resolution=(self.camera_width, self.camera_height))
        self.render_products.append(render_product)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = PytorchListener()
        self.pytorch_writer = rep.WriterRegistry.get("PytorchWriter")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device = ", self.device)
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device=self.device)
        self.pytorch_writer.attach(self.render_products)

        self.seed(seed)
        self.reward_range = (-10000, 10000)
        
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000000000, high=1000000000, shape=(1030,), dtype=np.float32)

        self.max_velocity = 1.5
        self.max_angular_velocity = math.pi*0.5
        self.event = 0

     
        convert_tensor = transforms.ToTensor()

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

        goal_path = '/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/img/goal.png'

        img_goal = clip_preprocess(Image.open(goal_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.img_goal_emb = self.clip_model.encode_image(img_goal)

        self.collision_step = 0
        self.collision = False

        return

    def _on_contact_report_event(self, contact_headers, contact_data):
        from pxr import PhysicsSchemaTools

        for contact_header in contact_headers:
            # instigator
            act0_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            # recipient
            act1_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            # the specific collision mesh that belongs to the Rigid Body
            cur_collider = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))

            # iterate over all contacts
            contact_data_offset = contact_header.contact_data_offset
            num_contact_data = contact_header.num_contact_data
            for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
                cur_contact = contact_data[index]

                # find the magnitude of the impulse
                cur_impulse =  cur_contact.impulse[0] * cur_contact.impulse[0]
                cur_impulse += cur_contact.impulse[1] * cur_contact.impulse[1]
                cur_impulse += cur_contact.impulse[2] * cur_contact.impulse[2]
                cur_impulse = math.sqrt(cur_impulse)

            if num_contact_data > 1: #1 is contect with flore
                self.collision = True

    def _get_dt(self):
        return self._dt

    def _is_collision(self):
        if self.collision:
            print("collision error!")
            self.collision = False
            return True 
        return False

    def _get_current_time(self):
        return self._my_world.current_time_step_index - self._steps_after_reset

    def _is_timeout(self):
        if self._get_current_time() >= self._max_episode_length:
            print("time out")
            return True
        return False


    def _get_gt_observations(self, previous_jetbot_position, previous_jetbot_orientation):
        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()

        if self.event == 0:
            goal_jetbot_orientation = get_quaternion_from_euler(np.pi)
            goal_world_position[0] = goal_world_position[0] - 0.9
        elif self.event == 1:
            goal_jetbot_orientation = get_quaternion_from_euler(0)
            goal_world_position[0] = goal_world_position[0] + 1.05
        else:
            goal_jetbot_orientation = get_quaternion_from_euler(np.pi/2)
            goal_world_position[1] = goal_world_position[1] - 0.9
        goal_world_position[2] = 0
        
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        motion_diff = np.linalg.norm(previous_jetbot_position - current_jetbot_position)
        rotation_diff = np.linalg.norm(previous_jetbot_orientation - current_jetbot_orientation)
        orientation_error = abs(euler_from_quaternion(current_jetbot_orientation)[0] - euler_from_quaternion(goal_jetbot_orientation)[0])

        observation = { 
            "goal_world_position": goal_world_position, 
            "current_jetbot_position": current_jetbot_position, 
            "current_jetbot_orientation":current_jetbot_orientation,
            "jetbot_linear_velocity": jetbot_linear_velocity,
            "jetbot_angular_velocity": jetbot_angular_velocity,
            "goal_jetbot_orientation": goal_jetbot_orientation,
            "goal_world_position": goal_world_position,
            "previous_dist_to_goal": previous_dist_to_goal,
            "current_dist_to_goal": current_dist_to_goal,
            "orientation_error": orientation_error,
        }
        print("observation is", observation)
        return observation

    def _get_terminated(self, observation):
        achievements = {
            "dist": False,
            "orient": False,
        }
        if observation["current_dist_to_goal"] < 0.2:
            achievements["dist"] = True
        if observation["orientation_error"] < 0.13:
            achievements["orient"] = True

        achieved = True
        for i in achievements:
            if not achievements[i]:
                achieved = False
                print("not achive yet ", i)

        return achieved, achievements

    def get_reward(self, obs):
        rewards = dict()
        # if (abs(obs["jetbot_linear_velocity"].any())>1):
        #     reward_v = -0.3*abs(max(obs["jetbot_linear_velocity"]))
        # else:
        #     reward_v = 0.5*abs(max(obs["jetbot_linear_velocity"]))
        
        # if (abs(obs["jetbot_angular_velocity"].any())>1):
        #     punish_w = -1.5*abs(max(obs["jetbot_angular_velocity"]))
        # else:
        #     punish_w = 0

        rewards["dir_to_goal"] = 0.5 if (obs["previous_dist_to_goal"] - obs["current_dist_to_goal"])>0 else 0
        rewards["dist_to_goal"] = 0.5/(1+obs["current_dist_to_goal"])
        rewards["orient_to_goal"] = 0.5/(1+abs(obs["orientation_error"]))
        
        print("rewards = ",rewards)
        achieved, achievements = self._get_terminated(obs)
        if achieved:
            terminated = True
            print("we made it")
            punish_vel = expit(abs(np.linalg.norm(obs["jetbot_linear_velocity"])) + abs(np.linalg.norm(obs["jetbot_angular_velocity"]))) - 0.5
            punish_time = 0.5*float(self._get_current_time())/float(self._max_episode_length)
            reward = 2 - punish_vel - punish_time
        else:
            terminated = False
            if not achievements["dist"]:
                reward = -1 + rewards["dist_to_goal"] + rewards["dir_to_goal"]
            else:
                reward = -0.3 + rewards["orient_to_goal"]

        return reward, terminated

    def move(self, action):
        raw_forward = action[0]
        raw_angular = action[1]

        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        angular_velocity = raw_angular * self.max_angular_velocity

        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)

        return

    def step(self, action):
        observations = self.get_observations()
        info = {}
        done = False
        truncated = False
        terminated = False

        previous_jetbot_position, previous_jetbot_orientation = self.jetbot.get_world_pose()
        self.move(action)   
        gt_observations = self._get_gt_observations(previous_jetbot_position, previous_jetbot_orientation)
        reward, terminated = self.get_reward(gt_observations)
        
        if not terminated:
            if self._is_timeout():
                truncated = True
                reward = reward - 1

            if self._is_collision():
                truncated = True
                reward = reward - 1

        print("general reward is", reward)

        
        return observations, reward, terminated, truncated, info


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._my_world.reset()
        #torch.cuda.empty_cache()
        info = {}
        self.event = np.random.randint(3)
        print("event = ", self.event)
        if self.event == 0:
            y =  3* np.random.rand() + 2.5
            x = 2.5
        elif self.event == 1:
            y =  3 * np.random.rand() + 2.5
            x = 0.4
        else:
            y = 7.1
            x = 1.5 + 0.2 * np.random.rand()

        self.goal.set_world_pose(np.array([x, y, 1]))

        if self.event == 0:
            x = x - 0.90
        elif self.event == 1:
            x = x + 1.05
        else:
            y = y - 0.90
        self.helper.set_world_pose(np.array([x, y, 0.05]))
        

        observations = self.get_observations()
        return observations, info

    def get_observations(self):
        self._my_world.render()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        #print("observ velocity", jetbot_linear_velocity, jetbot_angular_velocity)

        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            from torchvision.utils import save_image, make_grid
            img = images/255
            save_image(make_grid(img, nrows = 2), '/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/img/cartpole_export.png')
        else:
            print("Image tensor is NONE!")
        
        transform = T.ToPILImage()
        
        img_current = self.clip_preprocess(transform(img[0])).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_current_emb = self.clip_model.encode_image(img_current)

        return np.concatenate(
            [
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                self.img_goal_emb[0].cpu(),
                img_current_emb[0].cpu(),
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
