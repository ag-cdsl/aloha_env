# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import carb
from tasks.env_obst import AlphaBaseEnv

from stable_baselines3 import SAC
from omni.isaac.kit import SimulationApp

policy_path = "/isaac-sim/standalone_examples/base_aloha_env/Aloha/models/SAC/test7_1100000_steps.zip"
my_env = AlphaBaseEnv()
model = SAC.load(policy_path)

for _ in range(20):
    obs = my_env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = my_env.step(actions)
        my_env.render()

my_env.close()
