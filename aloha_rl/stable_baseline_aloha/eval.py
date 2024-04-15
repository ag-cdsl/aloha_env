# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import carb
from env import JetBotEnv

try:
    from stable_baselines3 import PPO
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install stable-baselines3 in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install stable-baselines3 "
    )
    exit()


policy_path = "./cnn_policy/jetbot_policy.zip"

my_env = JetBotEnv(headless=False)
model = PPO.load(policy_path)

for _ in range(20):
    obs, _ = my_env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, truncated, info = my_env.step(actions)
        my_env.render()

my_env.close()
