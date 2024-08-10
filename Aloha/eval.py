# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import carb
from tasks.env_gt_mapless_nav import AlphaBaseEnv

from stable_baselines3 import SAC
from omni.isaac.kit import SimulationApp

log_dir = "/isaac-sim/standalone_examples/base_aloha_env/Aloha/models/SAC_new"
policy_path = "/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/models/SAC/new_era_470000_steps"
env = AlphaBaseEnv()
model = SAC.load(policy_path,verbose=1,tensorboard_log=log_dir,)

for _ in range(100):
    obs, info = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

my_env.close()