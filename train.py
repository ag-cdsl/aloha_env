import time

import numpy as np
import torch
from omni.isaac.gym.vec_env import VecEnvBase
from stable_baselines3 import PPO

env = VecEnvBase(headless=False)
from tasks.reach import ReachTask
task = ReachTask(name="Aloha", n_envs=1)
env.set_task(task, backend="torch")



# Assuming env is already imported and set up
model = PPO(
    "MlpPolicy",
    env,
    n_steps=500,  # Reduced from 1000 to make updates more responsive
    batch_size=500,  # Reduced for faster but more frequent updates
    n_epochs=10,  # Reduced to use fresh data more frequently
    learning_rate=0.002,  # Increased to speed up convergence
    gamma=0.99,  # Keeps a good balance between short-term and long-term rewards
    device="cuda:0",
    ent_coef=0.01,  # Slightly higher to encourage exploration
    vf_coef=0.5,
    max_grad_norm=0.5,  # Lower max grad norm to help stabilize training with higher learning rate
    verbose=1,
    tensorboard_log="./standalone_examples/aloha-tdmpc/logs/"
)


model.learn(total_timesteps=50000)
model.save("./models/logs")

env.close()