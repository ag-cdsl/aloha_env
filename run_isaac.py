import time

import numpy as np
import torch

from main_impl.env import AlohaEnv

from omni.isaac.gym.vec_env import VecEnvBase


def make_aloha_env(n_envs, headless=False):
    env = VecEnvBase(headless=headless)
    from main_impl.task import AlohaTask
    task = AlohaTask(name="Aloha", n_envs=n_envs)
    env.set_task(task, backend="torch")
    return env


def traj_gen(a0, step, reverse=False):
    a = a0
    it = range(6)
    if reverse:
        it = reversed(it)
    for i in it:
        s = a[i] + 0.15
        while a[i] < 1:
            yield a
            a[i] += step
        while a[i] > s:
            yield a
            a[i] -= step
        # while a[i] < s:
        #     yield a
        #     a[i] += step
    while True:
        yield a


def main():
    env = make_aloha_env(2)

    i = 0
    obs = env.reset()
    
    gripper_cmd = 0.0
    gripper_cmd_step = 0.03
    
    arm_1_gen = traj_gen(obs["observation"][0, 15:21], 0.03)
    arm_2_gen = traj_gen(obs["observation"][0, 29:35], 0.03, reverse=True)
    
    while env._simulation_app.is_running():
        actions = torch.from_numpy(np.stack([
            env.action_space.sample()
            for _ in range(env._task.num_envs)
        ], axis=0))
        
        # wheels
        actions[:, 0:2] = 0
        if i > 100 and i < 250:
            actions[:, 0:2] = 0.7
        
        # gripper1
        actions[:, 2] = gripper_cmd
        
        # gripper2
        actions[:, 9] = -gripper_cmd
        
        # arm1
        arm1_cmd = next(arm_1_gen)
        actions[:, 3:9] = arm1_cmd
        actions[1, 3:9] = -arm1_cmd
        
        # arm2
        arm2_cmd = next(arm_2_gen)
        actions[:, 10:16] = arm2_cmd
        
        gripper_cmd = max(-1, min(1, gripper_cmd + gripper_cmd_step))
        if i and i % 200 == 0:
            gripper_cmd_step *= -1
        
        # print(i, f"arm_cmd: {arm1_cmd}, {arm2_cmd}")
        
        obs, r, done, info = env.step(actions)
        position = obs["observation"][0][:3]
        print(i, f"position {position}", f"shape observation {obs['observation'].shape}")
        if i == 0:
            time.sleep(1)
        i += 1
        
        if i % 2000 == 0:
            obs = env.reset()


if __name__ == "__main__":
    main()
