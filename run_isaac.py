import time

from aloha_rl.env import AlohaEnv


def main():
    env = AlohaEnv(
        headless=False
    )
    env.reset()
    
    i = 0
    while env._simulation_app.is_running():
        action = env.action_space.sample()
        obs, r, done, truncated, info = env.step(action)
        print(f"r: {r} | done: {done}")
        
        if i == 0:
            time.sleep(1)
        i += 1


if __name__ == "__main__":
    main()
