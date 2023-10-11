import gym
import numpy as np
import time
import os
import cv2

# Initialize the environment
env = gym.make('Taxi-v3', render_mode='human')
max_timesteps = 100  # Limit to truncate an episode

frames = []  # For storing rendered frames

for episode in range(1, 3):  # Running 10 episodes as an example
    
    state = env.reset()
    terminated = False
    truncated = False
    
    t = 0
    while not terminated and not truncated:
        action = env.action_space.sample()  # Random action
        #print("STEP: ", env.step(action))

        next_state, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        frame = env.render()
        frames.append(frame)

    if terminated:
        print(f"Episode {episode} terminated naturally.")
    elif truncated:
        print(f"Episode {episode} was truncated after {t} timesteps.")


