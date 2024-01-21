from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np


print("MINATAR SPACE INVADERS PPO 1M")

# Initialize the environment
env = gym.make('MinAtar/SpaceInvaders-v1')
env = TransformObservation(env, lambda obs: obs.astype(np.uint8))
env = TransformObservation(env, lambda obs: obs * 255.0)
env = DummyVecEnv([lambda: env])

# Define hyperparameters
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=0.00025,
    clip_range=0.1
)

# Define the checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=1000000, 
    save_path='./checkpoints/', 
    name_prefix='ppo_minspaceinvaders'
)

# Train the model
model.learn(total_timesteps=int(3e6), callback=checkpoint_callback)

model.save("ppo_minSpaceInvaders-v1_3M")

env.close()