from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np





#NAME
name = "PPO_minatar_Breakout-v1_10M"

print(name)

# Initialize the environment
env = gym.make('MinAtar/Breakout-v1')

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
    save_freq=int((5e6)), 
    save_path='./checkpoints/', 
    name_prefix=name
)

# Train the model
model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)

model.save(name)

env.close()