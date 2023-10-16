from stable_baselines3 import PPO
import gymnasium as gym  # Using gymnasium instead of gym
from stable_baselines3.common.callbacks import CheckpointCallback

print("STARTING 10K BREAKOUT TRAINING ON PONG")

# Initialize the environment
env = gym.make('ALE/Breakout-v5')

# Define hyperparameters
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1,
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=0.00025,
    clip_range=0.1
)

# Create the checkpoint callback
save_freq = 100000  # Save a checkpoint every 100K steps
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models/', name_prefix='rl_model')


# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_breakout_10K")