from stable_baselines3 import PPO
import gymnasium as gym  # Using gymnasium instead of gym

print("STARTING 500K PPO TRAINING ON PONG")

# Initialize the environment
env = gym.make('ALE/Pong-v5')

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

# Train the model
model.learn(total_timesteps=500000)

# Save the model
model.save("ppo_pong_500K")