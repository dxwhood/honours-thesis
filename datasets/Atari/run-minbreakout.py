import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('MinAtar/Breakout-v1')
try:
    model = PPO.load("ppo_minbreakout_100k")
except Exception as e:
    print(f"Error while loading: {e}")

model.set_env(env)

env.game.display_state(50)
# train, do steps, ...
env.game.close_display()