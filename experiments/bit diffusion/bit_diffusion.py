import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

# Conversion functions
def convert_box_to_grid(box_observation):
    """
    Converts a box observation space from a Mini Atari environment 
    into a binary grid with multiple channels.
    """

    binary_grid = box_observation.astype(np.int32)

    return binary_grid

def binbox2analog_bit(box_observation):
    """
    Converts the box observation into analog bits.
    """
    box_grid = convert_box_to_grid(box_observation)
    analog_bits_grid = (box_grid * 2) - 1
    #convert to floats for continous space
    analog_bits_grid = analog_bits_grid.astype(np.float32)
    return analog_bits_grid.reshape(10, 10, 4)

# Custom environment wrapper
class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming the original space is Box(False, True, (10, 10, 4), bool)
        orig_space = env.observation_space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=orig_space.shape, dtype=np.float32)

    def observation(self, observation):
        return binbox2analog_bit(observation)


# Create and wrap the environment
env = gym.make("MinAtar/Breakout-v1")
# wrapped_env = DummyVecEnv([lambda: CustomObservationWrapper(env)])
wrapped_env = CustomObservationWrapper(env)


# Initialize the model with an MLP policy
model = PPO("MlpPolicy", wrapped_env, verbose=1, learning_rate=0.0001, ent_coef=0.05) # Adjusted learning rate

# Train the model
model.learn(total_timesteps=500000) # Reduced for debugging

# Save model
model.save("ppo_minbreakout_analog_debug")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

wrapped_env.close()

