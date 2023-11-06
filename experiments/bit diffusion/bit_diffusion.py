# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

# # Conversion functions
# def convert_box_to_grid(box_observation):
#     """
#     Converts a box observation space from a Mini Atari environment 
#     into a binary grid with multiple channels.
#     """
#     binary_grid = box_observation.astype(int)
#     return binary_grid

# def binbox2analog_bit(box_observation):
#     """
#     Converts the box observation into analog bits.
#     """
#     box_grid = convert_box_to_grid(box_observation)
#     analog_bits_grid = (box_grid * 2) - 1
#     return analog_bits_grid.reshape(10, 10, 4)

# # Custom environment wrapper
# class CustomObservationWrapper(gym.ObservationWrapper):
#     def observation(self, observation):
#         return binbox2analog_bit(observation)

# # Create and wrap the environment
# env = gym.make("MinAtar/Breakout-v1")
# wrapped_env = DummyVecEnv([lambda: CustomObservationWrapper(env)])

# # Initialize the model
# model = PPO("MlpPolicy", wrapped_env, verbose=1)

# # Train the model
# model.learn(total_timesteps=1e5) # Reduced for debugging

# # Save model
# model.save("ppo_minbreakout_analog_10k")

# # Evaluate the trained model
# obs = wrapped_env.reset()
# terminated = False
# truncated = False

# while not terminated and not truncated:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated = wrapped_env.step(action)
#     wrapped_env.render()

# wrapped_env.close()


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
    binary_grid = box_observation.astype(int)
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
    def observation(self, observation):
        # print("Original observation: ", observation)
        # print("Transformed observation: ", binbox2analog_bit(observation))
        # #types:
        # print("Original observation type: ", type(observation))
        # print("Transformed observation type: ", type(binbox2analog_bit(observation)))
        return binbox2analog_bit(observation)

# Create and wrap the environment
env = gym.make("MinAtar/Breakout-v1")
# wrapped_env = DummyVecEnv([lambda: CustomObservationWrapper(env)])
wrapped_env = CustomObservationWrapper(env)


# Initialize the model with an MLP policy
model = PPO("MlpPolicy", wrapped_env, verbose=1, learning_rate=0.01) # Adjusted learning rate

# Train the model
model.learn(total_timesteps=1e6) # Reduced for debugging

# Save model
model.save("ppo_minbreakout_analog_10k")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

wrapped_env.close()

