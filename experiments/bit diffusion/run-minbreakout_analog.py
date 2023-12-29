import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

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

env = gym.make('MinAtar/Breakout-v1')

env = CustomObservationWrapper(env)

try:
    model = PPO.load("ppo_minbreakout_analog_debug")
except Exception as e:
    print(f"Error while loading: {e}")

model.set_env(env)



obs = env.reset()
terminated = False
truncated = False

#while not terminated and not truncated:
for i in range(100):
    try:
        # Assuming the first element in the tuple is the observation you need
        obs = obs[0] if isinstance(obs, tuple) else obs

        # Debugging: Print processed observation's type and shape
        print(f"Observation: {obs}")
        
        #processed_obs = binbox2analog_bit(obs)

        #print(f"Processed observation: {processed_obs}")

        action, _ = model.predict(obs, deterministic=True)
        

        print(f"Action: {action}")
       
        obs, reward, terminated, truncated, info = env.step(action)
        env.game.display_state(50)
        if terminated or truncated:
            print(f"Episode terminated with reward {reward}")
            obs = env.reset()

    except Exception as e:
        print(f"Error during prediction: {e}")
        break
env.game.close_display()
env.close()

