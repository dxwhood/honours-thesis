import gymnasium as gym
from stable_baselines3 import PPO

name = "ppo_minbreakout_100k"


# Create the environment
env = gym.make('MinAtar/Breakout-v1')

# Try to load the model
try:
    model = PPO.load(name)
    # Set the environment only if the model is loaded successfully
    model.set_env(env)
except Exception as e:
    print(f"Error while loading: {e}")
    model = None

# Check if the model is loaded before trying to display the state
if model:
    try:
        # Assuming display_state is a valid method in MinAtar environments
        env.reset()
        env.game.display_state(50)
        for i in range(500):
            print(f"Step {i}")
            action, _ = model.predict(env.game.state(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.game.display_state(50)
            if terminated:
                break
    except Exception as e:
        print(f"Error while displaying state: {e}")
    
    try:
        # Assuming close_display is a valid method in MinAtar environments
        env.game.close_display()
    except Exception as e:
        print(f"Error while closing display: {e}")
else:
    print("Model not loaded, cannot display game state.")

# Close the environment properly
print("Made it :)")
env.close()

