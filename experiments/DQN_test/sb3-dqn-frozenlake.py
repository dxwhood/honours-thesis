from stable_baselines3 import DQN
import gymnasium as gym  # Using gymnasium instead of gym

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode='human')

# Initialize the model to None
model = None

# Try to load the model
try:
    model = DQN.load("dqn_frozenlake_1M.zip", custom_objects={"action_space": env.action_space})
    print("Model loaded successfully.")
    model.set_env(env)
except Exception as e:
    print(f"Couldn't load the model: {e}")

# Only proceed if the model could be loaded
if model:

    # 10 runs

    for i in range(10):

        obs = env.reset()
        obs = obs[0]  # Use only the actual observation, discard metadata
        print(f"Type of observation: {type(obs)}")
        print(f"Value of observation: {obs}")

        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()  # Convert NumPy array to native Python type
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

    env.close()
else:
    print("Model not loaded. Exiting.")
