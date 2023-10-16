from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('ALE/Breakout-v5', render_mode='human')
print(f"Observation Space: {env.observation_space}")

try:
    model = PPO.load("ppo_breakout_4M")
except Exception as e:
    print(f"Error while loading: {e}")

model.set_env(env)

# Evaluate the trained model
obs = env.reset()
terminated = False
truncated = False

while not terminated and not truncated:
    try:
        # Assuming the first element in the tuple is the observation you need
        processed_obs = obs[0] if isinstance(obs, tuple) else obs

        # Debugging: Print processed observation's type and shape
        print(f"Processed Observation type: {type(processed_obs)}, Processed Observation shape: {processed_obs.shape if hasattr(processed_obs, 'shape') else 'N/A'}")
        
        action, _ = model.predict(processed_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

env.close()
