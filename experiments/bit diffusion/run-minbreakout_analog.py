import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('MinAtar/Breakout-v1')
try:
    model = PPO.load("ppo_minbreakout_analog_10k")
except Exception as e:
    print(f"Error while loading: {e}")

model.set_env(env)

#test model output space(make sure its not restricted to 0-1)


obs = env.reset()
terminated = False
truncated = False

#while not terminated and not truncated:
for i in range(100):
    try:
        # Assuming the first element in the tuple is the observation you need
        processed_obs = obs[0] if isinstance(obs, tuple) else obs

        # Debugging: Print processed observation's type and shape
        print(f"Processed Observation type: {type(processed_obs)}, Processed Observation shape: {processed_obs.shape if hasattr(processed_obs, 'shape') else 'N/A'}")
        
        action, _ = model.predict(processed_obs, deterministic=True)

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

