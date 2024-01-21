import cv2
import gymnasium as gym
from stable_baselines3 import PPO

# Initialize environment and model
env = gym.make('MinAtar/SpaceInvaders-v1')

try:
    model = PPO.load("ppo_minSpaceInvaders-v1_3M")
except Exception as e:
    print(f"Error while loading: {e}")

model.set_env(env)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('model_run.mp4', fourcc, 30.0, (160, 210))

# Loop through the environment
obs = env.reset()
terminated = False
truncated = False

while not terminated and not truncated:
    try:
        processed_obs = obs[0] if isinstance(obs, tuple) else obs
        action, _ = model.predict(processed_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture the frame from the display
        frame = env.game.display_state(50)
        
        # Add frame to video writer
        video_writer.write(frame)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

# Release the video writer
video_writer.release()

# Close the environment display
env.game.close_display()
env.close()
