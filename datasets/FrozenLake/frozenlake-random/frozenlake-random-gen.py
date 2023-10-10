import gym
import pandas as pd

def collect_random_data(env_name, num_trajectories=1500, map_size="8x8"):
    env = gym.make(env_name, desc=None, map_name=map_size, is_slippery=True)
    dataset = []

    for _ in range(num_trajectories):
        state = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()  # random policy
            next_state, reward, done, _ = env.step(action)
            dataset.append([state, action, reward, next_state, done])
            state = next_state

    env.close()
    return dataset

# Collect random trajectories for 8x8 FrozenLake
frozenlake_random_data = collect_random_data('FrozenLake-v1')

# Convert to DataFrame and save as CSV
df = pd.DataFrame(frozenlake_random_data, columns=['state', 'action', 'reward', 'next_state', 'done'])
df.to_csv('frozenlake8x8_random_dataset.csv', index=False)
