import gym
import pandas as pd

def collect_random_data(env_name, num_trajectories=1500):
    env = gym.make(env_name)
    dataset = []

    for _ in range(num_trajectories):
        state = env.reset()
        
        while True:
            action = env.action_space.sample()  # random policy
            next_state, reward, done, _ = env.step(action)
            dataset.append([state, action, reward, next_state, done])
            state = next_state

            if done:
                break

    env.close()
    return dataset

# Collect random trajectories
taxi_random_data = collect_random_data('Taxi-v3')

# Convert to DataFrame and save as CSV
df = pd.DataFrame(taxi_random_data, columns=['state', 'action', 'reward', 'next_state', 'done'])
df.to_csv('taxi_random_dataset.csv', index=False)
