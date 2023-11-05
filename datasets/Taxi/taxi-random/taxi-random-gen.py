import gymnasium as gym  
import pandas as pd

def collect_random_data(env_name, num_trajectories=1500):
    env = gym.make(env_name)
    dataset = []

    for _ in range(num_trajectories):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while True:  
            action = env.action_space.sample()  # Random policy
            next_state, reward, terminated, truncated, _ = env.step(action)

            dataset.append([state, action, reward, next_state, terminated, truncated])  
            
            if terminated or truncated:  
                break

            state = next_state

    env.close()
    return dataset

# collect random trajectories
taxi_random_data = collect_random_data('Taxi-v3')

# save as csv
df = pd.DataFrame(taxi_random_data, columns=['state', 'action', 'reward', 'next_state', 'terminated', 'truncated'])
df.to_csv('taxi_random_dataset_fixed.csv', index=False)
