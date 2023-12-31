import numpy as np
import gymnasium as gym  # Assuming gymnasium provides 'terminated' and 'truncated'
import pandas as pd

def q_learning(env, num_episodes=50000, learning_rate=0.8, discount_factor=0.95, exploration_prob=1.0, exploration_decay=0.995, min_exploration=0.1):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if np.random.uniform(0, 1) < exploration_prob:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            next_state, reward, terminated, truncated, _ = env.step(action)  # Assuming env.step returns terminated and truncated

            # Q-learning update step
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

        # Decaying exploration rate
        exploration_prob = max(min_exploration, exploration_prob * exploration_decay)

    return q_table

env = gym.make('Taxi-v3')
q_table = q_learning(env)
print("Q-table finished!")

def collect_q_data(q_table, env, num_trajectories=5000):
    dataset = []

    for _ in range(num_trajectories):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = np.argmax(q_table[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)  # Assuming env.step returns terminated and truncated
            dataset.append([state, action, reward, next_state, terminated, truncated])  # Adding terminated and truncated to the dataset
            state = next_state

    return dataset

taxi_expert_data = collect_q_data(q_table, env)

# Convert to DataFrame and save as CSV (using pandas)
df = pd.DataFrame(taxi_expert_data, columns=['state', 'action', 'reward', 'next_state', 'terminated', 'truncated'])  # Added 'terminated' and 'truncated'
df.to_csv('taxi_q_expert_dataset2.csv', index=False)
