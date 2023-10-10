import numpy as np
import gym
import pandas as pd


def q_learning(env, num_episodes=250000, learning_rate=0.8, discount_factor=0.999, exploration_prob=1.0, exploration_decay=0.9997, min_exploration=0.01, learning_rate_decay=0.9995, min_learning_rate=0.1):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print("Episode: ", episode)
        state = env.reset()
        done = False
        
        while not done:
            if np.random.uniform(0, 1) < exploration_prob:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, _ = env.step(action)

            # Reward reshaping
            if done and reward == 0:
                reward = -5

            # Q-learning update step
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

        # Decay exploration rate and learning rate
        exploration_prob = max(min_exploration, exploration_prob * exploration_decay)
        learning_rate = max(min_learning_rate, learning_rate * learning_rate_decay)

    return q_table

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
q_table_expert = q_learning(env)

def collect_q_data(q_table, env, num_trajectories=5000):
    dataset = []

    for _ in range(num_trajectories):
        state = env.reset()
        done = False

        while not done:
            action = np.argmax(q_table[state, :])
            next_state, reward, done, _ = env.step(action)
            dataset.append([state, action, reward, next_state, done])
            state = next_state

    return dataset

frozenlake_expert_data = collect_q_data(q_table_expert, env)

# Convert to DataFrame and save as CSV (using pandas)
df = pd.DataFrame(frozenlake_expert_data, columns=['state', 'action', 'reward', 'next_state', 'done'])
df.to_csv('frozenlake8x8_expert_dataset3.csv', index=False)
