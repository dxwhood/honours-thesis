import numpy as np
import gymnasium as gym
import pandas as pd



def q_learning(env, num_episodes=50000, learning_rate=0.85, discount_factor=0.95, exploration_prob=1.0, exploration_decay=0.995, min_exploration=0.05):
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

            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-learning update step
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

        # Decaying exploration rate
        exploration_prob = max(min_exploration, exploration_prob * exploration_decay)

        if episode % 100 == 0:
            print(f"Episode {episode} finished.")

    return q_table

env = gym.make('Taxi-v3')
q_table = q_learning(env)
print("Q-table finished!")

# evaluate agent performance
def evaluate_agent(q_table, env, num_episodes=100):
    total_reward = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = np.argmax(q_table[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state

    return total_reward / num_episodes

print("Average reward per episode: ", evaluate_agent(q_table, env))

#save the q-table
np.save('taxi_q_table.npy', q_table)





