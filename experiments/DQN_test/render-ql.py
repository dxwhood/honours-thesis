import numpy as np
import gymnasium as gym

# load the model and render for 5 episodes
q_table = np.load('taxi_q_table.npy')
env = gym.make('Taxi-v3', render_mode='human')

#render 5 episodes
for episode in range(5):
    state = env.reset()[0]
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        env.render()

print('Done')


