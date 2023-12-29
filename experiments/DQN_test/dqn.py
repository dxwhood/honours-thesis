import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Function to one-hot encode states
def one_hot_state(state, n_states):
    vec = np.zeros(n_states)
    vec[state] = 1
    return vec

# Load the model
model = torch.load('dqn-model1.pt', map_location=torch.device('cpu'))
print(model)

# Set to eval mode for inference
model.eval()

# Create environment
env = gym.make('Taxi-v3', render_mode='human')
n_states = env.observation_space.n  # Adjust based on your environment

#10 taxi runs

for i in range(10):

    # Initialize state
    state = env.reset()[0]
    state = one_hot_state(state, n_states)
    terminated = False
    truncated = False

    # Loop until done
    while not terminated or truncated:
        env.render()  # Display the environment
        with torch.no_grad():
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = one_hot_state(next_state, n_states)
        state = next_state

env.close()
