import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import math
import time
import seaborn as sns

from d2ql import D2QL, Critic
from bitDiffusion import *
from helpers import *


# Test of D2QL on the Taxi gym environment
# Dataset: taxi_q_expert_dataset.csv, collected from an expert QL agent

# Load the dataset
data = dataset2analog_taxi("taxi_q_expert_dataset.csv")


state_dim = 500 # Dimension of the state vector
encoded_state_dim = 9  # Bit Dimension of the state vector
action_dim = 6 # Bit Dimension of the action vector
encoded_action_dim = 3    # Bit Dimension of the bit-encoded action vector
hidden_dim = 64  # Number of neurons in each hidden layer
time_embedding_dim = 8  # Dimension of the time embedding

# Data Sampler
data_for_sampler = prepare_data_for_sampler(data)
data_sampler = Data_Sampler(data_for_sampler)
batch_size = 100

'''
state, action, reward, next_state, done = data_sampler.sample(batch_size)
print("State: ", state)
print("Action: ", action)
print("Reward: ", reward)
print("Next State: ", next_state)
print("Done: ", done)

print("\n\n NEXT SECTION: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

print("Action: ", action)
print(tensor_analog2int(action))
print(tensor_int2analog(tensor_analog2int(action), 3))

print("\n One hot test:")
print(onehot(analog2int(action[0]), 6))
print(tensor_onehot(tensor_analog2int(action), 6))
'''

# Create the D2QL agent
agent = D2QL(state_dim, action_dim, time_embedding_dim, hidden_dim)

# Train the agent
# TODO: Train the agent on the dataset
save = True
epochs = 2
training_step = 0
for e in range(epochs):
    print(f"Epoch {e+1} | Training step {training_step}:")
    metrics = agent.train(data_sampler, iterations=1000, batch_size=100)
    
    #print individual losses(ql, bc, actor, critic)
    print("QL Loss: ", metrics['ql_loss'])
    print("BC Loss: ", metrics['bc_loss'])
    print("Actor Loss: ", metrics['actor_loss'])
    print("Critic Loss: ", metrics['critic_loss'])
    print()

    # Update the training step
    training_step += 1000

# Save the agent
if save:
    agent.save("d2ql_taxi_agent_test")


# Evaluate the agent
# TODO: Evaluate the agent on the dataset

