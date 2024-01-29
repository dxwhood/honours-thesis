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




def int2analog(x, n=10):
    # Convert an integer to a PyTorch tensor
    x_tensor = torch.tensor([x], dtype=torch.int32)

    # Convert integers into the corresponding binary bits.
    shifts = torch.arange(n - 1, -1, -1, dtype=x_tensor.dtype)
    x_tensor = torch.bitwise_right_shift(x_tensor, shifts)
    x_tensor = torch.remainder(x_tensor, 2)

    # Convert the binary bits into the corresponding analog values.
    x_tensor = x_tensor.type(torch.float32)
    x_tensor = 2 * x_tensor - 1


    return x_tensor  

def analog2int(x):
    # Convert an analog bit representation back to an integer
    x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
    x = torch.round(x).type(torch.int32)  # Round and convert to int
    # Convert binary bits back to integer
    int_val = 0
    for i, bit in enumerate(reversed(x)):
        int_val += bit.item() * (2 ** i)
    return int_val

def t_int2analog(x, n=10):
    """
    Convert a tensor of integers to their analog bit representations.

    Args:
    - x (torch.Tensor): Tensor of integers.
    - n (int): Number of bits for the representation.

    Returns:
    - torch.Tensor: Tensor of analog bit representations.
    """
    # Expand x to have a new dimension for the bits
    x_expanded = x.unsqueeze(-1)

    # Create shifts for each bit position and reshape for broadcasting
    shifts = torch.arange(n - 1, -1, -1, dtype=torch.int32, device=x.device)
    shifts = shifts.reshape(1, n)

    # Convert integers into binary bits using broadcasting
    x_bits = torch.bitwise_right_shift(x_expanded, shifts)
    x_bits = torch.remainder(x_bits, 2)

    # Convert binary bits into analog values
    x_analog = x_bits.type(torch.float32)
    x_analog = 2 * x_analog - 1

    return x_analog

def t_analog2int(x):
    """
    Convert a tensor of analog bit representations back to integers.

    Args:
    - x (torch.Tensor): Tensor of analog bit representations.

    Returns:
    - torch.Tensor: Tensor of integers.
    """
    # Convert from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = torch.round(x).type(torch.int32)

    # Prepare multipliers for each bit position
    n = x.shape[-1]  # Assuming the last dimension contains the bits
    multipliers = 2 ** torch.arange(n - 1, -1, -1, dtype=torch.int32, device=x.device)
    multipliers = multipliers.expand_as(x)

    # Convert binary bits back to integers
    int_vals = torch.sum(x * multipliers, dim=-1)

    return int_vals


def onehot(x, n):
    """
    Convert an integer to its one-hot encoded representation.

    Args:
    - x (int): Integer.
    - n (int): Size of one-hot encoding.

    Returns:
    - torch.Tensor: Tensor of one-hot encoded representation.
    """

    one_hot = torch.zeros(n)
    one_hot[x] = 1
    return one_hot


def t_onehot(x, n):
    """
    Convert a tensor of integers to their one-hot encoded representations.

    Args:
    - x (torch.Tensor): Tensor of integers.
    - n (int): Size of one-hot encoding.

    Returns:
    - torch.Tensor: Tensor of one-hot encoded representations.
    """
    # Ensure x is a 1D tensor
    if len(x.shape) > 1:
        raise ValueError("Input tensor must be 1-dimensional")

    # Create a zero tensor of shape [len(x), n]
    one_hot = torch.zeros((len(x), n), device=x.device)

    # Use torch.arange to create indices and torch.scatter_ to fill one-hot encodings
    indices = torch.arange(len(x), device=x.device)
    one_hot.scatter_(1, x.unsqueeze(1), 1)

    return one_hot

# def cosine_beta_schedule(T, s=0.1):
#     """
#     Generate a cosine beta schedule for T timesteps.

#     Args:
#     - T (int): The number of timesteps in the diffusion process.
#     - s (float): A hyperparameter controlling the sharpness of the cosine curve. Typical values are between 0.001 and 0.1.

#     Returns:
#     - betas (torch.Tensor): A tensor of beta values for the diffusion schedule.
#     """
#     # Define the steps
#     steps = torch.arange(0, T, dtype=torch.float32) / T

#     # Calculate the alphas using the cosine schedule
#     alphas = torch.cos(((steps + s) / (1 + s)) * np.pi * 0.5) ** 2
#     alphas = alphas / alphas[0]

#     # Calculate beta values from alphas
#     betas = 1 - alphas[1:] / alphas[:-1]
    
#     # Ensure that beta values are within a valid range
#     betas = torch.clip(betas, 0.0001, 0.9999)

#     if len(betas) < T:
#         last_beta = betas[-1]
#         betas = torch.cat([betas, last_beta.unsqueeze(0)])

#     return betas

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

def compute_alpha_bar(beta_schedule):
    alpha = 1. - beta_schedule
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha, alpha_bar


def t_apply_noise(x, timesteps, beta_schedule):
    """
    Applies noise to a batch of actions at specific timesteps.

    Args:
    - x (torch.Tensor): The initial actions tensor with shape [batch_size, action_dim].
    - timesteps (torch.Tensor): The specific timesteps for each action in the batch.
    - beta_schedule (torch.Tensor): The beta schedule tensor.

    Returns:
    - torch.Tensor: The noised version of the actions at the specified timesteps.
    """
    alpha, alpha_bar = compute_alpha_bar(beta_schedule)

    # Extract alpha_bar values for each timestep in the batch
    timesteps = timesteps.to(alpha_bar.device)
    alpha_bar_t = alpha_bar[timesteps]
    alpha_bar_t = alpha_bar_t.to(x.device)


    # Add noise to each action in the batch
    epsilon = torch.randn_like(x)


    if False:
        print()
        print("$"*50)
        print("In t_apply_noise: ")
        #print("x: ", x)
        print("x size: ", x.size())
        print("x shape: ", x.shape)
        #print("timesteps: ", timesteps)
        print("timesteps size: ", timesteps.size())
        print("timesteps shape: ", timesteps.shape)
        #print("beta_schedule: ", beta_schedule)
        print("beta_schedule size: ", beta_schedule.size())
        print("beta_schedule shape: ", beta_schedule.shape)
        #print("alpha: ", alpha)
        print("alpha size: ", alpha.size())
        print("alpha shape: ", alpha.shape)
        #print("alpha_bar: ", alpha_bar)
        print("alpha_bar size: ", alpha_bar.size())
        print("alpha_bar shape: ", alpha_bar.shape)
        #print("alpha_bar_t: ", alpha_bar_t)
        print("alpha_bar_t size: ", alpha_bar_t.size())
        print("alpha_bar_t shape: ", alpha_bar_t.shape)
        #print("epsilon: ", epsilon)
        print("epsilon size: ", epsilon.size())
        print("epsilon shape: ", epsilon.shape)
        print("$"*50)
        print()

    

    xt = torch.sqrt(alpha_bar_t.unsqueeze(1)) * x + torch.sqrt(1. - alpha_bar_t.unsqueeze(1)) * epsilon

    return xt

    
# def reverse_diffusion(x_noisy, timesteps, model, state, beta_schedule):
#     alpha, alpha_bar = compute_alpha_bar(beta_schedule)
#     # Extract alpha_bar values for each timestep in the batch
#     alpha_bar_t = alpha_bar[timesteps]
#     # Generate model output for the batch
#     model_output = model(x_noisy, timesteps, state).to(x_noisy.device)

#     alpha_bar_t = alpha_bar_t.to(x_noisy.device)
#     model_output = model_output.to(x_noisy.device)


#     x_denoised = (x_noisy - torch.sqrt(1. - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
#     return x_denoised

def reverse_diffusion(x_noisy, timestep, model, state, beta_schedule):
    """
    Performs a single reverse diffusion step.

    Args:
    - x_noisy (torch.Tensor): The current noisy version of the action data.
    - timestep (int): The current timestep in the reverse diffusion process.
    - model (nn.Module): The neural network model used for denoising.
    - state (torch.Tensor): The current state of the environment.
    - beta_schedule (torch.Tensor): The beta schedule tensor.

    Returns:
    - torch.Tensor: The denoised version of the action data after the reverse diffusion step.
    """

    
    alpha, alpha_bar = compute_alpha_bar(beta_schedule)

    # Convert timestep to a tensor
    timestep_tensor = torch.tensor([timestep], dtype=torch.float32, device=model.device)

    # Generate model output
    model_output = model(x_noisy, timestep_tensor, state)

    # everything on same device
    x_noisy = x_noisy.to(model.device)
    alpha_bar = alpha_bar.to(model.device)
    model_output = model_output.to(model.device)
    

    x_denoised = (x_noisy - torch.sqrt(1. - alpha_bar[timestep]) * model_output) / torch.sqrt(alpha_bar[timestep])
    return x_denoised


def t_reverse_diffusion(x_noisy, timesteps, model, states, beta_schedule):
    """
    Reverses the diffusion process for a batch of noised actions.

    Args:
    - x_noisy (torch.Tensor): The batch of noised action tensors.
    - timesteps (torch.Tensor): The specific timesteps at which to reverse diffusion (batched).
    - model (nn.Module): The neural network model used in the reverse diffusion process.
    - states (torch.Tensor): The states corresponding to each action in the batch.
    - beta_schedule (torch.Tensor): The beta schedule tensor.

    Returns:
    - torch.Tensor: The reverse diffused version of the actions at the specified timesteps.
    """


    alpha, alpha_bar = compute_alpha_bar(beta_schedule)
    # Ensure timesteps tensor is on the same device as alpha_bar
    alpha_bar = alpha_bar.to(model.device)

    #if timesteps not int (python int not dtype), to alphabar device
    if type(timesteps) == torch.Tensor:
        timesteps = timesteps.to(alpha_bar.device)
    

    # Extract alpha_bar values for each timestep in the batch
    alpha_bar_t = alpha_bar[timesteps]

    # Generate model output for the batch
    #make sure all is on the same device
    x_noisy = x_noisy.to(model.device)
    states = states.to(model.device)
    

    model_output = model(x_noisy, timesteps, states)

    if False:
        print()
        print("&"*50)
        print("In t_reverse_diffusion: ")
        #print("x_noisy: ", x_noisy)
        print("x_noisy size: ", x_noisy.size())
        print("x_noisy shape: ", x_noisy.shape)
        #print("timesteps: ", timesteps)
        print("timesteps size: ", timesteps.size())
        print("timesteps shape: ", timesteps.shape)
        #print("model: ", model)
        #print("model size: ", model.size())
        #print("model shape: ", model.shape)
        #print("states: ", states)
        print("states size: ", states.size())
        print("states shape: ", states.shape)
        #print("beta_schedule: ", beta_schedule)
        print("beta_schedule size: ", beta_schedule.size())
        print("beta_schedule shape: ", beta_schedule.shape)
        #print("alpha: ", alpha)
        print("alpha size: ", alpha.size())
        print("alpha shape: ", alpha.shape)
        #print("alpha_bar: ", alpha_bar)
        print("alpha_bar size: ", alpha_bar.size())
        print("alpha_bar shape: ", alpha_bar.shape)
        #print("alpha_bar_t: ", alpha_bar_t)
        print("alpha_bar_t size: ", alpha_bar_t.size())
        print("alpha_bar_t shape: ", alpha_bar_t.shape)
        print("model_output: ", model_output)
        print("model_output size: ", model_output.size())
        print("model_output shape: ", model_output.shape)
        print("&"*50)
        print()


    #x_denoised = (x_noisy - torch.sqrt(1. - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
    x_denoised = (x_noisy - torch.sqrt(1. - alpha_bar_t.unsqueeze(-1)) * model_output) / torch.sqrt(alpha_bar_t.unsqueeze(-1))

    return x_denoised



# Prepropocessing

def dataset2analog_taxi(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1, converters={4: lambda x: int(x == 'True')})

    #convert the data into analog bits
    analog_data = []
    for i in range(data.shape[0]):
        analog_state = []
        for j in range(data.shape[1]):
            #if action only use 3 bits (6 possible actions)
            if j == 1:
                analog_state.append(int2analog(data[i][j], 3))
            else:
                analog_state.append(int2analog(data[i][j], 9))
        analog_data.append(analog_state)

    #convert the reward and done back to int
    for i in range(data.shape[0]):
        analog_data[i][2] = torch.tensor(data[i][2], dtype=torch.float32)
        analog_data[i][4] = torch.tensor(data[i][4], dtype=torch.float32)
        
    return analog_data


def prepare_data_for_sampler(data):
    # Separate components
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for data_point in data:
        states.append(data_point[0])
        actions.append(data_point[1])
        rewards.append(data_point[2])
        next_states.append(data_point[3])
        dones.append(data_point[4])

    # Convert lists to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards).unsqueeze(1)  # Adding an extra dimension
    next_states = torch.stack(next_states)
    dones = torch.stack(dones).unsqueeze(1)

    # Prepare the data dictionary
    data_for_sampler = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones
    }

    return data_for_sampler
    

class Data_Sampler(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.state = (dataset['states']).type(torch.float32)
        self.action = (dataset['actions']).type(torch.float32)
        self.next_state = (dataset['next_states']).type(torch.float32)
        self.reward = (dataset['rewards']).type(torch.float32)
        self.done = (dataset['dones']).type(torch.float32)

        self.n = self.state.shape[0]  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps")
        print("Data Sampler Initiated with device ", self.device)
    

    def sample(self, batch_size):
       ind = np.random.randint(0, self.n, batch_size)
       return (
           	self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
            self.done[ind].to(self.device)
       )
    




