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

from helpers import *

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        #if int, convert to tensor
        if isinstance(x, int): x = torch.tensor([x], dtype=torch.float32)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class MLP(nn.Module):
    def __init__(self, encoded_state_dim, encoded_action_dim, time_embedding_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.encoded_state_dim = encoded_state_dim
        self.encoded_action_dim = encoded_action_dim
        self.time_embedding_dim = time_embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if torch.backends.mps.is_available(): self.device = torch.device("mps")

        # Time Embedding Layer

        self.time_mlp = nn.Sequential(
        SinusoidalPosEmb(time_embedding_dim),
        nn.Linear(time_embedding_dim, time_embedding_dim * 2),
        nn.Mish(),
        nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )

        # Input dimension: state dimension + bit-encoded action dimension + time embedding dimension
        input_dim = encoded_state_dim + encoded_action_dim + time_embedding_dim

        # Define the MLP layers
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()  # Output dimension is the bit-encoded action dimension
        )
        
        self.final_layer = nn.Linear(hidden_dim, encoded_action_dim)

    def forward(self, x, time, state):
        # Generate time embedding
        #if time isnt int, to device
        if not isinstance(time, int): time = time.to(self.device)
        time_emb = self.time_mlp(time)

        # Reshape time_emb to match the batch size of state and bit_action
        time_emb = time_emb.to(x.device)

        #print devices
        # print("state device: ", state.device)
        # print("time_emb device: ", time_emb.device)
        # print("encoded_action device: ", encoded_action.device)


        # Concatenate state, time embedding, and bit-encoded action

        #print shapes   
        if False:
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("x shape: ", x.shape)
            print("time_emb shape: ", time_emb.shape)
            print("state shape: ", state.shape)
            print("x device: ", x.device)
            print("time_emb device: ", time_emb.device)
            print("state device: ", state.device)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


        #unsqueeze if x is 1d tensor
        if len(x.shape) == 1: x = x.unsqueeze(0)
        if len(state.shape) == 1: state = state.unsqueeze(0)


        x = torch.cat([x, time_emb, state], dim=-1)
        x = x.to(self.device)
        # Pass through the MLP layers
        x = self.mid_layer(x)

        x = self.final_layer(x)

        return x


class BitDiffusion(nn.Module):
    def __init__(self, action_dim, model, T=50, predict_epsilon=True):
        super(BitDiffusion, self).__init__()
        self.action_dim = action_dim
        self.model = model
        #self.beta_schedule = cosine_beta_schedule(T)
        self.beta_schedule = linear_beta_schedule(T)
        self.predict_epsilon = predict_epsilon
        self.T = T
        #get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps")

        print("BitDiffusion Model Initiated with device ", self.device)

    #def forward_diffusion(self, x, t):
    #    return apply_noise(x, t, self.beta_schedule)
        
    def t_forward_diffusion(self, x, t, beta_schedule):
        """
        Applies forward diffusion to a batch of actions x at specific timesteps.

        Args:
        - x (torch.Tensor): The initial batch of action tensors.
        - t (torch.Tensor): The specific timesteps at which to apply diffusion (batched).
        - beta_schedule (torch.Tensor): The beta schedule tensor.

        Returns:
        - torch.Tensor: The diffused version of the actions at the specified timesteps.
        """
        if False:
            print()
            print("@"*50)
            print("In t_forward_diffusion: ")
            #print("x: ", x)
            print("x size: ", x.size())
            print("x shape: ", x.shape)
            #print("t: ", t)
            print("t size: ", t.size())
            print("t shape: ", t.shape)
            #print("beta_schedule: ", beta_schedule)
            print("beta_schedule size: ", beta_schedule.size())
            print("beta_schedule shape: ", beta_schedule.shape)
            print("@"*50)
            print()

        return t_apply_noise(x, t, beta_schedule)

    
    def reverse_diffusion(self, x_noisy, t, state):
        return t_reverse_diffusion(x_noisy, t, self.model, state, self.beta_schedule)
    
    def sample(self, state):
        # Sample an action from the model given a state
        #torch randn 1d tensors of size log2(action_dim) (action in bits)
        x = torch.randn(math.ceil(math.log2(self.action_dim)), device=self.device)
        for t in reversed(range(0, self.beta_schedule.size(0))):
            x = self.reverse_diffusion(x, t, state)
        return x
    
    def batched_sample(self, state):
        # Assume state is a batched tensor with shape [batch_size, state_dim]
        batch_size = state.size(0)

        # Initialize a batched tensor for x
        # Make sure x has a new first dimension for the batch size
        x = torch.randn(batch_size, math.ceil(math.log2(self.action_dim)), device=self.device)

        # Iterate over timesteps
        for t_step in reversed(range(0, self.beta_schedule.size(0))):
            # Create a tensor t filled with the current timestep for each batch element
            t = torch.full((batch_size,), t_step, dtype=torch.long, device=self.device)
            
            # Pass the batched x, t, and state to reverse_diffusion
            x = self.reverse_diffusion(x, t, state)

        return x


    def loss(self, x_start, state):
            """
            Calculates the loss for a batch of data.

            Args:
            - x_start (torch.Tensor): The original data (actions).
            - state (torch.Tensor): The states of the environment.
            - timesteps (torch.Tensor): The timesteps at which to apply and reverse the diffusion.
            - predict_epsilon (bool): If True, the model predicts noise; otherwise, it predicts the original data.

            Returns:
            - torch.Tensor: The calculated loss for the batch.
            """

            batch_size = x_start.size(0)
            # Randomly sample timesteps for each item in the batch
            timesteps = torch.randint(0, self.T, (batch_size,), device=x_start.device)
            

            noise = torch.randn_like(x_start)
            x_noisy = self.t_forward_diffusion(x_start, timesteps, self.beta_schedule)
            
            #if np.random.uniform() < 0.001:
                #make it random noise
                #x_recon = torch.randn_like(x_start)
                #add grad
                #x_recon.requires_grad = True

                #print("Random noise!")
            #else:    
                #x_recon = self.reverse_diffusion(x_noisy, timesteps, state)

            x_recon = self.reverse_diffusion(x_noisy, timesteps, state)


            if self.predict_epsilon:
                target = noise  # Model tries to predict the noise
            else:
                #1% chance:
                target = x_start  # Model tries to reconstruct the original data

            # Loss function, e.g., Mean Squared Error between the target and the reconstructed data
            x_recon = x_recon.to(self.device)
            target = target.to(self.device)
            loss = F.mse_loss(x_recon, target)
            if False:
                print()
                print("#"*50)
                print("In loss function: ")
                #print("timesteps: ", timesteps)
                print("timesteps size: ", timesteps.size())
                print("timesteps shape: ", timesteps.shape)
                #print("x_start: ", x_start)
                print("x_start size: ", x_start.size())
                print("x_start shape: ", x_start.shape)
                #print("state: ", state)
                print("state size: ", state.size())
                print("state shape: ", state.shape)
                #print("noise: ", noise)
                print("noise size: ", noise.size())
                print("noise shape: ", noise.shape)
                print("x_noisy: ", x_noisy)
                print("x_noisy size: ", x_noisy.size())
                print("x_noisy shape: ", x_noisy.shape)
                print("x_recon: ", x_recon)
                print("x_recon size: ", x_recon.size())
                print("x_recon shape: ", x_recon.shape)
                print("target: ", target)
                print("target size: ", target.size())
                print("target shape: ", target.shape)
                print("loss: ", loss)
                print("loss size: ", loss.size())
                print("loss shape: ", loss.shape)
                print("#"*50)
                print()


            return loss
    
    def forward(self, state):
        return self.batched_sample(state)

    def forward(self, x_noisy, t, state):
        return self.model.forward(x_noisy, t, state)
