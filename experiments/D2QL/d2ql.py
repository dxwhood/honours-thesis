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
import copy

from helpers import *
from bitDiffusion import *





class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1_model(x), self.q2_model(x)
    



class D2QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount=0.99,
                 eta = 1.0,
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_emas_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 gamma=0.99,
                 tau=0.005,
                ):  

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoded_state_dim = 9
        self.encoded_action_dim = 3
        self.max_action = max_action

        self.eta = eta
        self.n_timesteps = n_timesteps
        self.ema_decay = ema_decay
        self.step_start_ema = step_start_ema
        self.update_emas_every = update_emas_every
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.discount = discount
        self.lr_decay = lr_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps")

        self.actor_mlp = MLP(self.encoded_state_dim, self.encoded_action_dim, self.n_timesteps).to(self.device)
        self.actor = BitDiffusion(self.action_dim, self.actor_mlp, self.n_timesteps).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        




    def train(self, data_sampler, iterations=1000, batch_size=100): # can add replay buffer
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for i in range(iterations):
            #sample batch from dataset
            state, action, reward, next_state, done = data_sampler.sample(batch_size)

            # Q-value Learning
            current_q1, current_q2 = self.critic(t_onehot(t_analog2int(state), self.state_dim), t_onehot(t_analog2int(action), self.action_dim)) # check onehot/analog dims here its a batch
            next_action = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(t_onehot(t_analog2int(next_state), self.state_dim), t_onehot(t_analog2int(next_action), self.action_dim))
            target_q = torch.min(target_q1, target_q2)

            target_q = (t_analog2int(reward) + (1-done) * self.discount * target_q).detach()
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Policy Learning
            bc_loss = self.actor.loss(action, state)
            next_action = self.actor(state)

            q1_next_action, q2_next_action = self.critic(t_onehot(t_analog2int(state), self.state_dim), t_onehot(t_analog2int(next_action), self.action_dim))
            if np.random.uniform() > 0.5:
                q_loss = - q1_next_action.mean() / q2_next_action.abs().mean().detach()
            else:
                q_loss = - q2_next_action.mean() / q1_next_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target Network Update (not using ema for now)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step +=1

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        return metric
    
    

    def sample_action(self, state):
        """
        Generate a bit-encoded action for a given state using the trained model.

        Args:
        - model (BitDiffusion): The trained diffusion model.
        - state (torch.Tensor): The state tensor.

        Returns:
        - torch.Tensor: The generated bit-encoded action.
        """
        with torch.no_grad():
            # Prepare the state tensor (ensure it's the right shape)
            state = state.to(self.device).unsqueeze(0)  # Add batch dimension if necessary

            # Generate a sample action
            T = self.actor.beta_schedule.size(0)  # Total number of timesteps
            sample_shape = (1, self.actor.bit_dim)
            sampled_action = self.actor.sample(state, sample_shape).squeeze(0)  # Remove batch dimension

            return analog2int(sampled_action)
