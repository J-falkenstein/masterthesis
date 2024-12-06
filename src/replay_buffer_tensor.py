# replay buffer class very simple with tensor
import random
import torch

class ReplayBufferTensor:
    def __init__(
        self,
        buffer_size: int,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.device = device
        
        # TODO could put all data in one tensor and then sample indices, make faster on GPU
        # Initialize tensors to store data
        self.observations = torch.empty((buffer_size, 1), dtype=torch.float32, device = self.device)
        self.next_observations = torch.empty((buffer_size,1 ), dtype=torch.float32, device = self.device)
        self.actions = torch.empty((buffer_size, action_dim), dtype=torch.float32, device = self.device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float32, device = self.device)
    
    def add(self, obs, next_obs, action, reward):
        if self.buffer_counter >= self.buffer_size:
            print("Buffer full, overwriting oldest data")
            self.buffer_counter = 0
        
        # Append new data
        self.observations[self.buffer_counter] = obs.unsqueeze(0)
        self.next_observations[self.buffer_counter] = next_obs.unsqueeze(0)
        self.actions[self.buffer_counter] = action.unsqueeze(0)
        self.rewards[self.buffer_counter] = reward.unsqueeze(0)
        
        self.buffer_counter += 1
    
    def sample(self, batch_size):

        # TODO this self.buffer counter might be a problem if the buffer is overwritten and the counter is below batch size
        # currently with buffer size 1000000 this bug makes the algorithm learn, 
        # It somehow makes use of very small batches and the most recent data
        random_indices = torch.randperm(self.buffer_counter)[:batch_size]
        sampled_obs = self.observations[random_indices]
        sampled_next_obs = self.next_observations[random_indices]
        sampled_actions = self.actions[random_indices]
        sampled_rewards = self.rewards[random_indices]
        
        # Return sampled data
        return SampledData(
            sampled_obs.detach(),
            sampled_next_obs.detach(),
            sampled_actions.detach(),
            sampled_rewards.detach(),
        )

from dataclasses import dataclass

@dataclass
class SampledData:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor