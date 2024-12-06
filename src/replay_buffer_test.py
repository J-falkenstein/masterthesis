# replay buffer class very simple with tensor
import torch
import gc

class ReplayBufferTest:
    def __init__(
        self,
        buffer_size: int,
        env_name: str,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.device = device
        self.buffer_full = False
        self.env_name = env_name
        
        # TODO could put all data in one tensor and then sample indices, make faster on GPU
        obsShape = 3 if env_name == "BlottoWithInclompeteMultiDim" else 1
        self.observations = torch.empty((buffer_size, obsShape), dtype=torch.float32, device = self.device)
        self.actions = torch.empty((buffer_size, action_dim), dtype=torch.float32, device = self.device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float32, device = self.device)

    # I am not clearing the full buffer anymore if I add new data to the beginning as the sweep showed it also works without this bug/feature
    # this should implement a simple circular buffer that overwrites if necessary but never deletes old data
    def add(self, obs, action, reward):
        batch_size = obs.shape[0]

        if self.buffer_counter + batch_size >= self.buffer_size:
            self.buffer_full = True

        indices = torch.arange(self.buffer_counter, self.buffer_counter + batch_size) % self.buffer_size

        self.observations[indices] = obs[:batch_size].detach().to(self.device)
        self.actions[indices] = action[:batch_size].detach().to(self.device)
        self.rewards[indices] = reward[:batch_size].detach().to(self.device)

        self.buffer_counter = (self.buffer_counter + batch_size) % self.buffer_size

    
    def sample(self, batch_size):

        # if the buffer is full we can sample from the whole buffer, while we add data to the begining of it <- this only works because we do not have real trajoctories of env steps, otherwise these must be deleted together
        if self.buffer_full:
            random_indices = torch.randperm(self.buffer_size)[:batch_size]
        else:
            random_indices = torch.randperm(self.buffer_counter)[:batch_size]
        sampled_obs = self.observations[random_indices]
        sampled_actions = self.actions[random_indices]
        sampled_rewards = self.rewards[random_indices]
        
        return SampledData(
            sampled_obs,
            sampled_actions,
            sampled_rewards,
        )

from dataclasses import dataclass

@dataclass
class SampledData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor