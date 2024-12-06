# python file to create a custom gym environment modeling a 2 player blotto game
import gymnasium as gym
from git import Object
import numpy as np
import torch 



# two bidders with a budget of 1, the highest bid wins and pays the amount of the bid
# the value of the item for each bidder is dependet on the state and for each round and bidder drawn randomly between 0 and 1
class AllPayAuctionStates(gym.Env): 
    def __init__(self, num_agents, device):
        self.num_agents = num_agents
        self.action_space = []
        self.observation_space = []
        self.state = []
        self.device = device
        for _ in range(self.num_agents):
            self.action_space.append(gym.spaces.Box(low=0, high=1, dtype=np.float32))
            # self.observation_space.append(gym.spaces.Discrete(1))
            self.observation_space.append(gym.spaces.Box(low=0, high=1, dtype=np.float32))

        # random selection of valuations in the beginning 
        # Performs uniform sampling 
        self.state.append(torch.tensor(self.observation_space[0].sample(), device=self.device))
        self.state.append(torch.tensor(self.observation_space[1].sample(), device=self.device))

        # TODO allocate space for state and observation space on the device to avoid memory allocation during step and reset
   

   # not needed anymore only get reward 
    def step(self, action_n):
        obs_n = []
        reward_n = []
        truncations_n = []
        terminations_n = []
        info_n = []
        
        if action_n[0][0] > action_n[1][0]:
            reward_n.append(self.state[0]-action_n[0])
            reward_n.append(-action_n[1])
        else:
            reward_n.append(-action_n[0])
            reward_n.append(self.state[1]-action_n[1])


        # after step randomly uddate the state
        self.state[0] = torch.tensor(self.observation_space[0].sample(), device=self.device)
        self.state[1] = torch.tensor(self.observation_space[1].sample(), device=self.device)
        obs_n.append(self.state[0])
        obs_n.append(self.state[1])
        
        info_n.append([{}])
        info_n.append([{}])
        return obs_n, reward_n, truncations_n, terminations_n, info_n

    def reset(self):
        self.state[0] = torch.tensor(self.observation_space[0].sample(), device=self.device)
        self.state[1] = torch.tensor(self.observation_space[1].sample(), device=self.device)
        obs_n = []
        obs_n.append(self.state[0])
        obs_n.append(self.state[1])
        return obs_n
    
    def get_rewards(self, action_n, obs_n):
        # Convert actions to tensors and move to device
        actions_0 = action_n[0]
        actions_1 = action_n[1]

        # Convert observations to tensors and move to device
        # TODO find out if needed to convert maybe already is on device
        obs_0 = obs_n[0]
        obs_1 = obs_n[1]

        # Calculate rewards using vectorized operations
        rewards_0 = torch.where(actions_0 > actions_1, obs_0 - actions_0, -actions_0)
        rewards_1 = torch.where(actions_1 >= actions_0, obs_1 - actions_1, -actions_1)

        # Stack rewards for both agents
        return [rewards_0, rewards_1]
    
    def calculate_l2_loss(self, test_actions, validation_obs):
        return torch.sqrt(pow(abs(test_actions - self.get_equilibrium_action(validation_obs)), 2).mean())
    
    def calculate_utility_loss_currentStrategies(self, action_n, obs_n):
        equilibrium_action_n = [self.get_equilibrium_action(obs) for obs in obs_n]
        reward_equilibrium_n = []
        for i, eq_action in enumerate(equilibrium_action_n):
            eq_actions = action_n.copy()
            eq_actions[i] = eq_action
            reward_equilibrium = self.get_rewards(eq_actions, obs_n)[i]
            reward_equilibrium_n.append(reward_equilibrium)
        reward_own_n = []
        for i, _ in enumerate(action_n):
            reward_own = self.get_rewards(action_n, obs_n)[i]
            reward_own_n.append(reward_own)
        utility_loss_n = []
        for i in range(len(obs_n)):
            utility_loss_n.append(1-(torch.sum(reward_own_n[i])/torch.sum(reward_equilibrium_n[i])))
        return utility_loss_n
    
    def calculate_utility_loss(self, action_n, obs_n):
        equilibrium_action_n = [self.get_equilibrium_action(obs) for obs in obs_n]
        reward_equilibrium_n = []
        for i in range(len(obs_n)):
            reward_equilibrium = self.get_rewards(equilibrium_action_n, obs_n)[i]
            reward_equilibrium_n.append(reward_equilibrium)
        reward_own_n = []
        for i, action_i in enumerate(action_n):
            eq_actions = equilibrium_action_n.copy()
            eq_actions[i] = action_i
            reward_own = self.get_rewards(eq_actions, obs_n)[i]
            reward_own_n.append(reward_own)
        utility_loss_n = []
        for i in range(len(obs_n)):
            utility_loss_n.append(1-(torch.sum(reward_own_n[i])/torch.sum(reward_equilibrium_n[i])))
        return utility_loss_n

    def get_equilibrium_action(self, obs): 
        return 0.5*pow(obs, 2)

    def sample_obs(self, nr_of_samples, agent_id):
        low = int(self.observation_space[agent_id].low)
        high = int(self.observation_space[agent_id].high)
        # TODO does not work wiht multi-dim observations
        samples = torch.rand((nr_of_samples,1), device=self.device)
        samples = low + samples * (high - low)
        return samples

class FirstPricedAuctionStates(Object): 
    def __init__(self, num_agents, device):
        self.num_agents = num_agents
        self.action_space = []
        self.observation_space = []
        self.state = []
        self.device = device
        for _ in range(self.num_agents):
            self.action_space.append(gym.spaces.Box(low=0, high=1, dtype=np.float32))
            # self.action_space_low = 0
            # self.action_space_high = 1
            self.observation_space.append(gym.spaces.Box(low=0, high=1, dtype=np.float32))
            # self.observation_space_low = 0
            # self.observation_space_high = 1

        # sample uniformly between 0 and 1
        #self.state = torch.rand(2, dtype=torch.float32)

        # random selection of valuations in the beginning 
        self.state.append(torch.tensor(self.observation_space[0].sample(), device=self.device))
        self.state.append(torch.tensor(self.observation_space[1].sample(), device=self.device))

        # ME: State as a tensor instead of a list. This is also true for the step function and the other mechanisms. 
        # ME: Gym could also be an inefficiency here. 
   
    def step(self, action_n):
        obs_n = []
        reward_n = []
        truncations_n = []
        terminations_n = []
        info_n = []
        
        # TODO if i have for each agent potentially multiple actions, I also have to compare multiple times, as I basically want to play multiple games 
        # the following code must be adjusted
        if action_n[0][0] > action_n[1][0]:
            reward_n.append(self.state[0]-action_n[0])
            reward_n.append(torch.tensor([0]))
        else:
            reward_n.append(torch.tensor([0]))
            reward_n.append(self.state[1]-action_n[1])


        # after step randomly uddate the state
        # TODO this is not needed anymore 
        self.state[0] = torch.tensor(self.observation_space[0].sample(), device=self.device)
        self.state[1] = torch.tensor(self.observation_space[1].sample(), device=self.device)
        # # self.state = torch.rand(2, dtype=torch.float32) 

        obs_n.append(self.state[0])
        obs_n.append(self.state[1])
        
        info_n.append([{}])
        info_n.append([{}])
        return obs_n, reward_n, truncations_n, terminations_n, info_n

    def reset(self):
        self.state[0] = torch.tensor(self.observation_space[0].sample(), device=self.device)
        self.state[1] = torch.tensor(self.observation_space[1].sample(), device=self.device)
        # self.state = torch.rand(2, dtype=torch.float32)
        obs_n = []
        obs_n.append(self.state[0])
        obs_n.append(self.state[1])
        return obs_n
    
    def sample_obs(self, nr_of_samples, agent_id):
        low = int(self.observation_space[agent_id].low)
        high = int(self.observation_space[agent_id].high)
        samples = torch.rand((nr_of_samples,1), device=self.device)
        samples = low + samples * (high - low)
        return samples
       
    def get_rewards(self, action_n, obs_n):
    # Convert actions to tensors and move to device
        actions_0 = action_n[0]
        actions_1 = action_n[1]

        # Convert observations to tensors and move to device
        obs_0 = obs_n[0]
        obs_1 = obs_n[1]

        # Calculate rewards using vectorized operations
        rewards_0 = torch.where(actions_0 > actions_1, obs_0 - actions_0, torch.tensor(0.0, device=self.device))
        rewards_1 = torch.where(actions_1 >= actions_0, obs_1 - actions_1, torch.tensor(0.0, device=self.device))

        # Stack rewards for both agents
        return [rewards_0, rewards_1]
    
    def calculate_l2_loss(self, test_actions, validation_obs):
        # return pow(abs(test_actions - (0.5)*validation_obs),2).mean()
        return torch.sqrt(pow(abs(test_actions - self.get_equilibrium_action(validation_obs)), 2).mean())
 
    
    def calculate_utility_loss_currentStrategies(self, action_n, obs_n):
        equilibrium_action_n = [self.get_equilibrium_action(obs) for obs in obs_n]
        reward_equilibrium_n = []
        for i, eq_action in enumerate(equilibrium_action_n):
            eq_actions = action_n.copy()
            eq_actions[i] = eq_action
            reward_equilibrium = self.get_rewards(eq_actions, obs_n)[i]
            reward_equilibrium_n.append(reward_equilibrium)
        reward_own_n = []
        for i, _ in enumerate(action_n):
            reward_own = self.get_rewards(action_n, obs_n)[i]
            reward_own_n.append(reward_own)
        utility_loss_n = []
        for i in range(len(obs_n)):
            utility_loss_n.append(1-(torch.sum(reward_own_n[i])/torch.sum(reward_equilibrium_n[i])))
        return utility_loss_n

    def calculate_utility_loss(self, action_n, obs_n):
        equilibrium_action_n = [self.get_equilibrium_action(obs) for obs in obs_n]
        reward_equilibrium_n = []
        for i in range(len(obs_n)):
            reward_equilibrium = self.get_rewards(equilibrium_action_n, obs_n)[i]
            reward_equilibrium_n.append(reward_equilibrium)
        reward_own_n = []
        for i, action_i in enumerate(action_n):
            eq_actions = equilibrium_action_n.copy()
            eq_actions[i] = action_i
            reward_own = self.get_rewards(eq_actions, obs_n)[i]
            reward_own_n.append(reward_own)
        utility_loss_n = []
        for i in range(len(obs_n)):
            utility_loss_n.append(1-(torch.sum(reward_own_n[i])/torch.sum(reward_equilibrium_n[i])))
        return utility_loss_n
    
    def get_equilibrium_action(self, obs): 
        return 0.5*obs

    def close(self):
        print("Closing the environment")
        pass

