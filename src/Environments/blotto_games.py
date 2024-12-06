# python file to create a custom gym environment modeling a 2 player blotto game
import gymnasium as gym
import numpy as np
import torch



# A blotto game with inclompete information
# https://www.sciencedirect.com/science/article/pii/S0165176509002055


class BlottoWithInclompete(gym.Env): 
    def __init__(self, num_agents, num_objects, device):
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.fixed_budget = False
        # self.value_object is an array of size num_objects with the value of each object, the value of all objects must equal 1, the distribution should be random
        # self.values_objects = np.random.dirichlet(np.ones(num_objects),size=1)[0]
        self.values_objects = torch.tensor([0.05, 0.25, 0.7], device=device)

        self.action_space = []
        # the budget of each player is the observation, they sample independent values on an interval between 0 and 1
        self.observation_space = []
        self.state = []
        self.device = device
        for _ in range(self.num_agents):
            self.action_space.append(gym.spaces.Box(low=0, high=1, shape=(num_objects,), dtype=np.float32))
            self.observation_space.append(gym.spaces.Box(low=0, high=1, dtype=np.float32))

        if self.fixed_budget: 
            self.state.append(torch.tensor([1.0], device=self.device))
            self.state.append(torch.tensor([1.0], device=self.device))
        else:
            self.state.append(torch.tensor(self.observation_space[0].sample(), device=self.device))
            self.state.append(torch.tensor(self.observation_space[1].sample(), device=self.device))
   
    def step(self, action_n):
        # proof if valid action here? TODO decide if this is the best option to do this, or this too far aways from the model?
        # action for each player is a vector of size num_objects with values between 0 and 1, the sum of all values must be the budget of the corresponding player
        action_n[0] = (action_n[0]/torch.sum(action_n[0]))*self.state[0]
        action_n[1] = (action_n[1]/torch.sum(action_n[1]))*self.state[1]

        obs_n = []
        truncations_n = []
        terminations_n = []
        info_n = []
        
        # Initialize reward tensors
        reward_n = [torch.tensor([0], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)]

        # Calculate the payoff/reward for each player
        for i in range(self.num_objects):
            if action_n[0][i] > action_n[1][i]:
                reward_n[0] += torch.tensor([self.values_objects[i]])
            elif action_n[0][i] < action_n[1][i]:
                reward_n[1] += torch.tensor([self.values_objects[i]])
            else:
                reward_n[0] += torch.tensor([self.values_objects[i]/2])
                reward_n[1] += torch.tensor([self.values_objects[i]/2])

        # after step randomly uddate the state
        if self.fixed_budget: 
            self.state[0]= (torch.tensor([1.0], device=self.device))
            self.state[1]= (torch.tensor([1.0], device=self.device))
        else:
            self.state[0]= (torch.tensor(self.observation_space[0].sample(), device=self.device))
            self.state[1]= (torch.tensor(self.observation_space[1].sample(), device=self.device))
        obs_n.append(self.state[0])
        obs_n.append(self.state[1])
        
        info_n.append([{}])
        info_n.append([{}])
        return obs_n, reward_n, truncations_n, terminations_n, info_n

    def sample_obs(self, nr_of_samples, agent_id):
        # samples = np.array([self.observation_space[agent_id].sample() for _ in range(nr_of_samples)])
        # return torch.tensor(samples, device=self.device)  
        low = int(self.observation_space[agent_id].low)
        high = int(self.observation_space[agent_id].high)
        # TODO does not work wiht multi-dim observations
        samples = torch.rand((nr_of_samples,1), device=self.device)
        samples = low + samples * (high - low)
        return samples
    
    def reset(self):
        if self.fixed_budget: 
            self.state[0]= (torch.tensor([1.0], device=self.device))
            self.state[1]= (torch.tensor([1.0], device=self.device))
        else:
            self.state[0]= (torch.tensor(self.observation_space[0].sample(), device=self.device))
            self.state[1]= (torch.tensor(self.observation_space[1].sample(), device=self.device))
        obs_n = []
        obs_n.append(self.state[0])
        obs_n.append(self.state[1])
        return obs_n

    def get_rewards(self, action_n, obs_n):
        obs_0 = obs_n[0]
        obs_1 = obs_n[1]
        action_0 = action_n[0]
        action_1 = action_n[1]
        # ###### whatch out for when actions are not constrained by the model
        # action_0 = (action_0/torch.sum(action_0, dim=1, keepdim=True))*obs_0
        # action_1 = (action_1/torch.sum(action_1, dim=1, keepdim=True))*obs_1
        # ########
        batch_size, action_dim = action_0.shape
        total_reward_0 = torch.zeros((batch_size, 1), device=self.device)
        total_reward_1 = torch.zeros((batch_size, 1), device=self.device)

        for i in range(action_dim):
            action_0i = action_0[:, i]
            action_1i = action_1[:, i]

            value_i = self.values_objects[i]

            # Compute rewards based on comparison
            reward_0 = torch.where(action_0i > action_1i, value_i, torch.zeros_like(action_0i))
            reward_1 = torch.where(action_1i > action_0i, value_i, torch.zeros_like(action_1i))

            # Handle the case where actions are equal
            tie_reward = torch.where(action_0i == action_1i, value_i / 2, torch.zeros_like(action_0i))

            # Sum up rewards for the batch
            total_reward_0 += reward_0.unsqueeze(1) + tie_reward.unsqueeze(1)
            total_reward_1 += reward_1.unsqueeze(1) + tie_reward.unsqueeze(1)

        return [total_reward_0, total_reward_1]


    def calculate_l2_loss(self, test_actions, validation_obs):
        # action dimesnion [256, 3]
        # obs dimension [256, 1]
        # scale each of the 256 actions triples by sum(of these 3)*corresponding obs
        # get_equilbirium_action should also return actions in dimension [256, 3]

        # ######'TODO whatch out
        # test_actions = (test_actions/torch.sum(test_actions, dim=1, keepdim=True))*validation_obs
        # #######TODO        

        # scale each of the 256 actions triples by sum(of these 3)*corresponding obs 
        # test_actions = (test_actions / test_actions.sum(dim=-1, keepdim=True)) * validation_obs
        equilibrium_actions = self.get_equilibrium_action(validation_obs)
        squared_errors = (test_actions - equilibrium_actions) ** 2
        mean_squared_error_per_sample = squared_errors.mean(dim=-1)
        rmse = torch.sqrt(mean_squared_error_per_sample.mean())
        return rmse
  
    def calculate_utility_loss_currentStrategies(self, action_n, obs_n):
        ### watch out 
        # action_n = [(action/torch.sum(action, dim=1, keepdim=True))*obs for action, obs in zip(action_n, obs_n)]

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
        ######' whatch out this part if the actions are not already constrained
        # action_0 = (action_0/torch.sum(action_0, dim=1, keepdim=True))*obs_values_0
        # action_1 = (action_1/torch.sum(action_1, dim=1, keepdim=True))*obs_values_1
        # action_n = [(action/torch.sum(action, dim=1, keepdim=True))*obs for action, obs in zip(action_n, obs_n)]
        #######
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
        # for each dimension the valuation times the observation/budget is the equilibrium action
        # [256, 1] obs dimension
        # [256, 3] action dimension/return dimension
        # self.values_objects dimension [3]
        return (self.values_objects * obs)# why did copilot unsqueeze(0)? TODO check if makes sense
    
    
class BlottoWithIncompleteMultiDim(): 
    def __init__(self, num_agents, num_objects, device):
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.fixed_budget = False
        # self.value_object is an array of size num_objects with the value of each object, the value of all objects must equal 1, the distribution should be random
        # self.values_objects = np.random.dirichlet(np.ones(num_objects),size=1)[0]

        ## not needed anymore as our obs is now the valuation
        # self.values_objects = torch.tensor([0.05, 0.25, 0.7], device=device)

        self.action_space = []
        # the budget of each player is the observation, they sample independent values on an interval between 0 and 1
        self.observation_space = []
        self.state = []
        self.device = device
        for _ in range(self.num_agents):
            self.action_space.append(gym.spaces.Box(low=0, high=1, shape=(num_objects,), dtype=np.float32))
            # TODO try with new observation space check where obs space is used 
            self.observation_space.append(gym.spaces.Box(low=0, high=1, shape=(num_objects,), dtype=np.float32))


    def sample_obs(self, nr_of_samples, agent_id):
        # Generate random samples uniformly in the positive octant
        samples = torch.rand((nr_of_samples, 3), device=self.device)
        norms = torch.norm(samples, dim=1, keepdim=True)
        samples = samples / norms
        
        return samples
    
    def get_rewards(self, action_n, obs_n):
        # rewards should be the same
        obs_values_0 = obs_n[0]
        obs_values_1 = obs_n[1]
        action_0 = action_n[0]
        action_1 = action_n[1]
        # #######' whatch out
        # action_0 = (action_0/torch.sum(action_0, dim=1, keepdim=True))
        # action_1 = (action_1/torch.sum(action_1, dim=1, keepdim=True))
        # ########
        batch_size, action_dim = action_0.shape
        total_reward_0 = torch.zeros((batch_size, 1), device=self.device)
        total_reward_1 = torch.zeros((batch_size, 1), device=self.device)

        for i in range(action_dim):
            action_0i = action_0[:, i]
            action_1i = action_1[:, i]

            value_0_i = obs_values_0[:, i]
            value_1_i = obs_values_1[:, i]

            # Compute rewards based on comparison
            reward_0 = torch.where(action_0i > action_1i, value_0_i, torch.zeros_like(action_0i))
            reward_1 = torch.where(action_1i > action_0i, value_1_i, torch.zeros_like(action_1i))

            # Handle the case where actions are equal
            # tie is wrong but will never happen anyways because continious case
            tie_reward = torch.where(action_0i == action_1i, value_1_i / 2, torch.zeros_like(action_0i))

            # Sum up rewards for the batch
            total_reward_0 += reward_0.unsqueeze(1) + tie_reward.unsqueeze(1)
            total_reward_1 += reward_1.unsqueeze(1) + tie_reward.unsqueeze(1)

        return [total_reward_0, total_reward_1]
    
    def calculate_l2_loss(self, test_actions, validation_obs):
        # action dimesnion [256, 3]
        # obs dimension [256, 1]
        # scale each of the 256 actions triples by sum(of these 3)*corresponding obs
        # get_equilbirium_action should also return actions in dimension [256, 3]

        #######' whatch out
        #test_actions = (test_actions/torch.sum(test_actions, dim=1, keepdim=True))
        ########

        # scale each of the 256 actions triples by sum(of these 3)*corresponding obs 
        # test_actions = (test_actions / test_actions.sum(dim=-1, keepdim=True)) * validation_obs
        equilibrium_actions = self.get_equilibrium_action(validation_obs)
        squared_errors = (test_actions - equilibrium_actions) ** 2
        mean_squared_error_per_sample = squared_errors.mean(dim=-1)
        rmse = torch.sqrt(mean_squared_error_per_sample.mean())
        return rmse
    
    def calculate_utility_loss_currentStrategies(self, action_n, obs_n):
        #######' 
       # action_0 = (action_0/torch.sum(action_0, dim=1, keepdim=True))*obs_values_0
        #action_1 = (action_1/torch.sum(action_1, dim=1, keepdim=True))*obs_values_1
        # action_n = [(action/torch.sum(action, dim=1, keepdim=True)) for action, obs in zip(action_n, obs_n)]
        ########

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
        #######'TODO whatch out
       # action_0 = (action_0/torch.sum(action_0, dim=1, keepdim=True))*obs_values_0
        #action_1 = (action_1/torch.sum(action_1, dim=1, keepdim=True))*obs_values_1
        # action_n = [(action/torch.sum(action, dim=1, keepdim=True)) for action, obs in zip(action_n, obs_n)]
        ########TODO
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
        # for each dimension the valuation times the observation/budget is the equilibrium action
        # [256, 3] obs dimension
        # [256, 3] action dimension/return dimension
        return (obs ** 2)# why did copilot unsqueeze(0)? TODO check if makes sense

    def close(self): 
        print("closing env")