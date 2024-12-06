import gymnasium as gym
import numpy as np



#### legacy code does not work with tensor replay buffer ####
# two bidders one with a budget of 10 the other with a budget of 2, the highest bid wins and everybody their bid
class AllPayAuction(gym.Env): 
    def __init__(self, num_agents):
        self.num_agents = num_agents
        # TODO add number of battlefield to args
        # self.number_battlefields = 1
        # TODO: adjust the shape of the action space to the number of battlefields
 
        # TODO: not sure if we need a observation space for each agent
        self.action_space = []
        self.observation_space = []
        # self.state = []
        for i in range(self.num_agents):
            if i==0:
                self.action_space.append(gym.spaces.Box(low=0, high=10, dtype=np.float32))
            else: 
                self.action_space.append(gym.spaces.Box(low=0, high=2, dtype=np.float32))
            # we only have one observation, as the state always stays the same
            # self.observation_space.append(gym.spaces.Box(low=0, high=1, dtype=np.float32))
            # self.state = np.array(self.observation_space[i].sample())
            self.observation_space.append(gym.spaces.Discrete(1))

        # TODO: this must fit to the networks input, what is the correct observation space shape?
        # sample observation 

        #self.state = self.observation_space[1].sample(1)
        self.state = np.array([1.])
   
    def step(self, action_n):
        obs_n = []
        reward_n = []
        truncations_n = []
        terminations_n = []
        info_n = []
        # TODO: add to args
        valuation_battlefield = 10
        # TODO adjust for multiple battlefields and maybe more agents
        # TODO: decide how to handle ties 
        # given all actions in action_n we choose the winner by comparing the value of action1[0],
        # aciton is two dimension to prepare when we have multiple battlefields
        if action_n[0][0] > action_n[1][0]:
            reward_n.append(np.array([valuation_battlefield-action_n[0][0]]))
            reward_n.append(np.array([-action_n[1][0]]))
        else:
            reward_n.append(np.array([-action_n[0][0]]))
            reward_n.append(np.array([valuation_battlefield-action_n[1][0]]))

        # TODO update the state based on the actions (if we dont have states this doesn change anything)
        # self.state[0] = self.observation_space[0].sample(1)
        # self.state[1] = self.observation_space[0].sample(1)
        # obs_n.append(self.state[0])
        # obs_n.append(self.state[1])
        obs_n.append(self.state)
        obs_n.append(self.state)
        
        # TODO because we only have one state, are we always done after one step? How does this affect the training loop?
        # truncations_n.append(False), terminations_n.append(True)

        # TODO info needs to be in this format: List[Dict[str, Any]], check if this is the correct format
        # create an object of List[Dict[str, Any]] with two empty dictionaries and append them to info_n
        info_n.append([{}])
        info_n.append([{}])
        return obs_n, reward_n, truncations_n, terminations_n, info_n

    def reset(self):
        # reset the state to the start state
        # TODO: decide how to handle as we do not need states currently and it stays the same
        obs_n = []
        # TODO here sample state 
        obs_n.append(self.state)
        obs_n.append(self.state)
        return obs_n
    
# two bidders with a budget of 10, the highest bid wins and pays the amount of the bid
# the value of the item is 10 for bidder 1 and 4 for bidder 2
class FirstPricedAuction(gym.Env): 
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.action_space = []
        self.observation_space = []
        for i in range(self.num_agents):
            if i==0:
                self.action_space.append(gym.spaces.Box(low=0, high=10, dtype=np.float32))
            else: 
                self.action_space.append(gym.spaces.Box(low=0, high=10, dtype=np.float32))
            self.observation_space.append(gym.spaces.Discrete(1))

        self.state = np.array([1.])
   
    def step(self, action_n):
        obs_n = []
        reward_n = []
        truncations_n = []
        terminations_n = []
        info_n = []
        # alternative the state of the game could determine the value of the battlefield 
        valuation_battlefield_bidder1 = 10
        valuation_battlefield_bidder2 = 4
        if action_n[0][0] > action_n[1][0]:
            reward_n.append(np.array([valuation_battlefield_bidder1-action_n[0][0]]))
            reward_n.append(np.array([0]))
        else:
            reward_n.append(np.array([0]))
            reward_n.append(np.array([valuation_battlefield_bidder2-action_n[1][0]]))


        obs_n.append(self.state)
        obs_n.append(self.state)
        
        info_n.append([{}])
        info_n.append([{}])
        return obs_n, reward_n, truncations_n, terminations_n, info_n

    def reset(self):
        obs_n = []
        obs_n.append(self.state)
        obs_n.append(self.state)
        return obs_n