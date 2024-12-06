import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import normflows as nf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import time


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, agent_index):
        super().__init__()
        self.fc1 = nn.Linear(int(np.array(env.observation_space[agent_index].shape).prod() + np.prod(env.action_space[agent_index].shape)), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# class from the original code, where a simple gaussian distribution is learned and squashed
class Actor_Gaussian(nn.Module):
    def __init__(self, env, env_name, agent_index, do_squash_action=True):
        super().__init__()
        self.do_squash_action = do_squash_action # if true action will be squased and rescaled
        input = env.observation_space[agent_index].shape
        input = np.array(input, dtype=int)
        input =input.prod()
        self.fc1 = nn.Linear(input, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space[agent_index].shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space[agent_index].shape))
        self.action_dim = np.prod(env.action_space[agent_index].shape)
        self.env_name = env_name
        self.model_blotto_constraints = True # normal must be true TODO as parameter args

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space[agent_index].high - env.action_space[agent_index].low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space[agent_index].high + env.action_space[agent_index].low) / 2.0, dtype=torch.float32)
        )
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, obs, nr_samples=1):
        if len(obs.shape) == 1 and nr_samples == 1:
            # add batch dimension if it is not there
            raise ValueError("The observation should have a batch dimension")

        mean, log_std = self(obs)
        std = log_std.exp()

        # self.action_dim should be equal to the dimension of the second dimension of the action, where the first dimension is the batch dimension
        if self.action_dim > 1:
            variance = std.pow(2)
            covariance_matrix = torch.diag_embed(variance)
            normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        else:
            normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() if nr_samples == 1 else normal.rsample([nr_samples])
        if not self.do_squash_action:
            # sum over multivariate action dimension -> there is onthing to sum over as this is one value for the entire action
            log_prob = normal.log_prob(x_t)
            if self.action_dim > 1: 
                log_prob = log_prob.unsqueeze(1)
            return x_t, log_prob, mean
                
        log_prob = normal.log_prob(x_t)
        if self.action_dim > 1: 
            log_prob = log_prob.unsqueeze(1) # For some reaseon log_prob.multiVariate and log_prob.Normal return different dimensions
        
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        if len(action.shape)== 1: 
            raise ValueError("The action should have a batch dimension")

        term_to_subtract = torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        term_to_subtract = term_to_subtract.sum(1, keepdim=True)
        log_prob -= term_to_subtract
        # this is not correct multidim action but this mean is never used anyway
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        # it is kinda weired because i have some type of game logic inside my model now, which makes the model less general
        # normalizing the multi dim action to the obs which is the budget in blotto games
        # sqaushing and rescaling the action is not needed when i am doing this 
        if self.model_blotto_constraints: 
            if self.action_dim>1:
                if self.env_name == "BlottoWithInclompete":
                    action = (action/torch.sum(action, dim=1, keepdim=True))*obs
                    log_prob = log_prob + torch.log(torch.sum(action, dim=1, keepdim=True)) - torch.log((obs))
                else: 
                    action = (action/torch.sum(action, dim=1, keepdim=True))
                    log_prob = log_prob + torch.log(torch.sum(action, dim=1, keepdim=True))

        return action, log_prob, mean

    def get_mean_action(self, obs):
        mean, _ = self(obs)
        action=  torch.tanh(mean) * self.action_scale + self.action_bias
        # it is kinda weired because i have some type of game logic inside my model now, which makes the model less general
        # normalizing the multi dim action to the obs which is the budget in blotto games
        if self.model_blotto_constraints: 
            if self.action_dim>1:
                if self.env_name == "BlottoWithInclompete":
                    action = (action/torch.sum(action, dim=1, keepdim=True))*obs
                else: 
                    action = (action/torch.sum(action, dim=1, keepdim=True))
        return action
    
    # TODO adjust to scaling in blotto games and sum
    def evaluate_logDensity(self, obs, a):
        """Calculate the log density of the action given the observation.
        Args:
            obs (torch.Tensor): The observation.
            a (torch.Tensor): The action must be in the action space of the environment."""
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if not self.do_squash_action:
            log_prob = normal.log_prob(a)
            return log_prob
        
          # check if a is in the right intervall 
        if (torch.any(a > self.action_bias + self.action_scale) or torch.any(a < self.action_bias - self.action_scale)):
            raise ValueError(f"Action is not in the action space of the environment. Action: {a}, Action space: {self.action_bias - self.action_scale} to {self.action_bias + self.action_scale}")
        
        a_reversed = (a - self.action_bias) / self.action_scale #scaled back to between -1 and 1
        a_unsquased = torch.atanh(a_reversed) #unsquash

        # this sqeeze shouldnt work with multi dim actions, at best we would need to squeze because we want to evaluate a batch of actions 
        log_prob = normal.log_prob(a_unsquased)#.squeze(1))
        # Add small epsilon to avoid log(0)
        term_to_subtract = torch.log(self.action_scale * (1 - a_reversed.pow(2)) + 1e-6)
        log_prob -= term_to_subtract
        return log_prob
    
# wrapper class for our flow actor including the flow model and the state encoder for our base gaussian distribution
class Actor_Flow(object):

    def __init__(self, env, env_name, agent_index, args):
        # Define flows code from https://vincentstimper.github.io/normalizing-flows/examples/neural_spline_flow/
        self.args = args
        self.env_name = env_name
        self.use_gaussian_in_flow = args.use_gaussian_in_flow
        self.action_dim = np.prod(env.action_space[agent_index].shape)
        K = args.nr_flow_transformations
        # TODO make flow smaller K = 2 oder 1 
        hidden_units = args.hidden_units # 10 hidden unit might be enough
        hidden_layers = args.hidden_layers # 2 are used
        latent_size = np.prod(env.action_space[agent_index].shape)
        self.total_time_nfm_fwd_call = 0

        # construct the flow model
        # we use neural spline flows from https://arxiv.org/abs/1906.04032
        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)] # ME: Does the flow live on the GPU? Yes, beacuase we call to(device) in the SAC_Agent
            flows += [nf.flows.LULinearPermute(latent_size)]
        # Set base distribuiton, not needed for our case, because we have a learned base distribution
        q0 = nf.distributions.DiagGaussian(int(latent_size), trainable=False)
        self.nfm = nf.NormalizingFlow(q0=q0, flows=flows)

        # construct model to learn gaussian (base distributions) parameter
        # The action is squased when recieving it form actor_gaussian
        if self.use_gaussian_in_flow: self.guassian_model = Actor_Gaussian(env, self.env_name, agent_index, False)

        self.action_scale = torch.tensor((env.action_space[agent_index].high - env.action_space[agent_index].low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((env.action_space[agent_index].high + env.action_space[agent_index].low) / 2.0, dtype=torch.float32)

    def to(self, device):
        self.nfm = self.nfm.to(device)
        if self.use_gaussian_in_flow: self.guassian_model = self.guassian_model.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return self
        
    def parameters(self):
        if self.use_gaussian_in_flow: return chain(self.nfm.parameters(), self.guassian_model.parameters())
        else: return self.nfm.parameters()

    def get_total_fwd_time(self):
        return self.total_time_nfm_fwd_call
    
    def get_action(self, obs, nr_samples=1, returnTransformationActions=False):
        # TODO for blotto games this runs quickly out of cuda memore, find out why and how to fix it
        if self.use_gaussian_in_flow:
            gaussian_action, log_prob, _ = self.guassian_model.get_action(obs, nr_samples)
        else: 
            if self.env_name == 'BlottoWithInclompete':
                gaussian_action = obs.repeat(1,self.args.num_objects)
            else: gaussian_action= obs
            log_prob = 0
        # for flow without base distribution but not needed for now
        # if self.use_gaussian_in_flow: gaussian_action, gaussian_log_prob, _ = self.guassian_model.get_action(x, nr_samples)
        # elif self.args.env_name == "BlottoWithInclompete":
        #     gaussian_action, gaussian_log_prob = x.repeat(1,self.args.num_objects), 0
        # else: gaussian_action, gaussian_log_prob = x, 0
        # transform gaussian action of shape [256, 1] to [256, 3] by simply repeating the action
        # gaussian_action = gaussian_action.repeat(1,3)
        # action_transformed = self.nfm.forward(gaussian_action)
        action_transformed, log_det_flos_fwd = self.nfm.forward_and_log_det(gaussian_action) 
        # action_transformed, log_det_flows = self.nfm.forward_and_log_det(gaussian_action)
        # this does not work with batch dimension directly, the implementation is different using len(action) to define dimenstion of log_prob
        # dont know if this will work with multi dim actions
        # For single dim action can simple add unsqueze(1) to log_prob
        #reversed_action, log_det_flows = self.nfm.inverse_and_log_det(action_transformed)
        # compare the log_prob flows without the base distribution and log_prob_flows_oneFunction
        log_det_flows = - log_det_flos_fwd.unsqueeze(1)
        
        sample_saqushed = torch.tanh(action_transformed)
        action = sample_saqushed * self.action_scale + self.action_bias
        log_det_squashingScaling_negative = torch.log(self.action_scale * (1 - sample_saqushed.pow(2)) + 1e-6)
        log_det_squashingScaling_negative = log_det_squashingScaling_negative.sum(1, keepdim=True)

        log_prob += (log_det_flows - log_det_squashingScaling_negative)
        
        if returnTransformationActions: return gaussian_action, action_transformed, action

        # it is kinda weired because i have some type of game logic inside my model now, which makes the model less general
        # normalizing the multi dim action to the obs which is the budget in blotto games
        if self.action_dim>1:
            if self.env_name == "BlottoWithInclompete":
                action = (action/torch.sum(action, dim=1, keepdim=True))*obs
                log_prob = log_prob + torch.log(torch.sum(action, dim=1, keepdim=True)) - torch.log((obs))
            else: 
                action = (action/torch.sum(action, dim=1, keepdim=True))
                log_prob = log_prob + torch.log(torch.sum(action, dim=1, keepdim=True))


        return action, log_prob, None
    
    # TODO adjust to scaling in blotto games
    def evaluate_logDensity(self, obs, a):
        """Calculate the log density of the action given the observation.
        Args:
            obs (torch.Tensor): The observation.
            a (torch.Tensor): The action must be in the action space of the environment."""
        # check if a is in the right intervall 
        if (torch.any(a > self.action_bias + self.action_scale) or torch.any(a < self.action_bias - self.action_scale)):
            raise ValueError(f"Action is not in the action space of the environment. Action: {a}, Action space: {self.action_bias - self.action_scale} to {self.action_bias + self.action_scale}")
        
        a_reversed = (a - self.action_bias) / self.action_scale
        a_unsquased = torch.atanh(a_reversed)

        # returns the inverse of the transformation z = T^-1(a) and the log prob of the flow
        z, log_det_flow = self.nfm.inverse_and_log_det(a_unsquased)
        log_prob_baseDistribution = self.guassian_model.evaluate_logDensity(obs, z)
        log_det_squashingScaling_negative = torch.log(self.action_scale * (1 - a_reversed.pow(2)) + 1e-6)
        log_prob = log_prob_baseDistribution + log_det_flow - log_det_squashingScaling_negative
        # TODO sum over action dimension if we have multivariate distribution or somehow adjust it at least

        return log_prob
    
# Legacy code, not adapted to new code
class Actor_Beta(nn.Module):
    def __init__(self, env, agent_index):
        super().__init__()
        input = env.observation_space[agent_index].shape
        input = np.array(input, dtype=int)
        input =input.prod()
        self.fc1 = nn.Linear(input, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_alpha = nn.Linear(256, np.prod(env.action_space[agent_index].shape))
        self.fc_beta = nn.Linear(256, np.prod(env.action_space[agent_index].shape))
        self.softplus = nn.Softplus()

        # # the result of the beta function lies between 0 and 1, rescaling different than for gaussion squashed into -1 and 1
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space[agent_index].high - env.action_space[agent_index].low), dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space[agent_index].low), dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        alpha = self.fc_alpha(x)
        beta = self.fc_beta(x)
        # ensuring postive values for alpha and beta parameters
        alpha = self.softplus(alpha)
        beta = self.softplus(beta)
        
        return alpha, beta

    def get_action(self, obs, nr_samples=1):
        if len(obs.shape) == 1 and nr_samples == 1:
            # add batch dimension if it is not there
            # this should never happen thats why throw error
            raise ValueError("The observation should have a batch dimension")
        alpha, beta = self(obs)
        beta_distribtion = torch.distributions.Beta(alpha, beta)
        # for reparameterization trick, not sure how they do it with the beta function
        x_t = beta_distribtion.rsample() if nr_samples == 1 else beta_distribtion.rsample([nr_samples])
        # the result of the beta function lies between 0 and 1
        action = x_t * self.action_scale + self.action_bias
        log_prob = beta_distribtion.log_prob(x_t)
        # TODO do I need to adjust the log_prob as for the gaussian? 

        return action, log_prob, None

    def plot_target(self, obs):
        with torch.no_grad():
            sample_rescaled, log_prob, _ = self.get_action(obs, 1000)

        prob = torch.exp(log_prob.view(-1))
        sample_array = sample_rescaled.view(-1).cpu().numpy()
        prob_array = prob.cpu().numpy()
        return sample_array, prob_array
    
    def plot_heatmap(self, device): 
        # create a seborn heatmap with the probabilities as colors, valuations as x-axis and intervals as y-axis
        valuations = np.arange(0.01, 1.01, 0.01)
        action_intervals = [(round(i * 0.01, 2), round((i + 1) * 0.01, 2)) for i in range(100)]
        action_intervals = action_intervals[::-1]
        
        # Calculate probabilities for each valuation and action interval
        probabilities = []
        for interval in action_intervals:
            row_probs = []
            for val in valuations:
                lower_bound, upper_bound = interval
                probability = self.calculate_probability(val, lower_bound, upper_bound, device)
                row_probs.append(probability)
            probabilities.append(row_probs)
        
        heatmap_data = np.array(probabilities)

        plt.figure()
        sns.heatmap(heatmap_data)
        plt.xticks(ticks=np.arange(0, 100, 10), labels=np.arange(0, 1, 0.1).round(2))
        plt.xlabel("Object Valuation")
        plt.yticks(ticks=np.arange(10, 110, 10), labels=np.arange(0, 1, 0.1).round(2)[::-1])
        plt.ylabel("Action Interval")
        fig = plt.gcf()
        plt.close()
        return fig 

    def calculate_probability(self, valuation, lower_bound, upper_bound, device):
        # calcualte interval probabilities like this: probability = normal_dist.cdf(upper_bound) - normal_dist.cdf(lower_bound)
        # cdf is not implemented for beta distribtution-> we will work with log_prob of the pdf, which will result in values larger than 1, but proportionally make sense (i think)
        # this means we do not need upper and lower bound but rather just calculate the log_prob of a value in the middle
        middle = (upper_bound + lower_bound) / 2
        with torch.no_grad():
            alpha, beta = self(torch.tensor([valuation], dtype=torch.float32).to(device))
            beta_distribtion = torch.distributions.Beta(alpha, beta)
            middle_reversed = (middle - self.action_bias) / self.action_scale.to(device)
            log_probability = beta_distribtion.log_prob(torch.tensor([middle_reversed]).to(device))
            probability =  torch.exp(log_probability.view(-1))
        return probability.item()
