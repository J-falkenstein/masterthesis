import math
import os
import torch
from models import Actor_Flow, SoftQNetwork, Actor_Gaussian, Actor_Beta
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import time
import torchist
import numpy as np


# contains the actor critic logic, does not inlcude the training loop
class SAC_Agent(object):

    def __init__(self, env, env_name, agent_index, device, args):
        self.env = env
        self.env_name = env_name
        self.device = device
        self.args = args
        self.agent_index = agent_index
        self.fixed_strategy = False
        if self.args.fix_all_but_one_agent and agent_index != 0:
            self.fixed_strategy = True
        if args.actor_type == "flow":
            self.actor = Actor_Flow(env,  self.env_name, agent_index, args).to(device)
        elif args.actor_type == "gaussian":
            self.actor = Actor_Gaussian(env, self.env_name, agent_index).to(device)
        elif args.actor_type == "beta":
            self.actor = Actor_Beta(env, agent_index).to(device)
        else:
            raise ValueError(f"Actor type {args.actor_type} not supported")
        self.qf1 = SoftQNetwork(env, agent_index).to(device)
        self.qf2 = SoftQNetwork(env, agent_index).to(device)
        #self.qf1_target = SoftQNetwork(env, agent_index).to(device)
        #self.qf2_target = SoftQNetwork(env, agent_index).to(device)
        # self.qf1_target.load_state_dict(self.qf1.state_dict())
        # self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        # Automatic entropy tuning
        if self.args.autotune:
            self.target_entropy = -3#-torch.prod(torch.Tensor(env.action_space[agent_index].shape).to(device)).item() 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            # todo make as args
            self.alpha = self.args.alpha
            self.alpha_initial = self.args.alpha
            #self.lambda_ = 0.000276073
            #self.lambda_ =  4.60083e-5 7.0975e-5
            self.lambda_ = 2.608e-5 # 3.26e-5


    def update_alpha(self, global_step): 
        self.alpha = self.alpha_initial * math.exp(-self.lambda_ * global_step)

    def select_action(self, obs):
        if self.fixed_strategy: 
            return self.env.get_equilibrium_action(obs)
        actions, log_prob, _ = self.actor.get_action(obs)
        return actions
    
    def select_mean_action(self, obs):
        if self.fixed_strategy: 
            return self.env.get_equilibrium_action(obs)
        actions = self.actor.get_mean_action(obs)
        return actions

    def pretrain_identity(self):
        # Create a dataset of 10000 samples of random values between 0 and 1
        nr_samples = 10000
        batch_size = 256
        if self.env_name == "BlottoWithInclompeteMultiDim":
            data = self.env.sample_obs(nr_samples, self.agent_index)
        else: data = torch.rand(nr_samples, 1, device=self.device)
        dataset = TensorDataset(data, data)  # using the input data as target
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        action_space_size = self.env.action_space[self.agent_index].shape[0]

        for epoch in range(50):
            total_loss = 0.0
            for inputs, targets in data_loader:
                # expand targets to match the shape of the actions (e.g. in Blotto Game)
                targets = targets.expand(-1, action_space_size)
                self.actor_optimizer.zero_grad()
                actions, _, _ = self.actor.get_action(inputs)
                loss = F.mse_loss(actions, targets)
                loss.backward()
                self.actor_optimizer.step()
                total_loss += loss.item()

        if self.env_name == "BlottoWithInclompeteMultiDim":
            test_data = self.env.sample_obs(100, self.agent_index)
        else: test_data = torch.linspace(0, 1, 100, device=self.device).view(-1, 1)

        with torch.no_grad():
            test_actions, _, _ = self.actor.get_action(test_data)

        # Plotting
        test_data = test_data.expand(-1, self.env.action_space[self.agent_index].shape[0])
        plt.figure(figsize=(10, 5))
        plt.scatter(test_data.cpu().numpy(), test_actions.cpu().numpy(), color='r', label='Output vs Input')
        plt.plot(test_data.cpu().numpy(), test_data.cpu().numpy(), label='y=x (ideal line)', linestyle='--')
        plt.title('Input vs Output after Pretraining')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(True)
        fig = plt.gcf()
        plt.close()

        return fig, test_actions.mean().item()

    def update_parameters(self, data, global_step):
        if self.fixed_strategy:
            # return self.alpha, alpha_loss, qf_loss, actor_loss, qf1_loss, qf2_loss, qf1_a_values, qf2_a_values
            alpha_loss = torch.tensor(0.0)
            qf_loss = torch.tensor(0.0)
            actor_loss = torch.tensor(0.0)
            qf1_loss = torch.tensor(0.0)
            qf2_loss = torch.tensor(0.0)
            qf1_a_values = torch.tensor(0.0)
            qf2_a_values = torch.tensor(0.0)
            return self.alpha, alpha_loss, qf_loss, actor_loss, qf1_loss, qf2_loss, qf1_a_values, qf2_a_values
        with torch.no_grad():
            #next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            #qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            #qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            # why do I subtract the entropy term here? Isnt this just for the q value calculation?
            #min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() #+ self.args.gamma * (min_qf_next_target).view(-1)
            # next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

        # These actions are detached from the computation graph, so their calculation will not be used to calculate the gradientsa
        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # needed because we always want return the values for logging -> not very clean 
        alpha_loss = None
        actor_loss = None

        if global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(self.args.policy_frequency):  
                # compensate for the delay by doing 'actor_update_interval' instead of 1
                # The calcualtion of the action (inside the actor) makes use of the reparametrization trick, so we can calculate the gradients of this funciton
                # We dont need REINFORCE because of our q values not because of the repametrization trick. 
                # The actor loss is not dependent on the rewards where we wouldn't know the gradient of the action with respect to the reward. 
                # The reparamezation trick allows to calculate the gradients of our stochastic acotr netowrks. If we would have a deterministic actor, we would not need the reparametrization trick
                # We also only use the log_pi for the entropy term and not like in REINFORCE for estimating the gradients
                pi, log_pi, _ = self.actor.get_action(data.observations)
                if self.args.env_name == "BlottoWithInclompete":
                    # rescale next action to the budget/observation of the player
                    pi = (pi/pi.sum(dim=1, keepdim=True))*data.observations
                qf1_pi = self.qf1(data.observations, pi)
                qf2_pi = self.qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                # we want a large qf value, because we substract qf from the loss, the loss should be as small as possible
                # Hence, also the log_pi should be as small as possible
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                # Does this also calcualte the gradients with regards of the input data of the qf networks as they are used in the calculation of the loss?
                # while actor_loss.backward() computes the necessary gradients that flow through the Q-function networks to the actor network,
                # only the actor network parameters are updated
                # Does this also calcualte the gradients with regards of the input data of the qf networks as they are used in the calculation of the loss?
                # while actor_loss.backward() computes the necessary gradients that flow through the Q-function networks to the actor network,
                # only the actor network parameters are updated
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                if self.args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks by using the updated primary networks
        # the target networks are never trained
        # if global_step % self.args.target_network_frequency == 0:
        #     for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
        #         target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        #     for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
        #         target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        # return the losses and values for logging
        return self.alpha, alpha_loss, qf_loss, actor_loss, qf1_loss, qf2_loss, qf1_a_values, qf2_a_values

    def plot_deterministic_policy(self):
        # for 1000 values between 0 and 1 plot the action that the actor would take
        test_data = torch.linspace(0, 1, 1000, device=self.device).view(-1, 1)
        with torch.no_grad():
            test_actions, _, _ = self.actor.get_action(test_data)
        plt.figure(figsize=(10, 5))
        plt.plot(test_data.cpu().numpy(), test_actions.cpu().numpy(), color='r', label='Output vs Input')
        plt.xlim(0, 1)
        plt.ylim(0, 1) 
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel('Observation')
        plt.ylabel('Action')
        plt.grid(True)
        fig = plt.gcf()
        plt.close()
        return fig

    def calculate_l2_loss(self):
        if self.env_name == "BlottoWithInclompeteMultiDim":
            validation_obs = self.env.sample_obs(2 ** 16, self.agent_index)
        else: validation_obs = torch.linspace(0, 1, 2 ** 16, device=self.device).view(-1, 1)
        with torch.no_grad():
            test_actions, _, _ = self.actor.get_action(validation_obs)
        l2_loss = self.env.calculate_l2_loss(test_actions, validation_obs)
        return l2_loss
    
    def calculate_l2_loss_meanAction(self):
        if self.env_name == "BlottoWithInclompeteMultiDim":
            validation_obs = self.env.sample_obs(2 ** 16, self.agent_index)
        else: validation_obs = torch.linspace(0, 1, 2 ** 16, device=self.device).view(-1, 1)
        with torch.no_grad():
            test_actions = self.actor.get_mean_action(validation_obs)
        l2_loss = self.env.calculate_l2_loss(test_actions, validation_obs)
        return l2_loss

    def plot_target(self):
        # TODO instead of for loop use tensor batch with three observations
        tables = []
        for obs in [0.2, 0.5, 0.8]:
            obs = torch.tensor([obs], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                sample, log_prob, _ = self.actor.get_action(obs, 1000)
            prob = torch.exp(log_prob.view(-1))
            sample_array = sample.view(-1).cpu().numpy()
            prob_array = prob.cpu().numpy()
            table_data = [[x, y] for (x, y) in zip(sample_array, prob_array)]
            tables.append([table_data, round(obs.item(), 2)])
        return tables
    
    def plot_heatmap(self):
        test_data = torch.linspace(0, 1, 2 ** 16, device=self.device).view(-1, 1)
        with torch.no_grad():
            test_actions, _, _ = self.actor.get_action(test_data)
        
        data = torch.cat((test_data, test_actions), dim=1)
        hist = torchist.histogramdd(data, bins=[100, 100], low = [0, 0], upp=[1, 1])#, range=[[0., 1.], [0., 1.]])
        heatmap = hist.cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest', aspect='auto',
                extent=[0, 1, 0, 1])
        # TODO find out what these edges are 
        plt.colorbar()
        plt.xlabel('Object Valuation')
        plt.ylabel('Action Interval')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        fig = plt.gcf()
        plt.close()
        return fig 

    def plot_relative_action_blotto(self):
        # for 1000 values/budget between 0 and 1 plot the relative action_o, action_1, and action_2 that the actor would take
        test_data = torch.linspace(0, 1, 2**16, device=self.device).view(-1, 1)
        with torch.no_grad():
            test_actions,_, _ = self.actor.get_action(test_data)
        # calculate the relative actions each actiondimensino divided by obs or testData test data is shpae (1000,1) and test_actions is shape (1000,3). relative action should be shape (1000,3)
        relative_action = test_actions/(test_data+ 1e-8)
        # create a plot for each action dimension mapping relative action on the x axis and log_prob on the y axis
        fig, axs = plt.subplots(3, figsize=(10, 15))
        for i in range(3):
            # Compute the histogram using torchist
            hist = torchist.histogram(relative_action[:, i], bins=100, low=0.0, upp=1.0)
            
            # Plot the histogram
            bin_edges = torch.linspace(0, 1, 101).cpu().numpy()
            axs[i].bar(bin_edges[:-1], hist.cpu().numpy(), width=(bin_edges[1] - bin_edges[0]), color='#C4D9F0', alpha=0.7, label=f'Action {i}')
            axs[i].set_xlim(0, 1)
            axs[i].grid(True, linestyle='--', linewidth=0.5)
            axs[i].set_xlabel('Relative Action')
            axs[i].set_ylabel('Count')
            axs[i].legend()

        plt.grid(True)
        plt.close()
        return fig

    def plot_heatmap_blotto(self): 
        if self.env_name == "BlottoWithInclompeteMultiDim":
            test_data = self.env.sample_obs(2**16, self.agent_index)
        else: test_data = torch.linspace(0.009, 1, 2**16, device=self.device).view(-1, 1)
        with torch.no_grad():
            test_actions,_, _ = self.actor.get_action(test_data)
        heatmaps = []
        for i in range(3):
            if self.env_name == "BlottoWithInclompeteMultiDim":
                data = torch.cat((test_data[:, i].view(-1, 1), test_actions[:, i].view(-1, 1)), dim=1)
            else: data = torch.cat((test_data, test_actions[:, i].view(-1, 1)), dim=1)
            hist = torchist.histogramdd(data, bins=[100, 100], low=[0, 0], upp=[1, 1])
            heatmap = hist.cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
            ax.set_xlabel('Object Valuation')
            ax.set_ylabel(f'Action Interval {i}')
            ax.grid(True, linestyle='--', linewidth=0.5)
            fig.colorbar(cax)
            plt.close(fig)
            heatmaps.append(fig)
        return heatmaps
  
#######UNDER construction##########
    def plot_baseDistribution_and_Transformation_fixedObs(self):
        fig_n = []
        for i in [0.2, 0.5, 0.8]:
            test_data = torch.tensor([i], device=self.device).repeat(10000, 1)
            with torch.no_grad():
                gaussian_actions, transformed_actions, action = self.actor.get_action(test_data, 1, True)
            
            # Determine the range based on the actual data
            min_gaussian, max_gaussian = -3, 3#torch.min(gaussian_actions).item(), torch.max(gaussian_actions).item()
            min_transformed, max_transformed = -3, 3 # torch.min(transformed_actions).item(), torch.max(transformed_actions).item()
            min_action, max_action = -3, 3 #torch.min(action).item(), torch.max(action).item()

            num_bins = 100
            
            # Base Distribution
            hist_gaussian = torch.histc(gaussian_actions, bins=num_bins, min=min_gaussian, max=max_gaussian)
            bin_edges_gaussian = torch.linspace(min_gaussian, max_gaussian, num_bins + 1).cpu().numpy()

            # Transformed Distribution
            hist_transformed = torch.histc(transformed_actions, bins=num_bins, min=min_transformed, max=max_transformed)
            bin_edges_transformed = torch.linspace(min_transformed, max_transformed, num_bins + 1).cpu().numpy()

            # Final Action Distribution
            hist_action = torch.histc(action, bins=num_bins, min=min_action, max=max_action)
            bin_edges_action = torch.linspace(min_action, max_action, num_bins + 1).cpu().numpy()
            
            fig, axs = plt.subplots(3, figsize=(10, 8))
            
            # Plot Base Distribution
            axs[0].bar(bin_edges_gaussian[:-1], hist_gaussian.cpu().numpy(), width=(bin_edges_gaussian[1] - bin_edges_gaussian[0]), color='#C4D9F0', alpha=0.7, label=f'Base Distribution {i}')
            axs[0].set_xlim(min_gaussian, max_gaussian)
            axs[0].grid(True, linestyle='--', linewidth=0.5)
            axs[0].set_xlabel('Action')
            axs[0].set_ylabel('Count')
            axs[0].legend()

            # Plot Transformed Distribution
            axs[1].bar(bin_edges_transformed[:-1], hist_transformed.cpu().numpy(), width=(bin_edges_transformed[1] - bin_edges_transformed[0]), color='#C4D9F0', alpha=0.7, label=f'Transformed Distribution {i}')
            axs[1].set_xlim(min_transformed, max_transformed)
            axs[1].grid(True, linestyle='--', linewidth=0.5)
            axs[1].set_xlabel('Action')
            axs[1].set_ylabel('Count')
            axs[1].legend()

            # Plot Final Action Distribution
            axs[2].bar(bin_edges_action[:-1], hist_action.cpu().numpy(), width=(bin_edges_action[1] - bin_edges_action[0]), color='#C4D9F0', alpha=0.7, label=f'Final Action Distribution {i}')
            axs[2].set_xlim(min_action, max_action)
            axs[2].grid(True, linestyle='--', linewidth=0.5)
            axs[2].set_xlabel('Action')
            axs[2].set_ylabel('Count')
            axs[2].legend()

            plt.grid(True)
            plt.close()
            fig_n.append(fig)
        return fig_n

    def plot_baseDistribution_and_Transformation_heatmap(self):
        # plot a heatmap for each of the three action types showing the distribution of the base and transformed actions
        # only for auction with one dimension
        test_data = torch.linspace(0, 1, 2**17, device=self.device).view(-1, 1)
        with torch.no_grad():
            gaussian_actions, transformed_actions, action = self.actor.get_action(test_data, 1, True)
        
        heatmaps = []
        for i in range(3):
            if i == 0:
                data = torch.cat((test_data, gaussian_actions), dim=1)
                name = 'Base Distribution'
            elif i == 1:
                data = torch.cat((test_data, transformed_actions), dim=1)
                name = 'Transformed Distribution'
            else:
                data = torch.cat((test_data, action), dim=1)
                name = 'Final Action Distribution'
            
            # Determine the min and max dynamically for both dimensions
            # min_test_data, max_test_data = torch.min(test_data).item(), torch.max(test_data).item()
            # min_action_data, max_action_data = torch.min(data[:, 1]).item(), torch.max(data[:, 1]).item()

            hist = torchist.histogramdd(data, bins=[100, 200], low=[0, -2], upp=[1, 2])
            heatmap = hist.cpu().numpy()

            fig, ax = plt.subplots(figsize=(15, 15))
            cax = ax.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest', aspect='auto', extent=[0, 1, -2, 2])
            ax.set_xlabel('Object Valuation')
            ax.set_ylabel(f'Action Interval')
            ax.set_title(name)
            ax.grid(True, linestyle='--', linewidth=0.5)
            fig.colorbar(cax)
            plt.close(fig)
            heatmaps.append(fig)
        
        return heatmaps
    
    # not used anymore
    def scatter_plot_3d_blottoActions_fixedValue(self): 
        # Fixed test data value
        # TODO with a fixed dataset this is not super interesting as it should just focu on one point in the action space
        # What if I just sample from the obs space randomly and still plot the actions together? 
        # And then I could also plot the observations
        test_data = torch.tensor([0.5], device=self.device).repeat(10000, 1)
        
        with torch.no_grad():
            gaussian_actions, transformed_actions, action = self.actor.get_action(test_data, 1, True)

        fig = plt.figure(figsize=(15, 15))

        # Plot Gaussian Actions
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(gaussian_actions[:, 0].cpu().numpy(), gaussian_actions[:, 1].cpu().numpy(), gaussian_actions[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.1)
        ax1.set_title('Gaussian Actions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(gaussian_actions[:, 0].min().item(), gaussian_actions[:, 0].max().item())
        ax1.set_ylim(gaussian_actions[:, 1].min().item(), gaussian_actions[:, 1].max().item())
        ax1.set_zlim(gaussian_actions[:, 2].min().item(), gaussian_actions[:, 2].max().item())

        # Plot Transformed Actions
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(transformed_actions[:, 0].cpu().numpy(), transformed_actions[:, 1].cpu().numpy(), transformed_actions[:, 2].cpu().numpy(), c='r', marker='o', alpha=0.1)
        ax2.set_title('Transformed Actions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim(transformed_actions[:, 0].min().item(), transformed_actions[:, 0].max().item())
        ax2.set_ylim(transformed_actions[:, 1].min().item(), transformed_actions[:, 1].max().item())
        ax2.set_zlim(transformed_actions[:, 2].min().item(), transformed_actions[:, 2].max().item())

        # Plot Final Actions
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(action[:, 0].cpu().numpy(), action[:, 1].cpu().numpy(), action[:, 2].cpu().numpy(), c='g', marker='o', alpha=0.1)
        ax3.set_title('Final Actions')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_xlim(action[:, 0].min().item(), action[:, 0].max().item())
        ax3.set_ylim(action[:, 1].min().item(), action[:, 1].max().item())
        ax3.set_zlim(action[:, 2].min().item(), action[:, 2].max().item())

        plt.tight_layout()
        plt.close(fig)
        return fig   

    def scatter_plot_3d_blottoActions_random(self):
        # Fixed test data value
        if self.env_name == "BlottoWithInclompeteMultiDim":
            test_data = self.env.sample_obs(2**18, self.agent_index)
        else: test_data = torch.linspace(0.009, 1, 2**18, device=self.device).view(-1, 1)
        
        with torch.no_grad():
            gaussian_actions, transformed_actions, action = self.actor.get_action(test_data, 1, True)

        fig = plt.figure(figsize=(15, 15))

        # if test data dimension is not 3 we have to repeat it to fit the action dimension
        # for blotto1d this makes the obs plot kinda boring at i
        if test_data.shape[1] != 3:
            test_data = test_data.repeat(1, 3)
        # Plot observations
        ax0 = fig.add_subplot(141, projection='3d')
        ax0.scatter(test_data[:, 0].cpu().numpy(), test_data[:, 1].cpu().numpy(), test_data[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
        ax0.set_title('Observations')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        ax0.set_xlim(0,1)
        ax0.set_ylim(0,1)
        ax0.set_zlim(0,1)

        # Plot Gaussian Actions
        ax1 = fig.add_subplot(142, projection='3d')
        ax1.scatter(gaussian_actions[:, 0].cpu().numpy(), gaussian_actions[:, 1].cpu().numpy(), gaussian_actions[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
        ax1.set_title('Gaussian Actions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(gaussian_actions[:, 0].min().item(), gaussian_actions[:, 0].max().item())
        ax1.set_ylim(gaussian_actions[:, 1].min().item(), gaussian_actions[:, 1].max().item())
        ax1.set_zlim(gaussian_actions[:, 2].min().item(), gaussian_actions[:, 2].max().item())

        # Plot Transformed Actions
        ax2 = fig.add_subplot(143, projection='3d')
        ax2.scatter(transformed_actions[:, 0].cpu().numpy(), transformed_actions[:, 1].cpu().numpy(), transformed_actions[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
        ax2.set_title('Transformed Actions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim(transformed_actions[:, 0].min().item(), transformed_actions[:, 0].max().item())
        ax2.set_ylim(transformed_actions[:, 1].min().item(), transformed_actions[:, 1].max().item())
        ax2.set_zlim(transformed_actions[:, 2].min().item(), transformed_actions[:, 2].max().item())

        # Plot Final Actions
        ax3 = fig.add_subplot(144, projection='3d')
        ax3.scatter(action[:, 0].cpu().numpy(), action[:, 1].cpu().numpy(), action[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
        ax3.set_title('Final Actions')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_xlim(0,1)
        ax3.set_ylim(0,1)
        ax3.set_zlim(0,1)

        if self.env_name == "BlottoWithInclompeteMultiDim":
            # # Create a meshgrid for the plane x + y + z = 1
            # x = np.linspace(0, 1, 10)
            # y = np.linspace(0, 1, 10)
            # X, Y = np.meshgrid(x, y)
            # Z = 1 - X - Y
            # # Plot the plane on ax3
            # ax3.plot_surface(X, Y, Z, color='c', alpha=0.3, rstride=100, cstride=100)
            none = None
        else: 
            line_start = [0, 0, 0]
            line_end = [0.05, 0.25, 0.7]
            ax3.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]], color='c')

        plt.tight_layout() 
        plt.close(fig)
        return fig
    

    def scatter_plot_3d_blottoActions_random_singleFigures(self):
        # Fixed test data value
        if self.env_name == "BlottoWithInclompeteMultiDim":
            test_data = self.env.sample_obs(2**18, self.agent_index)
        else: 
            test_data = torch.linspace(0.009, 1, 2**18, device=self.device).view(-1, 1)
        
        with torch.no_grad():
            gaussian_actions, transformed_actions, action = self.actor.get_action(test_data, 1, True)
            equilbrium_action = self.env.get_equilibrium_action(test_data)

        # if test data dimension is not 3 we have to repeat it to fit the action dimension
        if test_data.shape[1] != 3:
            test_data = test_data.repeat(1, 3)
        
        # Create a list to hold the figures
        figs = []
        # Create individual figures
        for i, (data, title) in enumerate(zip(
                [test_data, gaussian_actions, transformed_actions, action, equilbrium_action], 
                ['Observations', 'Gaussian Actions', 'Transformed Actions', 'Final Actions', 'Equilibrium Actions'])):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), data[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
            if i in [3, 5]:
                ax.set_xlabel("a1", fontsize=20)
                ax.set_ylabel("a2", fontsize=20)
                ax.set_zlabel("a3", fontsize=20)
            elif i in [0]:
                ax.set_xlabel("v1", fontsize=20)
                ax.set_ylabel("v2", fontsize=20)
                ax.set_zlabel("v3", fontsize=20)
            elif i in [1]:
                ax.set_xlabel("u1", fontsize=20)
                ax.set_ylabel("u2", fontsize=20)
                ax.set_zlabel("u3", fontsize=20)
            elif i in [2]:
                ax.set_xlabel("x1", fontsize=20)
                ax.set_ylabel("x2", fontsize=20)
                ax.set_zlabel("x3", fontsize=20)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            if i in [0, 3, 4]:  # test_data, action, equilibrium_action
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
            else:
                ax.set_xlim(auto=True)
                ax.set_ylim(auto=True)
                ax.set_zlim(auto=True)
                
            plt.close(fig)
            figs.append(fig)

        return figs
    
    
    
    def scatter_plot_3d_blottoMeanActionsGaussian_random(self):
        # Fixed test data value
        if self.env_name == "BlottoWithInclompeteMultiDim":
            test_data = self.env.sample_obs(2**17, self.agent_index)
        else: test_data = torch.linspace(0.009, 1, 2**17, device=self.device).view(-1, 1)
        
        with torch.no_grad():
            action = self.actor.get_mean_action(test_data)
            # action = self.env.get_equilibrium_action(test_data)

        fig = plt.figure(figsize=(15, 15))

        # if test data dimension is not 3 we have to repeat it to fit the action dimension
        # for blotto1d this makes the obs plot kinda boring at i
        if test_data.shape[1] != 3:
            test_data = test_data.repeat(1, 3)
        # Plot observations
        ax0 = fig.add_subplot(121, projection='3d')
        ax0.scatter(test_data[:, 0].cpu().numpy(), test_data[:, 1].cpu().numpy(), test_data[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
        ax0.set_title('Observations')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        ax0.set_xlim(0,1)
        ax0.set_ylim(0,1)
        ax0.set_zlim(0,1)

        # Plot Final Actions
        ax3 = fig.add_subplot(122, projection='3d')
        ax3.scatter(action[:, 0].cpu().numpy(), action[:, 1].cpu().numpy(), action[:, 2].cpu().numpy(), c='b', marker='o', alpha=0.03, s=0.05)
        ax3.set_title('Final Actions')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_xlim(0,1)
        ax3.set_ylim(0,1)
        ax3.set_zlim(0,1)
         # Draw a line from (0, 0, 0) to (0.05, 0.25, 0.7)

        # if self.env_name == "BlottoWithInclompeteMultiDim":
        #     # Create a meshgrid for the plane x + y + z = 1
        #     x = np.linspace(0, 1, 10)
        #     y = np.linspace(0, 1, 10)
        #     X, Y = np.meshgrid(x, y)
        #     Z = 1 - X - Y
        #     # Plot the plane on ax3
        #     ax3.plot_surface(X, Y, Z, color='c', alpha=0.3, rstride=100, cstride=100)
        # else: 
        #     line_start = [0, 0, 0]
        #     line_end = [0.05, 0.25, 0.7]
        #     ax3.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]], color='r')


        plt.tight_layout() 
        plt.close(fig)
        return fig