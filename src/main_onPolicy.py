import math
import random
import time
import numpy as np
import torch
import tyro
from sac_agent import SAC_Agent
from Environments.simple_auctions import FirstPricedAuctionStates, AllPayAuctionStates
from Environments.blotto_games import BlottoWithInclompete, BlottoWithIncompleteMultiDim
from replay_buffer_tensor import ReplayBufferTensor
from replay_buffer_test import ReplayBufferTest
from args import Args

def create_env(args, device):
    if args.env_name == "FirstPricedAuctionStates":
        env = FirstPricedAuctionStates(args.num_agents, device)
    elif args.env_name == "AllPayAuctionStates":
        env = AllPayAuctionStates(args.num_agents, device)
    elif args.env_name == "BlottoWithInclompete":
        env = BlottoWithInclompete(args.num_agents, args.num_objects, device)
        if args.track:
            for i in range(args.num_objects):
                wandb.log({f"game/value_{i}": env.values_objects[i]}, commit=False)
    elif args.env_name == "BlottoWithInclompeteMultiDim":
        env = BlottoWithIncompleteMultiDim(args.num_agents, args.num_objects, device)
    else:
        raise ValueError(f"Env name {args.env_name} not supported")
    return env

def calculate_all_utility_losses():
    
    nr_samples = 2**16 if global_step != args.total_timesteps-1 else 2**18

    # Initialize lists to store observations, actions, and utility losses for each type
    utility_loss_obs_n = []
    utility_loss_action_general_n = []
    utility_loss_action_meanAction_n = []
    
    # Sample observations and select actions for each agent
    for i in range(args.num_agents):
        obs = env.sample_obs(nr_samples, i)
        utility_loss_obs_n.append(obs)
        utility_loss_action_general_n.append(agent_n[i].select_action(obs))
        #utility_loss_action_currentStrategies_n.append(agent_n[i].select_action(obs))
        if args.actor_type != "flow": utility_loss_action_meanAction_n.append(agent_n[i].select_mean_action(obs))
    
    # Calculate utility losses for general, current strategies, and mean actions
    utility_loss_general_n = env.calculate_utility_loss(utility_loss_action_general_n, utility_loss_obs_n)
    utility_loss_currentStrategies_n = env.calculate_utility_loss_currentStrategies(utility_loss_action_general_n, utility_loss_obs_n)
    if args.actor_type != "flow":  
        utility_loss_meanAction_n = env.calculate_utility_loss(utility_loss_action_meanAction_n, utility_loss_obs_n)
        utility_loss_meanAction_currentStrategies_n = env.calculate_utility_loss_currentStrategies(utility_loss_action_meanAction_n, utility_loss_obs_n)

    # Log individual and average utility losses for each type
    for i in range(args.num_agents):
        wandb.log({f"agent_{i}/losses/utility_loss": utility_loss_general_n[i]}, commit=False, step=global_step)
        wandb.log({f"agent_{i}/losses/utility_loss_currentStrategies": utility_loss_currentStrategies_n[i]}, commit=False, step=global_step)
        if args.actor_type != "flow":  
            wandb.log({f"agent_{i}/losses/utility_loss_meanAction": utility_loss_meanAction_n[i]}, commit=False, step=global_step)
            wandb.log({f"agent_{i}/losses/utility_loss_meanAction_currentStrategies": utility_loss_meanAction_currentStrategies_n[i]}, commit=False, step=global_step)

    average_utility_loss_general = utility_loss_general_n[0] if args.fix_all_but_one_agent else sum(utility_loss_general_n) / args.num_agents
    average_utility_loss_currentStrategies = utility_loss_currentStrategies_n[0] if args.fix_all_but_one_agent else sum(utility_loss_currentStrategies_n) / args.num_agents
    if args.actor_type != "flow": 
        average_utility_loss_meanAction = utility_loss_meanAction_n[0] if args.fix_all_but_one_agent else sum(utility_loss_meanAction_n) / args.num_agents
        average_utility_loss_meanAction_currentStrategies = utility_loss_meanAction_currentStrategies_n[0] if args.fix_all_but_one_agent else sum(utility_loss_meanAction_currentStrategies_n) / args.num_agents
    
    wandb.log({"average_losses/utility_loss": average_utility_loss_general}, commit=False, step=global_step)
    wandb.log({"average_losses/utility_loss_currentStrategies": average_utility_loss_currentStrategies}, commit=False, step=global_step)
    if args.actor_type != "flow": 
        wandb.log({"average_losses/utility_loss_meanAction": average_utility_loss_meanAction}, commit=False, step=global_step)
        wandb.log({"average_losses/utility_loss_meanAction_currentStrategies": average_utility_loss_meanAction_currentStrategies}, commit=False, step=global_step)

def calculate_l2_loss(): 
    total_validation_l2_loss = 0
    validation_l2_loss_0 = 0
    for i in range(args.num_agents):
        validation_l2_loss_i = agent_n[i].calculate_l2_loss()
        if i == 0: validation_l2_loss_0 = validation_l2_loss_i
        total_validation_l2_loss += validation_l2_loss_i
        wandb.log({f"agent_{i}/losses/l2_loss": validation_l2_loss_i}, commit = False, step=global_step)
    
    average_validation_l2_loss = validation_l2_loss_0 if args.fix_all_but_one_agent else total_validation_l2_loss / args.num_agents
    wandb.log({"average_losses/validation_l2_loss": average_validation_l2_loss}, commit = False, step=global_step)
    return average_validation_l2_loss

def calculate_l2_loss_meanAction(): 
    if args.actor_type == "flow": return
    total_validation_l2_loss = 0
    validation_l2_loss_0 = 0
    for i in range(args.num_agents):
        validation_l2_loss_i = agent_n[i].calculate_l2_loss_meanAction()
        if i == 0: validation_l2_loss_0 = validation_l2_loss_i
        total_validation_l2_loss += validation_l2_loss_i
        wandb.log({f"agent_{i}/losses/l2_loss_meanAction": validation_l2_loss_i}, commit = False, step=global_step)
    
    average_validation_l2_loss = validation_l2_loss_0 if args.fix_all_but_one_agent else total_validation_l2_loss / args.num_agents
    wandb.log({"average_losses/validation_l2_loss_meanAction": average_validation_l2_loss}, commit = False, step=global_step)

if __name__ == "__main__":
    start_time_beginnning = time.time()
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "mps" if args.mps and torch.backends.mps.is_available() else "cpu")

    # kinda cheating the sweeps as i have set the values to high but no time restart it 
    # args.total_timesteps = round(args.total_timesteps/args.nr_updates_before_newGames)
    # args.tracking_frequency = round(args.tracking_frequency/args.nr_updates_before_newGames)

    if args.alpha == 0: args.autotune = False

    print(f"Device: {device}")

    # run_name = f"{args.exp_name}_{device}_{args.env_name}_{args.actor_type}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
            group=args.wandb_group,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    env = create_env(args, device)

    agent_n = []
    rb_n = []
    for i in range(args.num_agents):
        rb_n.append(ReplayBufferTest(args.buffer_size, args.env_name, args.num_objects, device))
        if not args.shared_agent: 
            agent_n.append(SAC_Agent(env, args.env_name, i, device, args))
            if args.pretrain_identity: 
                plt_path, test_mean = agent_n[i].pretrain_identity()
                if args.track: 
                    wandb.log({f"pretraining/pretrain_plot{i}": wandb.Image(plt_path), f"pretraining/pretrain_test_mean{i}": test_mean}, commit = False)
    if args.shared_agent:
        agent = SAC_Agent(env, 0, device, args)
        if args.pretrain_identity: 
            plt_path, test_mean = agent.pretrain_identity()
            if args.track: 
                wandb.log({"pretraining/pretrain_plot": wandb.Image(plt_path), "pretraining/pretrain_test_mean": test_mean}, commit = False)

        agent_n = [agent for _ in range(args.num_agents)]

    # initialize obs_n
    obs_n = [0.5 for _ in range(args.num_agents)]

    global_step = 0
    while global_step < args.total_timesteps:

        action_n = []

        for i in range(args.num_agents):
            # TODO: implement random sampling like learning starts in the first global step -> not that important because we dont have to explore hidden trajoctories as we have a one-shot game
            obs_n[i] = env.sample_obs(args.nr_games_before_update, i)
            action_n.append(agent_n[i].select_action(obs_n[i]))
        reward_n = env.get_rewards(action_n, obs_n)
        for i in range(args.num_agents):
            rb_n[i].add(obs_n[i], action_n[i], reward_n[i])

        if args.track and (global_step % (args.tracking_frequency*10) == 0 or global_step == args.learning_starts+1 or global_step == args.total_timesteps-1):
            calculate_all_utility_losses()


        if args.track and (global_step % args.tracking_frequency == 0 or global_step == args.total_timesteps-1):
            print("SPS:", int(global_step / (time.time() - start_time_beginnning)))
            for i in range(args.num_agents):
                # TODO udpate alpha 
                if args.decay_alpha:
                    agent_n[i].update_alpha(global_step)
                wandb.log({f"agent_{i}/charts/reward": torch.mean(reward_n[i])}, commit = False, step=global_step)
                for j in range(args.num_objects):
                    if args.env_name != "BlottoWithInclompeteMultiDim":
                        wandb.log({f"agent_{i}/charts/action_{j}": torch.mean(action_n[i][:,j]), 
                                f"agent_{i}/charts/action_{j}_relative": torch.mean(action_n[i][:,j]/obs_n[i])}, commit = False, step=global_step)
                if args.num_objects == 1:
                    observations = torch.tensor([0.01, 0.2, 0.3, 0.5, 0.7, 0.8, 0.99], device=device).view(-1, 1)
                    actions = agent_n[i].select_action(observations)
                    for obs, action in zip(observations, actions):
                        wandb.log({f"agent_{i}/charts/sample_action_for_{round(obs.item(), 2)}": round(action.item(), 2)}, commit=False, step=global_step)
                average_validation_l2_loss = calculate_l2_loss()
                calculate_l2_loss_meanAction()

            if  average_validation_l2_loss < 0.0001: 
                print("Early stopping, average l2 loss below 0.0001")
                calculate_all_utility_losses()
                for i in range(args.num_agents):
                    if (args.actor_type != "flow" or args.use_gaussian_in_flow) and np.prod(env.action_space[i].shape)==1:
                        tables = agent_n[i].plot_target()
                        for [table_data, obs] in tables:
                            table = wandb.Table(data=table_data, columns = ["sample", "prob"])
                            wandb.log({f"agent_{i}/charts/target_Distribution_v{obs}_resclaed": wandb.plot.scatter(table, "sample", "prob", title=f"Valuation {obs}")}, commit = False)               
                        heatmap_fig = agent_n[i].plot_heatmap()
                        wandb.log({f"agent_{i}/heatmaps/target_distribution_step_{global_step}": wandb.Image(heatmap_fig)}, commit = i==args.num_agents-1 )
                break
    

        # Updating LOGIC:
        for i in range(args.num_agents):

            if args.track and (global_step % args.tracking_frequency == 0 or global_step == args.total_timesteps-1):
                total_alpha, total_alpha_loss, total_qf_loss, total_actor_loss,\
                 total_qf1_loss, total_qf2_loss, total_qf1_a_values, total_qf2_a_values = (0,) * 8

            for _ in range(args.nr_updates_before_newGames):
                table_data = rb_n[i].sample(args.batch_size)
                alpha, alpha_loss, qf_loss, actor_loss, qf1_loss,\
                    qf2_loss, qf1_a_values, qf2_a_values = agent_n[i].update_parameters(table_data, global_step) 
                if args.track and (global_step % args.tracking_frequency == 0 or global_step == args.total_timesteps-1):
                    total_alpha += alpha
                    if alpha_loss is not None: total_alpha_loss += alpha_loss
                    total_qf_loss += qf_loss
                    if actor_loss is not None: total_actor_loss += actor_loss
                    total_qf1_loss += qf1_loss
                    total_qf2_loss += qf2_loss
                    total_qf1_a_values += qf1_a_values.mean()
                    total_qf2_a_values += qf2_a_values.mean()               

            if args.track and (global_step % args.tracking_frequency == 0 or global_step == args.total_timesteps-1):
                avg_alpha = total_alpha / args.nr_updates_before_newGames
                avg_qf_loss = total_qf_loss / args.nr_updates_before_newGames
                if actor_loss is not None:  avg_actor_loss = total_actor_loss / args.nr_updates_before_newGames
                avg_qf1_loss = total_qf1_loss / args.nr_updates_before_newGames
                avg_qf2_loss = total_qf2_loss / args.nr_updates_before_newGames
                avg_qf1_a_values = total_qf1_a_values / args.nr_updates_before_newGames
                avg_qf2_a_values = total_qf2_a_values / args.nr_updates_before_newGames
                if actor_loss is not None:
                    wandb.log({f"agent_{i}/losses/actor_loss": actor_loss.item()}, commit = False, step=global_step)
                if args.autotune and alpha_loss is not None:
                    wandb.log({f"agent_{i}/losses/alpha_loss": alpha_loss.item()}, commit = False, step=global_step)
                wandb.log({f"agent_{i}/losses/qf1_values": avg_qf1_a_values.item(),
                   f"agent_{i}/losses/qf2_values": avg_qf2_a_values.item(),
                   f"agent_{i}/losses/qf1_loss": avg_qf1_loss.item(),
                   f"agent_{i}/losses/qf2_loss": avg_qf2_loss.item(),
                   f"agent_{i}/losses/qf_loss": avg_qf_loss.item() / 2.0,
                   f"agent_{i}/losses/alpha": avg_alpha}, commit = i==args.num_agents-1, step=global_step)

            if args.track and (global_step % (args.tracking_frequency*16) == 0 or global_step == args.learning_starts+1 or global_step == args.total_timesteps-1):
                    if np.prod(env.action_space[i].shape)==1:
                        # TODO its getting dirty as some of these fucntion are not implemented for gaussian and vice versa
                        # fig_n = agent_n[i].plot_baseDistribution_and_Transformation_fixedObs()
                        # for  n, fig in enumerate(fig_n):
                        #     wandb.log({f"agent_{i}/charts/baseDistribution_and_Transformation_fixedObs{n}": wandb.Image(fig)}, commit = False )
                        # heatmap_n = agent_n[i].plot_baseDistribution_and_Transformation_heatmap()
                        # for n, heatmap in enumerate(heatmap_n):
                        #     wandb.log({f"agent_{i}/heatmaps/baseDistribution_and_Transformation_heatmap{n}": wandb.Image(heatmap)}, commit = False )   
                        tables = agent_n[i].plot_target()
                        for [table_data, obs] in tables:
                            table = wandb.Table(data=table_data, columns = ["sample", "prob"])
                            wandb.log({f"agent_{i}/charts/target_Distribution_v{obs}_resclaed": wandb.plot.scatter(table, "sample", "prob", title=f"Valuation {obs}")}, commit = False)               
                        heatmap_fig = agent_n[i].plot_heatmap()
                        wandb.log({f"agent_{i}/heatmaps/target_distribution": wandb.Image(heatmap_fig)}, commit = i==args.num_agents-1 )
                    else: 
                    # Blotto games
                        if args.actor_type == "flow":
                            figs = agent_n[i].scatter_plot_3d_blottoActions_random_singleFigures()
                            for figNumber , fig in enumerate(figs):
                                wandb.log({f"agent_{i}/charts/scatter_plot_3d_{figNumber}": wandb.Image(fig)}, commit = False )
                        else: 
                            fig = agent_n[i].scatter_plot_3d_blottoMeanActionsGaussian_random()
                            wandb.log({f"agent_{i}/charts/scatter_plot_3d_blottoMeanActionsGaussian": wandb.Image(fig)}, commit = False )
                        # if args.env_name == "BlottoWithInclompete":
                        #     fig = agent_n[i].plot_relative_action_blotto()
                        #     wandb.log({f"agent_{i}/charts/relative_action": wandb.Image(fig)}, commit = i==args.num_agents-1 )
                        heatmaps = agent_n[i].plot_heatmap_blotto()
                        for j, heatmap in enumerate(heatmaps):
                            wandb.log({f"agent_{i}/heatmaps/heatmap_{j}": wandb.Image(heatmap)}, commit = False)
        global_step += args.nr_updates_before_newGames
    env.close()