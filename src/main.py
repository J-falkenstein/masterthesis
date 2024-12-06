import math
import random
import time
import numpy as np
import torch
import tyro
from sac_agent import SAC_Agent
from Environments.simple_auctions import FirstPricedAuctionStates, AllPayAuctionStates
from Environments.blotto_games import BlottoWithInclompete
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
                wandb.log({f"game/value_{i}": env.values_objects[i]})
    else:
        raise ValueError(f"Env name {args.env_name} not supported")
    return env


if __name__ == "__main__":
    start_time_beginnning = time.time()
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "mps" if args.mps and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = f"{args.exp_name}_{device}_{args.env_name}_{args.actor_type}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
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
        rb_n.append(ReplayBufferTest(args.buffer_size, args.num_objects, device))
        if not args.shared_agent: 
            agent_n.append(SAC_Agent(env, i, device, args))
            if args.pretrain_identity: plt_path, test_mean = agent_n[i].pretrain_identity()
    if args.shared_agent: 
        agent = SAC_Agent(env, 0, device, args)
        if args.pretrain_identity: plt_path, test_mean = agent.pretrain_identity()
        agent_n = [agent for _ in range(args.num_agents)]
    
    if args.pretrain_identity and args.track:
        wandb.log({"pretraining/pretrain_plot": wandb.Image(plt_path), "pretraining/pretrain_test_mean": test_mean})

    # TRY NOT TO MODIFY: start the game
    obs_n = env.reset()

    # Training Loop
    for global_step in range(args.total_timesteps):
        action_n = []
        for i in range(args.num_agents):
            if global_step < args.learning_starts:
                action_n.append(torch.tensor(env.action_space[i].sample(), dtype=torch.float32, device=device))  
            else:
                # sequeeze the action to remove the batch dimension
                action_n.append(agent_n[i].select_action(obs_n[i]).squeeze(0))

        # after the step the actions are rescaled to the budget
        next_obs_n, reward_n, _, _, _ = env.step(action_n)

        if global_step % 400 == 0 and args.track:
            average_validation_l2_loss = 0
            for i in range(args.num_agents):
                wandb.log({f"agent_{i}/charts/reward": reward_n[i]}, step=global_step)
                average_validation_l2_loss += agent_n[i].calculate_l2_loss()
                wandb.log({f"agent_{i}/losses/l2_loss": agent_n[i].calculate_l2_loss()}, step=global_step)
                for j in range(args.num_objects):
                    wandb.log({f"agent_{i}/charts/action_{j}": action_n[i][j]}, step=global_step)
                    wandb.log({f"agent_{i}/charts/action_{j}_relative": action_n[i][j]/obs_n[i]}, step=global_step)
                for obs in [0.01, 0.2, 0.3, 0.5, 0.7, 0.8, 0.99]:
                    if args.num_objects == 1:
                        wandb.log({f"agent_{i}/charts/sample_action_for_{obs}": agent_n[i].select_action(torch.tensor([obs], device=device))}, step=global_step)
            wandb.log({"average_validation_l2_loss": average_validation_l2_loss/args.num_agents}, step=global_step)
            if average_validation_l2_loss/args.num_agents < 0.0001: 
                print("Early stopping, average l2 loss below 0.0001")
                if args.actor_type == "flow" and not args.use_gaussian_in_flow:
                    # TODO also do for other envs and actors (make function with logging from below)
                    strategy_plot = agent_n[i].plot_deterministic_policy()
                    wandb.log({f"agent_{i}/strategy_plot/strategy_plot_step_{global_step}": wandb.Image(strategy_plot)})
                break
            

        for i in range(args.num_agents):
            rb_n[i].add(obs_n[i], next_obs_n[i], action_n[i], reward_n[i])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs_n = next_obs_n

         # ALGO LOGIC: training, after the exploration start phase
        if global_step > args.learning_starts:
            for i in range(args.num_agents):

                table_data = rb_n[i].sample(args.batch_size)

                alpha, alpha_loss, qf_loss, actor_loss, qf1_loss,\
                    qf2_loss, qf1_a_values, qf2_a_values = agent_n[i].update_parameters(table_data, global_step)                


                # todo only log if actro loss and alpha loss are not none
                if global_step % 400 == 0 and args.track:
                    wandb.log({f"agent_{i}/losses/qf1_values": qf1_a_values.mean().item(),
                            f"agent_{i}/losses/qf2_values": qf2_a_values.mean().item(),
                            f"agent_{i}/losses/qf1_loss": qf1_loss.item(),
                            f"agent_{i}/losses/qf2_loss": qf2_loss.item(),
                            f"agent_{i}/losses/qf_loss": qf_loss.item() / 2.0,
                            f"agent_{i}/losses/alpha": alpha,
                            f"agent_{i}/charts/SPS": int(global_step / (time.time() - start_time_beginnning))}, step=global_step)
                    print("SPS:", int(global_step / (time.time() - start_time_beginnning)))
                    if actor_loss is not None:
                        wandb.log({f"agent_{i}/losses/actor_loss": actor_loss.item()}, step=global_step)
                    if args.autotune and alpha_loss is not None:
                        wandb.log({f"agent_{i}/losses/alpha_loss": alpha_loss.item()}, step=global_step)

                if args.track and (global_step % 25000 == 0 or global_step == args.learning_starts+1 or global_step == args.total_timesteps-1):
                    # TODO define track/visualiziation for the other cases: multi object games, and non stochastic policies 
                    if (args.actor_type != "flow" or args.use_gaussian_in_flow) and np.prod(env.action_space[i].shape)==1:
                        heatmap_fig = agent_n[i].plot_heatmap()
                        wandb.log({f"agent_{i}/heatmaps/target_distribution_step_{global_step}": wandb.Image(heatmap_fig)})
                        for obs in [0.2, 0.5, 0.8]:
                            # TODO instead of for loop use tensor batch with three observations
                            sample, prob = agent_n[i].plot_target(torch.tensor([obs], dtype=torch.float32, device=device))
                            table_data = [[x, y] for (x, y) in zip(sample, prob)]
                            table = wandb.Table(data=table_data, columns = ["sample", "prob"])
                            wandb.log({f"agent_{i}/charts/target_Distribution_v{obs}_resclaed": wandb.plot.scatter(table, "sample", "prob", title=f"Valuation {obs}")})               
                    elif args.actor_type == "flow" and not args.use_gaussian_in_flow:
                        strategy_plot = agent_n[i].plot_deterministic_policy()
                        wandb.log({f"agent_{i}/strategy_plot/strategy_plot_step_{global_step}": wandb.Image(strategy_plot)})
                    else: 
                        # we have multi object games
                        print("Heatmap and target distribution not supported for this actor type")
    env.close()