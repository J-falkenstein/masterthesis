#!/bin/bash

# List of experiments
experiments=(
    #gaussian with fixed alpha 0.0025 and auto decay
    "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussianFixedAlpha25 --alpha=0.0025"
    "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussianFixedAlpha25 --alpha=0.0025"
    "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --num_objects=3 --wandb_group=blottoGaussianFixedAlpha25 --alpha=0.0025"
    "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=gaussian --num_objects=3 --wandb_group=blottoMultiDGaussianFixedAlpha25 --alpha=0.0025"
    
    "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussianFixedAlphaDecay --alpha=0.25 --decay_alpha"
    "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussianFixedAlphaDecay --alpha=0.25 --decay_alpha"
    "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --num_objects=3 --wandb_group=blottoGaussianFixedAlphaDecay --alpha=0.25 --decay_alpha"
    "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=gaussian --num_objects=3 --wandb_group=blottoMultiDGaussianFixedAlphaDecay --alpha=0.25 --decay_alpha"
    
    #Flow with fixed alpha 0.0025 and auto decay
    "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpFlowFixedAlpha25 --alpha=0.0025"
    "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apFlowFixedAlpha25 --alpha=0.0025"
    "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoFlowFixedAlpha25 --num_objects=3 --alpha=0.0025"
    "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDFlowFixedAlpha25 --num_objects=3 --alpha=0.0025"

    "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpFlowFixedAlphaDecay --alpha=0.25 --decay_alpha"
    "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apFlowFixedAlphaDecay --alpha=0.25 --decay_alpha"
    "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoFlowFixedAlphaDecay --num_objects=3 --alpha=0.25 --decay_alpha"
    "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDFlowFixedAlphaDecay --num_objects=3 --alpha=0.25 --decay_alpha"
    
    
    
    #Flow with fixed alpha 0.001
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpFlowFixedAlpha --alpha=0.001"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apFlowFixedAlpha --alpha=0.001"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoFlowFixedAlpha --num_objects=3 --alpha=0.001"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDFlowFixedAlpha --num_objects=3 --alpha=0.001"

    
    #nr og Games 20000 - FP and AP done for seed 1 to 3
    #FLows
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpflow20000 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apflow20000 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoflow20000 --num_objects=3 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDflow20000 --num_objects=3 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # Gaussian
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussian20000 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussian20000 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    #"python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian20000 --num_objects=3 --batch_size=10000 --nr_games_before_update=10000 --buffer_size=10000"
    #"python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --wandb_group=blottoGaussian20000 --num_objects=3 --batch_size=10000 --nr_games_before_update=10000 --buffer_size=10000"
    # nr of Games 1000
    # flows
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpflow1000 --nr_games_before_update=1000"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apflow1000 --nr_games_before_update=1000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoflow1000 --num_objects=3 --nr_games_before_update=1000"
    # # Gaussian
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussian1000 --nr_games_before_update=1000"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussian1000 --nr_games_before_update=1000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --wandb_group=blottoGaussian1000 --num_objects=3 --nr_games_before_update=1000"
    # # nr of updates 2
    # # flows
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpflow2 --nr_updates_before_newGames=2"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apflow2 --nr_updates_before_newGames=2"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoflow2 --num_objects=3 --nr_updates_before_newGames=2"
    # # gaussian
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussian2 --nr_updates_before_newGames=2"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussian2 --nr_updates_before_newGames=2"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --wandb_group=blottoGaussian2 --num_objects=3 --nr_updates_before_newGames=2"
    # # nr of updates 8
    # # flows
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpflow8 --nr_updates_before_newGames=8"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apflow8 --nr_updates_before_newGames=8"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoflow8 --num_objects=3 --nr_updates_before_newGames=8"
    # # gaussian
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussian8 --nr_updates_before_newGames=8"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussian8 --nr_updates_before_newGames=8"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --wandb_group=blottoGaussian8 --num_objects=3 --nr_updates_before_newGames=8"
    # # nr of updates 32
    # # flows
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.00003 --q_lr=0.0662 --actor_type=flow --wandb_group=fpflow32 --nr_updates_before_newGames=32"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000675 --q_lr=0.0894 --actor_type=flow --wandb_group=apflow32 --nr_updates_before_newGames=32"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000457 --q_lr=0.0370 --actor_type=flow --wandb_group=blottoflow32 --num_objects=3 --nr_updates_before_newGames=32"
    # # gaussian
    # "python main_onPolicy.py --env_name=FirstPricedAuctionStates --total_timesteps=20000 --policy_lr=0.000112 --q_lr=0.0934 --actor_type=gaussian --wandb_group=fpGaussian32 --nr_updates_before_newGames=32"
    # "python main_onPolicy.py --env_name=AllPayAuctionStates --total_timesteps=20000 --policy_lr=0.000119 --q_lr=0.0855 --actor_type=gaussian --wandb_group=apGaussian32 --nr_updates_before_newGames=32"
    # "python main_onPolicy.py --env_name=BlottoWithInclompete --total_timesteps=20000 --policy_lr=0.000348 --q_lr=0.03005 --actor_type=gaussian --wandb_group=blottoGaussian32 --num_objects=3 --nr_updates_before_newGames=32"
)

# GPUs available
# old: gpu = {4 5 6}
# now more gpus availabel but 
gpus=(2 3 4 5 6 7)

# List of seeds
seeds=(1 2 3 4 5)

# List of agents
agents=(1 2)

# Create a tmux session for each GPU-agent combination
for gpu in "${gpus[@]}"; do
    for agent in "${agents[@]}"; do
        session_name="gpu${gpu}_agent${agent}"
        tmux new-session -d -s $session_name
        tmux send-keys -t $session_name "cd ~/master_thesis" C-m
        tmux send-keys -t $session_name "source .venvJanek/bin/activate" C-m
        tmux send-keys -t $session_name "cd src" C-m
        tmux send-keys -t $session_name "CUDA_VISIBLE_DEVICES=$gpu" C-m
    done
done

# Queue experiments in tmux sessions
gpu_index=0    
agent_index=0
for seed in "${seeds[@]}"; do
    for exp in "${experiments[@]}"; do
        gpu=${gpus[$gpu_index]}
        agent=${agents[$agent_index]}
        session_name="gpu${gpu}_agent${agent}"
        command="CUDA_VISIBLE_DEVICES=$gpu $exp --seed=$seed; sleep 40s"
        tmux send-keys -t $session_name "$command" C-m
        # Update GPU index for the next experiment
        gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
    done
    # Update agent index for the next seed
    agent_index=$(( (agent_index + 1) % ${#agents[@]} ))
done