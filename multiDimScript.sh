#!/bin/bash

# List of experiments
experiments=(
    # # Baseline flow
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=flow --wandb_group=blottoMultiDflow --num_objects=3"
    # # Alpha flow 
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=flow --alpha=0 --wandb_group=blottoMultiDflowAlpha --num_objects=3"
    #  flow fixe
    #"python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=flow --wandb_group=blottoMultiDflowFixed --num_objects=3 --fix_all_but_one_agent"
    # Baseline Gaussian
    #"python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussianFixed --num_objects=3 --fix_all_but_one_agent"
    # Baseline Gaussian
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian --num_objects=3"
    # # Alpha Gaussian 
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --alpha=0 --wandb_group=blottoMultiDGaussianAlpha --num_objects=3"
    # # Pretrain Gaussian
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --pretrain_identity --wandb_group=blottoMultiDGaussianPretraining --num_objects=3"
    # # Pretrain Flow
    # #"python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --pretrain_identity --wandb_group=blottoMultiDflowPretraining --num_objects=3"
    # # Pretrain with Alpha zero - Gaussian
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --pretrain_identity --alpha=0 --wandb_group=blottoMultiDGaussianPretrainingAlpha --num_objects=3"
    # Pretrain with Alpha zero - Flow
    "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --pretrain_identity --alpha=0 --wandb_group=blottoMultiDflowPretrainingAlpha --num_objects=3"
    # Gaussian for BlottoWithInclompeteMultiDim (20000, 10000, 32, 8, 2)
    #"python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian20000 --num_objects=3 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian1000 --num_objects=3 --nr_games_before_update=1000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian2 --num_objects=3 --nr_updates_before_newGames=2"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian8 --num_objects=3 --nr_updates_before_newGames=8"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.00049 --q_lr=0.0362 --actor_type=gaussian --wandb_group=blottoMultiDGaussian32 --num_objects=3 --nr_updates_before_newGames=32"
    # # Flow for BlottoWithInclompeteMultiDim (20000, 10000, 32, 8, 2)
    # #"python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDflow20000 --num_objects=3 --batch_size=20000 --nr_games_before_update=20000 --buffer_size=20000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDflow1000 --num_objects=3 --nr_games_before_update=1000"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDflow2 --num_objects=3 --nr_updates_before_newGames=2"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDflow8 --num_objects=3 --nr_updates_before_newGames=8"
    # "python main_onPolicy.py --env_name=BlottoWithInclompeteMultiDim --total_timesteps=20000 --policy_lr=0.000401 --q_lr=0.03822 --actor_type=flow --wandb_group=blottoMultiDflow32 --num_objects=3 --nr_updates_before_newGames=32"
)

# GPUs available
gpus=(4 5 6 7)

# List of seeds
seeds=(1 2 3 4)

# List of agents
# TODO adjsut if other agents become free
agents=(2)

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