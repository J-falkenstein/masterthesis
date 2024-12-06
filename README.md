# Finding mixed strategy equilibria with computable densities using soft-actor critic and normalizing flows
## Intro
This repository contains the code for my Master's thesis that adopts the Soft Actor-Critic (SAC) algorithm to a multi-agent setting with the goal of converging to a Nash Equilibrium in simple repeated games. The approach incorporates Normalizing Flows to model mixed strategies with tractable densities, allowing for detailed analysis of action probabilities. The method is applied to both complete and incomplete information games, where the focus is on computing approximate Nash equilibria in continuous-action environments. Experiments demonstrate the algorithm's ability to efficiently solve for high-quality approximate equilibria in these settings.


## Repo
The repository is loosely based on the SAC implementation of CleanRL ([CleanRL RL Algorithms - SAC](https://docs.cleanrl.dev/rl-algorithms/sac/)). I have split the code into several files:
- `main.py`
- `sac_agents.py`
- `models.py` (contains the actors of the agents, such as `Gaussian_Actor` or `Flow_Actor`)
- `Environment\` (contains the environments, such as `blottoGames`)
- `replay_buffer_tensor.py` (for our own replay buffer implementation)


The code can be invoked with various arguments, but the most important ones for us are:

- `--env_name` (e.g., "FirstPricedAuctionStates")
- `--actor_type` (e.g., "flow")

The `launch.json` file contains configurations for the debugger in VSCode.

## Main File

This file reads arguments, initializes the environment, agents, and the replay buffer. It also contains the training loop for SAC.

## SAC Agent


## Environments 


## Models/Actors 


### Gaussian

### Beta

### Flows
