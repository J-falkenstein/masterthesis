from dataclasses import dataclass

@dataclass
class Args:
    # Run setting
    exp_name: str = "sac"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    cuda_device: str = "cuda:7"
    """the device (nr. of GPU) to run the experiment on"""
    mps: bool = False
    """if toggled, mps will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    tracking_frequency: int = 1024
    """the frequency of logging to wandb"""
    wandb_project_name: str = "masterThesis"
    """the wandb's project name"""
    wandb_entity: str = "j-falkenstein"
    """the entity (team) of wandb's project"""
    wandb_group: str = None
    """the group of wandb's project"""

    # Env specific arguments
    num_agents: int = 2
    """number of agents in the environment"""
    env_name: str = "FirstPricedAuctionStates"
    """the environment id"""
    # This has to be set to 1 for auction and to 3 for blotto, other values could raise an error and are not tested
    num_objects: int = 1
    """number of objects/battelfields in the blotte environments"""

    # Actor specific arguments
    actor_type: str = "gaussian"
    """the type of actor model/network to use"""
    shared_agent: bool = False
    """if toggled, all players share the same agent, hence the same actor and critc network"""
    pretrain_identity: bool = False
    """if toggled, the actor will be pre-trained to output the identity (i.e., observation will be the action)"""
    fix_all_but_one_agent: bool = False
    """if toggled, all agents play the equilibrium strategy except one"""
    # Flow settings
    use_gaussian_in_flow: bool = True
    """if toggled, the gaussian distribution will be used as the base dsitribution in the flow actor"""
    nr_flow_transformations: int = 1
    """the number of flow transformations in the flow actor - K"""
    hidden_units: int = 10
    """the number of hidden units in a flow network. A.k.a num_hidden_channels Number of hidden units of the NN"""
    hidden_layers: int = 2
    """the number of layers in a flow network. A.k.a num_blocks: Number of residual blocks of the parameter NN"""

    # Algorithm specific arguments
    nr_games_before_update: int = 1
    """the number of games/steps added to buffer before the agent is updated"""
    nr_updates_before_newGames: int = 1
    """the number of gradient updates before new games/steps are added to the buffer"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 0.00035 
    """the learning rate of the policy network optimizer"""
    q_lr: float = 0.03005 
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.025
    """Entropy regularization coefficient."""
    decay_alpha: bool = False
    # error if decay and autotune are true
    alpha_lambda: float = 2.608e-5 #0.000276073 or 4.60083e-5 or 7.0975e-5 3.26e-5
    """decay rate of the entropy regularization coefficient."""
    autotune: bool =True 
    """automatic tuning of the entropy coefficient"""