# Master Thesis

The repository is based on the SAC implementation of CleanRL ([CleanRL RL Algorithms - SAC](https://docs.cleanrl.dev/rl-algorithms/sac/)). We have split the code into several files:

- `main.py`
- `sac_agents.py`
- `models.py` (contains the actors of the agents, such as `Gaussian_Actor` or `Flow_Actor`)
- `Environment` folder
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

We utilize the `normflows` package from [here](https://github.com/VincentStimper/normalizing-flows).

Adjusted the `normflows` package by adding the following function in `core.py`:

```python
def log_prob_noBaseProb(self, x):
    """Get log probability for batch but disregard the base distribution probability.
    So we can add the base distribution probability later on.

    Args:
      x: Batch

    Returns:
      log probability without the base distribution probability
    """
    log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    z = x
    for i in range(len(self.flows) - 1, -1, -1):
        z, log_det = self.flows[i].inverse(z)
        log_q += log_det
    # log_q += self.q0.log_prob(z)
    # could return Z to compare to input. 
    return log_q, z
```

This enables to calculate the log probability of the flow without the base distribution because we use our own parameterized base distribution.
