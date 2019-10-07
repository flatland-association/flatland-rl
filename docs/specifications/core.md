## Core Specifications

### Environment Class Overview

The Environment class contains all necessary functions for the interactions between the agents and the environment. The base Environment class is derived from rllib.env.MultiAgentEnv (https://github.com/ray-project/ray).

The functions are specific for each realization of Flatland (e.g. Railway, Vaccination,...)
In particular, we retain the rllib interface in the use of the step() function, that accepts a dictionary of actions indexed by the agents handles (returned by get_agent_handles()) and returns dictionaries of observations, dones and infos.

```python
class Environment:
    """Base interface for multi-agent environments in Flatland.

    Agents are identified by agent ids (handles).
    Examples:
        >>> obs, info = env.reset()
        >>> print(obs)
        {
            "train_0": [2.4, 1.6],
            "train_1": [3.4, -3.2],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
                "train_0": 1, "train_1": 0})
        >>> print(rewards)
        {
            "train_0": 3,
            "train_1": -1,
        }
        >>> print(dones)
        {
            "train_0": False,    # train_0 is still running
            "train_1": True,     # train_1 is done
            "__all__": False,    # the env is not done
        }
        >>> print(infos)
        {
            "train_0": {},  # info for train_0
            "train_1": {},  # info for train_1
        }
    """

    def __init__(self):
        pass

    def reset(self):
        """
        Resets the env and returns observations from agents in the environment.

        Returns:
        obs : dict
            New observations for each agent.
        """
        raise NotImplementedError()

    def step(self, action_dict):
        """
        Performs an environment step with simultaneous execution of actions for
        agents in action_dict.
        Returns observations from agents in the environment.
        The returns are dicts mapping from agent_id strings to values.

        Parameters
        -------
        action_dict : dict
            Dictionary of actions to execute, indexed by agent id.

        Returns
        -------
        obs : dict
            New observations for each ready agent.
        rewards: dict
            Reward values for each ready agent.
        dones : dict
            Done values for each ready agent. The special key "__all__"
            (required) is used to indicate env termination.
        infos : dict
            Optional info values for each agent id.
        """
        raise NotImplementedError()

    def render(self):
        """
        Perform rendering of the environment.
        """
        raise NotImplementedError()

    def get_agent_handles(self):
        """
        Returns a list of agents' handles to be used as keys in the step()
        function.
        """
        raise NotImplementedError()

```
