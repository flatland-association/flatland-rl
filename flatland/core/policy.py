from typing import Any, List, Dict


class Policy:
    """
    Abstract base class for Flatland policies. Used for evaluation.

    Loosely corresponding to https://github.com/ray-project/ray/blob/master/rllib/core/rl_module/rl_module.py, but much simpler.
    """

    def act(self, handle: int, observation: List[Any], **kwargs) -> Any:
        """
        Get action for agent. Called by `act_many()` for each agent.

        Parameters
        ----------
        handle: int
            the agent's handle
        observation: Any
            the agent's observation
        kwargs
            forward compatibility placeholder
        Returns
        -------
        Any
            the action dict

        """
        raise NotImplementedError()

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, Any]:
        """
        Get action_dict for all agents. Default implementation calls `act()` for each handle in the list.

        Parameters
        ----------
        handles: List[int]
            the agents' handles
        observations: List[Any]
            the agents' observations
        kwargs
            forward compatibility placeholder
        Returns
        -------
        Dict[int, Any]
            the action dict
        """
        return {handle: self.act(handle, observations[handle]) for handle in handles}
