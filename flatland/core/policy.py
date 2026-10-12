from abc import ABC
from typing import List, Dict, TypeVar, Generic

from flatland.core.env import Environment

EnvironmentT = TypeVar('EnvironmentT', bound=Environment)
ObsT = TypeVar('ObsT', covariant=True)
ActT = TypeVar('ActT', covariant=True)


class Policy(ABC, Generic[EnvironmentT, ObsT, ActT]):
    """
    Abstract base class for Flatland policies. Used for evaluation.

    Loosely corresponding to https://github.com/ray-project/ray/blob/master/rllib/core/rl_module/rl_module.py, but much simpler.
    """

    def act(self, observation: ObsT, **kwargs) -> ActT:
        """
        Get action for agent. Called by `act_many()` for each agent.

        Parameters
        ----------
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

    def act_many(self, handles: List[int], observations: List[ObsT], **kwargs) -> Dict[int, ActT]:
        """
        Get action_dict for all agents. Default implementation calls `act()` for each handle in the list.

        Override if you need to initialize before / cleanup after calling `act()` for individual agents.

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
        return {handle: self.act(observations[handle]) for handle in handles}
