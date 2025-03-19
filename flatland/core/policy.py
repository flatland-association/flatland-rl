from typing import Any


class Policy:
    """
    Abstract base class for Flatland policies. Used for evaluation.

    Loosely corresponding to https://github.com/ray-project/ray/blob/master/rllib/core/rl_module/rl_module.py, but much simpler.
    """

    def act(self, handle: int, observation: Any, **kwargs) -> Any:
        """

        Parameters
        ----------
        handle: int
            the agent's handle
        observation: Any
            the agent's observation
        kwargs

        Returns
        -------
        Any
            the action dict

        """
        pass
