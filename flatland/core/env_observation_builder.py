"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""
from typing import Optional, List

import numpy as np

from flatland.core.env import Environment


class ObservationBuilder:
    """
    ObservationBuilder base class.
    """

    def __init__(self, standardized_observation: Optional[bool] = False):
        self.env = None
        self.standardize = standardized_observation

    def set_env(self, env: Environment):
        self.env: Environment = env

    def set_standardize_observation(self, standardized_observation: bool):
        self.standardize = standardized_observation

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get_many(self, handles: Optional[List[int]] = None):
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.

        Parameters
        ----------
        handles : list of handles, optional
            List with the handles of the agents for which to compute the observation vector.

        Returns
        -------
        function
            A dictionary of observation structures, specific to the corresponding environment, with handles from
            `handles` as keys.
        """
        observations = {}
        if handles is None:
            handles = []
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle: int = 0):
        """
        Called whenever an observation has to be computed for the `env` environment, possibly
        for each agent independently (agent id `handle`).

        Can make use of the boolean class variable `standardize` to return the observation either
        in a freely defined structure or in a standardized form (a 1-dimensional array).

        To transform an observation to the standardized form, override the dummy-implementation
        of the class method `_standardize_observation(self, obs)`.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            An observation structure, specific to the corresponding environment.
        """
        raise NotImplementedError()

    def _standardize_observation(self, obs):
        """
        Transforms the observation to a standardized form. Should be called from within the class method `get` in
        case the class variable `standardize` is set to True.

        Parameters
        ----------
        obs : Any
            An observation structure

        Returns
        -------
        function
            The observation flattend to a 1-dimensional array.
        """

        return obs

    def _get_one_hot_for_agent_direction(self, agent):
        """Retuns the agent's direction to one-hot encoding."""
        direction = np.zeros(4)
        direction[agent.direction] = 1
        return direction


class DummyObservationBuilder(ObservationBuilder):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> bool:
        return True

    def get(self, handle: int = 0) -> bool:
        return True
