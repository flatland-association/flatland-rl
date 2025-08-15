"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""
from typing import Optional, List, Dict, Generic, TypeVar

import numpy as np
from numpy.random import RandomState

from flatland.core.env import Environment

ObservationType = TypeVar('ObservationType')
AgentHandle = int


class ObservationBuilder(Generic[ObservationType]):
    """
    ObservationBuilder base class.
    """

    def __init__(self):
        self.env: Optional[Environment] = None

    def set_env(self, env: Environment):
        self.env: Environment = env

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get_many(self, handles: Optional[List[AgentHandle]] = None) -> Dict[AgentHandle, ObservationType]:
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

    def get(self, handle: AgentHandle = 0) -> ObservationType:
        """
        Called whenever an observation has to be computed for the `env` environment, possibly
        for each agent independently (agent id `handle`).

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

    def _get_one_hot_for_agent_direction(self, agent) -> np.ndarray:
        """Retuns the agent's direction to one-hot encoding."""
        direction = np.zeros(4)
        direction[agent.direction] = 1
        return direction


class DummyObservationBuilder(ObservationBuilder[bool]):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: AgentHandle = 0) -> bool:
        return True


def gauss_perturbation_observation_builder_wrapper(
    builder: ObservationBuilder[np.ndarray], np_random: RandomState, mu: np.ndarray = None, sigma: np.ndarray = None
) -> ObservationBuilder[np.ndarray]:
    """
    Perturb a numpy array based observation with Gaussian noise.

    Parameters
    ----------
    builder : ObservationBuilder[np.ndarray]
    np_random : RandomState
    mu : np.ndarray
        mean of appropriate size, defaults to 0
    sigma : np.ndarray
        sigma of appropriate size, defaults to 1


    Returns
    -------
    observation with Gaussian noise added
    """

    class _GaussPeturbationObservationBuilder(ObservationBuilder[np.ndarray]):
        def __init__(self, builder: ObservationBuilder[np.ndarray], mu: np.ndarray = None, sigma: np.ndarray = None):
            super().__init__()
            self._mu = mu if mu is not None else 0
            self._sigma = sigma if sigma is not None else 1
            self._builder = builder
            self._np_random = np_random

        def set_env(self, env: Environment):
            builder.set_env(env)

        def reset(self):
            builder.reset()

        def get(self, handle: AgentHandle = 0) -> ObservationBuilder[np.ndarray]:
            obs: np.ndarray = self._builder.get(handle)
            return obs + self._np_random.normal(self._mu, self._sigma, obs.shape)

    return _GaussPeturbationObservationBuilder(builder)
