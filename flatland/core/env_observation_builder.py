"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset(env) and get() or get(handle).

+ `reset(env)` is called after each environment reset, to allow for pre-computing relevant data. It receives the
  (possibly newly generated) env instance, so any instantiations depending on env parameters (e.g. width, height)
  should be done here rather than in `__init__`.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""
from typing import Optional, List, Dict, Generic, TypeVar

import numpy as np
from numpy.random import RandomState

from flatland.core.env import Environment

EnvT = TypeVar('EnvT')
ObservationT = TypeVar('ObservationT')
AgentHandle = int


class ObservationBuilder(Generic[EnvT, ObservationT]):
    """
    ObservationBuilder base class.
    """

    def __init__(self):
        self.env: Optional[EnvT] = None

    def reset(self, env: EnvT):
        """
        Called after each environment reset, to allow for pre-computing relevant data. Receives the
        (possibly newly generated) env instance, so any instantiations depending on env parameters
        (e.g. width, height) should be made here rather than in `__init__`.

        Subclasses that need to pre-compute env-dependent data should override this method and call
        `super().reset(env)` first to keep `self.env` up to date.

        Parameters
        ----------
        env : EnvT
            the (possibly newly generated) environment instance
        """
        self.env: EnvT = env

    def get_many(self, handles: Optional[List[AgentHandle]] = None) -> Dict[AgentHandle, ObservationT]:
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

    def get(self, handle: AgentHandle = 0) -> ObservationT:
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
        """Returns the agent's direction to one-hot encoding."""
        direction = np.zeros(4)
        direction[agent.current_entry_point[1]] = 1
        return direction


class DummyObservationBuilder(ObservationBuilder[Environment, bool]):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def get(self, handle: AgentHandle = 0) -> bool:
        return True


def gauss_perturbation_observation_builder_wrapper(
    builder: ObservationBuilder[Environment, np.ndarray], np_random: RandomState, mu: np.ndarray = None, sigma: np.ndarray = None
) -> ObservationBuilder[Environment, np.ndarray]:
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

    class _GaussPerturbationObservationBuilder(ObservationBuilder[Environment, np.ndarray]):
        def __init__(self, builder: ObservationBuilder[Environment, np.ndarray], mu: np.ndarray = None, sigma: np.ndarray = None):
            super().__init__()
            self._mu = mu if mu is not None else 0
            self._sigma = sigma if sigma is not None else 1
            self._builder = builder
            self._np_random = np_random

        def reset(self, env: Environment):
            super().reset(env)
            builder.reset(env)

        def get(self, handle: AgentHandle = 0) -> ObservationBuilder[Environment, np.ndarray]:
            obs: np.ndarray = self._builder.get(handle)
            return obs + self._np_random.normal(self._mu, self._sigma, obs.shape)

    return _GaussPerturbationObservationBuilder(builder)
