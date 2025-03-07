from abc import abstractmethod
from typing import Generic

import gymnasium as gym
import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from flatland.core.env import Environment
from flatland.core.env_observation_builder import DummyObservationBuilder, AgentHandle, ObservationType
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv


class GymObservationBuilder(Generic[ObservationType], ObservationBuilder[ObservationType]):
    """
    Adds `observation_space` method to `ObservationBuilder`.
    """

    @abstractmethod
    def get_observation_space(self, handle: int = 0) -> gym.Space:
        """
        Takes in agent and returns the observation space for that (single) agent.
        """
        raise NotImplementedError()


class GymObservationBuilderWrapper(GymObservationBuilder):
    """
    Wraps an existing `ObservationBuilder` into a `GymObservationBuilder`.
    """

    def __init__(self, wrap: ObservationBuilder, observation_space: gym.Space):
        super().__init__()
        self.wrap = wrap
        self.observation_space = observation_space

    def set_env(self, env: Environment):
        self.wrap.set_env(env)

    def reset(self):
        self.wrap.reset()

    def get(self, handle: AgentHandle = 0) -> MultiAgentDict:
        return self.wrap.get(handle)

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        """
        Takes in agent and returns the observation space for that (single) agent.
        """
        return self.observation_space


class DummyObservationBuilderGym(GymObservationBuilderWrapper):
    """
    Gym-ified multi-agent `DummyObservationBuilder`.
    """

    def __init__(self):
        # workaround for multi-agent setting (i.e. do not flatten agent dict, only flatten per-agent observations)
        self.unflattened_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.utils.flatten_space(self.unflattened_observation_space)
        super().__init__(DummyObservationBuilder(), self.observation_space)

    def get(self, handle: AgentHandle = 0):
        # `flatten` converts bool to float as float is observation space's dtype
        return gym.spaces.utils.flatten(self.unflattened_observation_space, super().get(handle))


class GlobalObsForRailEnvGym(GymObservationBuilderWrapper):
    """
    Gym-ified multi-agent `GlobalObsForRailEnv`.
    """

    def __init__(self):
        super().__init__(GlobalObsForRailEnv(), None)
        self.observation_space = None

    def set_env(self, env: RailEnv):
        super().set_env(env)
        self._update_observation_space(env)

    def _update_observation_space(self, env):
        # workaround for multi-agent setting (i.e. do not flatten agent dict, only flatten per-agent observations)
        self.unflattened_observation_space = gym.spaces.Tuple(spaces=[
            # transition map
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.height, env.width, 16), dtype=float),
            # obs_agents_state
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.height, env.width, 5), dtype=float),
            # obs_targets
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.height, env.width, 2), dtype=float)
        ])
        self.observation_space = gym.spaces.flatten_space(self.unflattened_observation_space)

    def get(self, handle: int = 0):
        return gym.spaces.utils.flatten(self.unflattened_observation_space, super().get(handle))

    def reset(self):
        super().reset()
        self._update_observation_space(self.wrap.env)
