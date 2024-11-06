from abc import abstractmethod

import gymnasium as gym
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv


class GymObservationBuilder(ObservationBuilder):
    @abstractmethod
    def get_observation_space(self, handle: int = 0) -> gym.Space:
        """
        Takes in agent and returns the observation space for that agent.
        """
        raise NotImplementedError()


class GymObservationBuilderWrapper(GymObservationBuilder):
    def __init__(self, wrap: ObservationBuilder, observation_space: gym.Space):
        super().__init__()
        self.wrap = wrap
        self.observation_space = observation_space

    def set_env(self, env: Environment):
        self.wrap.set_env(env)

    def reset(self):
        self.wrap.reset()

    def get(self, handle: int = 0):
        return self.wrap.get(handle)

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        """
        Takes in agent and returns the observation space for that agent.
        """
        return self.observation_space


class DummyObservationBuilderGym(GymObservationBuilderWrapper):
    def __init__(self):
        # TODO is there no standard MultiAgentEnv-compatabile Env-Flattening wrapper?
        # workaround for multi-agent setting (i.e. do not flatten agent dict, only flatten per-agent observations)
        self.unflattened_observation_space = gym.spaces.Discrete(2)
        super().__init__(DummyObservationBuilder(), gym.spaces.utils.flatten_space(self.unflattened_observation_space))

    def get(self, handle: int = 0):
        return gym.spaces.utils.flatten(self.unflattened_observation_space, super().get(handle)).astype(float)


# TODO passive_env_checker.py:164: UserWarning: WARN: The obs returned by the `reset()` method was expecting numpy array dtype to be float32, actual type: float64
class GlobalObsForRailEnvGym(GymObservationBuilderWrapper):

    def __init__(self):
        super().__init__(GlobalObsForRailEnv(), None)
        self.observation_space = None

    def set_env(self, env: RailEnv):
        super().set_env(env)
        self._update_observation_space(env)

    def _update_observation_space(self, env):
        # TODO is there no standard MultiAgentEnv-compatabile Env-Flattening wrapper?
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
