from abc import abstractmethod

import gymnasium as gym

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.env_observation_builder import ObservationBuilder


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

    def reset(self):
        self.wrap.reset()

    def get(self, handle: int = 0):
        return self.wrap.get(handle)

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        """
        Takes in agent and returns the observation space for that agent.
        """
        return self.observation_space


# # TODO wrap - keep core clean from gym dependency?!
# class RailEnvSpace(gym.spaces.Space):
#     def __init__(
#         self,
#         obs_builder: ObservationBuilder,
#         shape: Sequence[int] | None = None,
#         dtype: npt.DTypeLike | None = None,
#         seed: int | np.random.Generator | None = None,
#
#     ):
#         super().__init__(shape, dtype, seed)
#         self.obs_builder = obs_builder
#
#     def sample(self, mask: Any | None = None):
#         # get = self.obs_builder.get()
#         get = self.obs_builder.get_many(self.obs_builder.env.get_agent_handles())
#         return get
#
#     def contains(self, x: Any) -> bool:
#         return True
#
#         # N.B.
#         self.observation_space = gym.spaces.Tuple(spaces=[
#             # transition map
#             RailEnvSpace(self, shape=(self.env.height, self.env.width, 16), dtype=np.float64),
#             # obs_agents_state
#             RailEnvSpace(self, shape=(self.env.height, self.env.width, 5), dtype=np.float64),
#             # obs_targets
#             RailEnvSpace(self, shape=(self.env.height, self.env.width, 2), dtype=np.float64)
#         ])

def dummy_observation_builder_wrapper(dummy_observation_builder: DummyObservationBuilder) -> GymObservationBuilderWrapper:
    return GymObservationBuilderWrapper(dummy_observation_builder, gym.spaces.Discrete(2))
