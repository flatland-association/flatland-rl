from typing import TypeVar

from flatland.core.policy import Policy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions

RailEnvT = TypeVar('RailEnvT', bound=RailEnv)
ObsT = TypeVar('ObsT', covariant=True)
RailEnvActionsT = TypeVar('RailEnvActionsT', bound=RailEnvActions)


class RailEnvPolicy(Policy[RailEnvT, ObsT, RailEnvActionsT]):
    pass
