from typing import TypeVar

from flatland.core.policy import Policy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions

T_env = TypeVar('T_env', bound=RailEnv)
T_obs = TypeVar('T_obs', covariant=True)
T_act = TypeVar('T_act', bound=RailEnvActions)


class RailEnvPolicy(Policy[T_env, T_obs, T_act]):
    pass
