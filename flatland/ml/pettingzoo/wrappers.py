from flatland.envs.rail_env import RailEnv
from flatland.ml.pettingzoo.pettingzoo_parallel_rail_env import PettingZooParallelEnvWrapper


class PettingzooFlatland:
    def __init__(self, wrap: RailEnv):
        assert hasattr(wrap.obs_builder, "get_observation_space"), f"{type(wrap.obs_builder)} is not gym-compatible, missing get_observation_space"
        self.wrap = wrap

    def parallel_env(self, **kwargs):
        return PettingZooParallelEnvWrapper(self.wrap, **kwargs)
