import sys
import warnings
from typing import Tuple, Dict, Optional

from numpy.random import RandomState

from flatland.core.effects_generator import EffectsGenerator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, SparseRailGen, RailGenerator, RailGeneratorProduct
from flatland.envs.rewards import Rewards


# defaults from Flatland 3 Round 2 Test_0, see https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
# Test_00,Level_0,7,30,30,2,2,10,42,False,2,20,50,540,"{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}"
def env_generator(n_agents=7,
                  x_dim=30,
                  y_dim=30,
                  n_cities=2,
                  max_rail_pairs_in_city=4,  # TODO should be 2
                  grid_mode=False,
                  max_rails_between_cities=2,
                  malfunction_duration_min=20,
                  malfunction_duration_max=50,
                  malfunction_interval=540,
                  speed_ratios=None,
                  line_length=2,
                  seed=None,
                  post_seed=None,
                  obs_builder_object=None,
                  acceleration_delta=1.0,
                  braking_delta=-1.0,
                  rewards: Rewards = None,
                  effects_generator: Optional[EffectsGenerator[RailEnv]] = None,
                  ) -> Tuple[RailEnv, Dict, Dict]:
    """
    Create an env with a given spec using `sparse_rail_generator`.
    Defaults are taken from Flatland 3 Round 2 Test_0, see `Environment Configurations <https://flatland.aicrowd.com/challenges/flatland3/envconfig.html`_.
    Parameters name come from `metadata.csv <https://flatland.aicrowd.com/challenges/flatland3/test-submissions-local.html>`_ in `debug-environments.zip <https://www.aicrowd.com/challenges/flatland-3/dataset_files>`_

    Parameters
    ----------
    n_agents: int
        number of agents
    x_dim: int
        number of columns
    y_dim: int
        number of rows
    n_cities: int
       Max number of cities to build. The generator tries to achieve this numbers given all the parameters. Goes into `sparse_rail_generator`.
    max_rail_pairs_in_city: int
        Number of parallel tracks in the city. This represents the number of tracks in the train stations. Goes into `sparse_rail_generator`.
    grid_mode: bool
        How to distribute the cities in the path, either equally in a grid or random. Goes into `sparse_rail_generator`.
    max_rails_between_cities: int
        Max number of rails connecting to a city. This is only the number of connection points at city boarder.
    malfunction_duration_min: int
        Minimal duration of malfunction. Goes into `ParamMalfunctionGen`.
    malfunction_duration_max: int
        Max duration of malfunction. Goes into `ParamMalfunctionGen`.
    malfunction_interval: int
        Inverse of rate of malfunction occurrence. Goes into `ParamMalfunctionGen`.
    speed_ratios: Dict[float, float]
        Speed ratios of all agents. They are probabilities of all different speeds and have to add up to 1. Goes into `sparse_line_generator`. Defaults to `{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}`.
    line_length : int
            The length of the lines. Goes into `sparse_line_generator`. Defaults to `2`.
    seed: int
         Initiate random seed generators. Goes into `reset`.
    post_seed: int
         Initiate random seed after the env is generated, goes into second `reset` with `regenerate_rail=False, regenerate_schedule=False`.
         Allows for backwards compatibility with flatland-3 `client.py` behaviour:
         at that stage, random state was not serialized in the pickles, so in order for malfunction generation to be deterministic,
         the envs were `reset` with a second seed from env var `RANDOM_SEED` - the same for all envs!
         Beware also from the difference in the behaviour of `reset between env from `env_generator` and a de-pickled one:
         in the former case, rail/line/schedule generation is performed with the current random state;
         in the latter case, rail/line/schedule are loaded from file (not changing!).
    obs_builder_object: Optional[ObservationBuilder]
        Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`
    acceleration_delta : float
        Defaults to `1.0`
    braking_delta : float
        Defaults to `-1.0`
    rewards : Rewards
        Rewards function. Defaults to `DefaultRewards`.
    effects_generator : EffectsGenerator[RailEnv]
        Effects generator. Defaults to `None`.
    Returns
    -------
    RailEnv
        The generated environment reset with the given seed.
    observations : Dict
        Initial observations from `reset()`
    info : Dict
        Initial infos from `reset()`
    """
    if speed_ratios is None:
        speed_ratios = {1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))

    # avoid division by zero.
    if malfunction_interval is None or malfunction_interval == 0:
        malfunction_interval = sys.maxsize

    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                min_duration=malfunction_duration_min,
                max_duration=malfunction_duration_max,
                malfunction_rate=1.0 / malfunction_interval
            )),
        line_generator=sparse_line_generator(speed_ratio_map=speed_ratios, line_length=line_length),
        number_of_agents=n_agents,
        obs_builder_object=obs_builder_object,
        record_steps=True,
        random_seed=seed,
        acceleration_delta=acceleration_delta,
        braking_delta=braking_delta,
        rewards=rewards,
        effects_generator=effects_generator,
    )
    observations, info = env.reset(random_seed=seed)
    if post_seed is not None:
        env.reset(random_seed=post_seed, regenerate_rail=False, regenerate_schedule=False)
    return env, observations, info


def _sparse_rail_generator_legacy(*args: object, **kwargs: object) -> RailGenerator:
    return _SparseRailGenLegacy(*args, **kwargs)


class _SparseRailGenLegacy(SparseRailGen):
    def __init__(self, seed: int, **kwargs):
        self._legacy_seed = seed
        super().__init__(**kwargs)

    def generate(self, width: int, height: int, num_agents: int, num_resets: int = 0, np_random: RandomState = None) -> RailGeneratorProduct:
        # ignore seed passed from env int rail_generator, draw from random generator initialised every time by its own seed
        # see e.g. https://gitlab.aicrowd.com/flatland/flatland/-/blob/master/simple_env_creation.ipynb?ref_type=heads
        return super().generate(width=width, height=height, num_agents=num_agents, num_resets=num_resets, np_random=RandomState(self._legacy_seed))


def env_generator_legacy(
    n_agents=7,
    x_dim=30,
    y_dim=30,
    n_cities=2,
    max_rail_pairs_in_city=4,  # TODO should be 2
    grid_mode=False,
    max_rails_between_cities=2,
    malfunction_duration_min=20,
    malfunction_duration_max=50,
    malfunction_interval=540,
    speed_ratios=None,
    line_length=2,
    seed=None,
    post_seed=None,
    obs_builder_object=None,
    acceleration_delta=1.0,
    braking_delta=-1.0,
    rewards: Rewards = None,
    effects_generator: Optional[EffectsGenerator[RailEnv]] = None,
) -> Tuple[RailEnv, Dict, Dict]:
    warnings.warn("Deprecated - use the patched env_generator. Keep only for regression tests. Update tests and drop in separate pr.")
    if speed_ratios is None:
        speed_ratios = {1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))

    # avoid division by zero.
    if malfunction_interval is None or malfunction_interval == 0:
        malfunction_interval = sys.maxsize

    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=_sparse_rail_generator_legacy(
            seed=seed,
            max_num_cities=n_cities,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city,
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                min_duration=malfunction_duration_min,
                max_duration=malfunction_duration_max,
                malfunction_rate=1.0 / malfunction_interval
            )),
        line_generator=sparse_line_generator(speed_ratio_map=speed_ratios, line_length=line_length),
        number_of_agents=n_agents,
        obs_builder_object=obs_builder_object,
        record_steps=True,
        random_seed=seed,
        acceleration_delta=acceleration_delta,
        braking_delta=braking_delta,
        rewards=rewards,
        effects_generator=effects_generator,
    )
    observations, info = env.reset(random_seed=seed)
    if post_seed is not None:
        env.reset(random_seed=post_seed, regenerate_rail=False, regenerate_schedule=False)
    return env, observations, info
