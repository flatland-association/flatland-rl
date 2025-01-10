from typing import Dict, Optional

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.line_generators import line_from_file, sparse_line_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator


def load_flatland_environment_from_file(file_name: str,
                                        load_from_package: str = None,
                                        obs_builder_object: ObservationBuilder = None,
                                        record_steps=False,
                                        ) -> RailEnv:
    """
    Parameters
    ----------
    file_name : str
        The pickle file.
    load_from_package : str
        The python module to import from. Example: 'env_data.tests'
        This requires that there are `__init__.py` files in the folder structure we load the file from.
    obs_builder_object: ObservationBuilder
        The obs builder for the `RailEnv` that is created.


    Returns
    -------
    RailEnv
        The environment loaded from the pickle file.
    """
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(
            max_depth=2,
            predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    environment = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name, load_from_package),
                          line_generator=line_from_file(file_name, load_from_package),
                          number_of_agents=1,
                          obs_builder_object=obs_builder_object,
                          record_steps=record_steps,
                          )
    return environment


def env_creator(n_agents: int = 7,
                x_dim: int = 30,
                y_dim: int = 30,
                n_cities: int = 2,
                max_rail_pairs_in_city: int = 4,
                grid_mode: bool = False,
                max_rails_between_cities: int = 2,
                malfunction_duration_min: int = 20,
                malfunction_duration_max: int = 50,
                malfunction_interval: int = 540,
                speed_ratios: Dict[float, float] = None,
                seed: int = 42,
                obs_builder_object: Optional[ObservationBuilder] = None) -> RailEnv:
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
    seed: int
         Initiate random seed generators. Goes into `reset`.
    obs_builder_object: Optional[ObservationBuilder]
        Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`

    Returns
    -------
    RailEnv
        The generated environment reset with the given seed.
    """
    if speed_ratios is None:
        speed_ratios = {1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))

    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(
            min_duration=malfunction_duration_min, max_duration=malfunction_duration_max, malfunction_rate=1.0 / malfunction_interval)),
        line_generator=sparse_line_generator(speed_ratio_map=speed_ratios, seed=seed),
        number_of_agents=n_agents,
        obs_builder_object=obs_builder_object,
        record_steps=True,
        random_seed=seed
    )
    env.reset(random_seed=seed)
    return env
