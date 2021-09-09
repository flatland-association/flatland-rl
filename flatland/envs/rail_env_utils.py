from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.line_generators import line_from_file


def load_flatland_environment_from_file(file_name: str,
                                        load_from_package: str = None,
                                        obs_builder_object: ObservationBuilder = None,
                                        record_steps = False,
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
