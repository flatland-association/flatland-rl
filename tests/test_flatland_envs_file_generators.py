import tempfile
import uuid
from pathlib import Path

import numpy as np

from flatland.envs import timetable_generators
from flatland.envs.line_generators import FileLineGenerator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import FileRailFromGridGen, sparse_rail_generator
from flatland.envs.timetable_generators import FileTimetableGenerator


def test_rail_env():
    n_agents = 7
    x_dim = 30
    y_dim = 30
    n_cities = 2
    max_rail_pairs_in_city = 4
    grid_mode = False
    max_rails_between_cities = 2
    malfunction_duration_min = 20
    malfunction_duration_max = 50
    malfunction_interval = 540
    speed_ratios = {1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}
    seed = 42

    obs_builder_object = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        rail_pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        line_pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        tt_pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        rail_env = RailEnv(
            width=x_dim,
            height=y_dim,
            rail_generator=FileRailFromGridGen.wrap(sparse_rail_generator(
                max_num_cities=n_cities,
                seed=seed,
                grid_mode=grid_mode,
                max_rails_between_cities=max_rails_between_cities,
                max_rail_pairs_in_city=max_rail_pairs_in_city
            ), rail_pkl=rail_pkl),
            malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(
                min_duration=malfunction_duration_min, max_duration=malfunction_duration_max, malfunction_rate=1.0 / malfunction_interval)),
            line_generator=FileLineGenerator.wrap(sparse_line_generator(speed_ratio_map=speed_ratios, seed=seed), line_pkl),
            timetable_generator=FileTimetableGenerator.wrap(timetable_generators.timetable_generator, tt_pkl),
            number_of_agents=n_agents,
            obs_builder_object=obs_builder_object,
            record_steps=True
        )
        rail_env.reset(random_seed=seed)

        rail_env_loaded = RailEnv(
            width=rail_env.width, height=rail_env.height,
            number_of_agents=rail_env.number_of_agents,
            rail_generator=FileRailFromGridGen(rail_pkl),
            line_generator=FileLineGenerator(line_pkl),
            timetable_generator=FileTimetableGenerator(tt_pkl)
        )
        rail_env_loaded.reset(random_seed=seed)

        assert rail_env.rail.grid.tolist() == rail_env_loaded.rail.grid.tolist()
        assert rail_env.agents == rail_env_loaded.agents

        # Beware: as reset does not run through the rail, line and timetable generation, it is in a different state although we passed in the same seed!
        np_random_state = rail_env.np_random.get_state()
        np_random_state_loaded = rail_env_loaded.np_random.get_state()
        assert np_random_state[0] == np_random_state_loaded[0]
        assert not np.array_equal(np_random_state[1], np_random_state_loaded[1])
        assert np_random_state[2] != np_random_state_loaded[2]
        assert np_random_state[3] == np_random_state_loaded[3]
        assert np_random_state[4] == np_random_state_loaded[4]
