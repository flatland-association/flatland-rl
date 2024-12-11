import os
import pickle

import numpy as np

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map, sparse_rail_generator
from flatland.utils.simple_rail import make_simple_rail


def test_load_new():
    filename = "test_load_new.pkl"

    rail, rail_map, optionals = make_simple_rail()
    n_agents = 2
    env_initial = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                          line_generator=sparse_line_generator(), number_of_agents=n_agents)
    env_initial.reset(False, False)

    rails_initial = env_initial.rail.grid
    agents_initial = env_initial.agents

    RailEnvPersister.save(env_initial, filename)

    env_loaded, _ = RailEnvPersister.load_new(filename)

    rails_loaded = env_loaded.rail.grid
    agents_loaded = env_loaded.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded


# TODO use env_creator from https://github.com/flatland-association/flatland-rl/pull/85 instead
def create_env():
    nAgents = 5
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    env = RailEnv(
        width=30,
        height=30,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=nAgents,
        obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
    )
    env.reset()
    return env


def readable_size(size2, decimal_point=3):
    for i in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size2 < 1024.0:
            break
        size2 /= 1024.0
    return f"{size2:.{decimal_point}f}{i}"


def test_save_load():
    env = create_env()
    RailEnvPersister.save(env, "test_save_load.pkl", True)
    print(readable_size(os.path.getsize("test_save_load.pkl")))

    env_loaded = create_env()
    RailEnvPersister.load(env_loaded, "test_save_load.pkl")

    assert env.__getstate__() == env_loaded.__getstate__()

    full_state = RailEnvPersister.get_full_state(env_loaded)
    full_state_loaded = RailEnvPersister.get_full_state(env)
    assert pickle.dumps(full_state) == pickle.dumps(full_state_loaded)


def test_dump_load():
    env = create_env()
    with open("test_save_load.pkl", "wb") as f:
        pickle.dump(env, f)

    print(readable_size(os.path.getsize("test_save_load.pkl")))

    with open("test_save_load.pkl", "rb") as f:
        env_loaded = pickle.load(f)

    expected = env.__getstate__()
    actual = env_loaded.__getstate__()
    assert expected == actual

    full_state = RailEnvPersister.get_full_state(env_loaded)
    full_state_loaded = RailEnvPersister.get_full_state(env)
    assert full_state == full_state_loaded
    assert pickle.dumps(full_state) == pickle.dumps(full_state_loaded)
