import json
import os
import pickle

import importlib_resources as ir
import numpy as np
import pytest

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
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


def test_legacy_envs():
    envs = [("env_data.railway", sRes) for sExt in ["mpk", "pkl"] for sRes in ir.contents("env_data.railway") if sRes.endswith(sExt)]
    for package, resource in envs:
        print("Loading: ", package, resource)
        env, _ = RailEnvPersister.load_new(resource, load_from_package=package)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step({i: RailEnvActions.MOVE_FORWARD for i in env.get_agent_handles()})
            done = done['__all__']

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

    # RailEnvPersister save and load restores full_state
    full_state = RailEnvPersister.get_full_state(env_loaded)
    full_state_loaded = RailEnvPersister.get_full_state(env)
    assert full_state == full_state_loaded
    assert pickle.dumps(full_state) == pickle.dumps(full_state_loaded)

    # RailEnvPersister save and load restore does not fully restore state == - TODO should we fix?!
    env_state = env.__getstate__(True)
    env_loaded_state = env_loaded.__getstate__(True)
    for k in env_state:
        if k != 'distance_map' and k != 'rail':
            assert env_state[k] == env_loaded_state[k], (k, env_state[k] == env_loaded_state[k])
        else:
            for kk in env_state[k]:
                if kk in ['agents_previous_computation', 'agents']:
                    assert env_state[k][kk] != env_loaded_state[k][kk], (k, kk)
                else:
                    assert env_state[k][kk] == env_loaded_state[k][kk], (k, kk)


# TODO use new_load and new_save instead
# TODO test all 4 formats yield the same!
# pickle dump and load (new implementation) restores state and full_state
def test_dump_load_pickle():
    env = create_env()
    # TODO bad code smell
    env.save_distance_maps = True
    with open("test_save_load.pkl", "wb") as f:
        pickle.dump(env, f)
    print(readable_size(os.path.getsize("test_save_load.pkl")))
    with open("test_save_load.pkl", "rb") as f:
        env_loaded = pickle.load(f)

    # pickle dump and load (new implementation) restores state ==
    expected = env.__getstate__(True)
    actual = env_loaded.__getstate__(True)
    assert expected == actual

    # pickle dump and load (new implementation) restores full_state as well
    full_state = RailEnvPersister.get_full_state(env)
    full_state_loaded = RailEnvPersister.get_full_state(env_loaded)
    assert full_state == full_state_loaded
    assert pickle.dumps(full_state) == pickle.dumps(full_state_loaded)

@pytest.mark.skip(
    "TODO dictionary keys converted to string, https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings")
def test_dump_load_json():
    env = create_env()
    with open("test_save_load.json", "w") as f:
        json.dump(env.__getstate__(), f, default=float)
    print(readable_size(os.path.getsize("test_save_load.json")))
    with open("test_save_load.json", "r") as f:
        env_loaded = RailEnv(0, 0).__setstate__(json.load(f))

    diff = []
    for k in env.__getstate__():
        if env.__getstate__()[k] != env_loaded.__getstate__()[k]:
            diff.append(k)
    print(diff)
    # json dump and load (new implementation) restores state ==
    expected = env.__getstate__()
    actual = env_loaded.__getstate__()
    assert expected == actual

    # json dump and load (new implementation) restores full_state as well
    full_state = RailEnvPersister.get_full_state(env_loaded)
    full_state_loaded = RailEnvPersister.get_full_state(env)
    assert full_state == full_state_loaded
    assert pickle.dumps(full_state) == pickle.dumps(full_state_loaded)


@pytest.mark.parametrize(
    "package, resource, expected",
    [
        ("env_data.tests.service_test.Test_0", "Level_0.pkl", -593),
        ("env_data.tests.service_test.Test_1", "Level_0.pkl", -593),
        ("env_data.tests.service_test.Test_0", "Level_1.pkl", -561.0),
        ("env_data.tests.service_test.Test_1", "Level_1.pkl", -561.0),
    ])
def test_regression_forward(package, resource, expected):
    env, _ = RailEnvPersister.load_new(resource, load_from_package=package)
    env.reset(
        regenerate_rail=True,
        regenerate_schedule=True,
        random_seed=1001
    )
    done = False
    total_rewards = 0
    while not done:
        _, rewards, done, _ = env.step({i: RailEnvActions.MOVE_FORWARD for i in env.get_agent_handles()})
        total_rewards += sum(rewards.values())
        done = done['__all__']

    assert total_rewards == expected


@pytest.mark.parametrize(
    "package, resource",
    [
        ("env_data.tests.service_test.Test_0", "Level_0.pkl"),
        ("env_data.tests.service_test.Test_1", "Level_0.pkl"),
        ("env_data.tests.service_test.Test_0", "Level_1.pkl"),
        ("env_data.tests.service_test.Test_1", "Level_1.pkl"),
    ])
def test_regression_random(package, resource):
    # N.B. grid contains symmetric switch - find edge cases with random controller...
    for _ in range(100):
        env, _ = RailEnvPersister.load_new(resource, load_from_package=package)
        env.reset(
            regenerate_rail=True,
            regenerate_schedule=True,
            random_seed=1001
        )
        done = False
        total_rewards = 0
        while not done:
            _, rewards, done, _ = env.step({i: np.random.randint(0, 5) for i in env.get_agent_handles()})
            total_rewards += sum(rewards.values())
            done = done['__all__']
