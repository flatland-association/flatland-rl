import importlib_resources as ir
import numpy as np
import pytest

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
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
