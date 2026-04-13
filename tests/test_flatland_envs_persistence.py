import os
import tempfile

import importlib_resources as ir
import numpy as np
import pytest

from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
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
        # print(f"{env._elapsed_steps} {rewards} {done}")
        print(f"{env._elapsed_steps} {env.agents[0].speed_counter}")
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


def test_persistence_level_free():
    env, _, _ = env_generator_legacy(x_dim=100, y_dim=100, p_level_free=0.9, seed=453)

    assert env.resource_map.level_free_positions == {(53, 50), (53, 55), (57, 48), (48, 48), (53, 44)}

    assert len(env.resource_map.level_free_positions) > 0
    RailEnvPersister.save(env, filename="level_free.pkl")
    env_loaded, _ = RailEnvPersister.load_new(filename="level_free.pkl")
    assert env_loaded.resource_map.level_free_positions == {(53, 50), (53, 55), (57, 48), (48, 48), (53, 44)}


def test_multiple_malfunction_generators():
    env = RailEnv(width=50, height=50, number_of_agents=50,
                  malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(min_duration=20, max_duration=30, malfunction_rate=1.0 / 200)),
                  effects_generator=MalfunctionEffectsGenerator(
                      ParamMalfunctionGen(MalfunctionParameters(min_duration=22, max_duration=33, malfunction_rate=1.0 / 222))),
                  )
    env.reset()
    assert len(env.effects_generator.__getstate__()) == 2
    assert env.effects_generator.__getstate__() == [
        {
            'param_malfunction_gen': {'malfunction_rate': 0.0045045045045045045, 'min_duration': 22, 'max_duration': 33},
            'malfunction_cached_random_state': None,
            'malfunction_rand_idx': 0,
        },
        {
            'param_malfunction_gen': {'malfunction_rate': 0.005, 'min_duration': 20, 'max_duration': 30},
            'malfunction_cached_random_state': None,
            'malfunction_rand_idx': 0,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(env, filename=os.path.join(tmpdirname, "env.pkl"))
        env, _ = RailEnvPersister.load_new(filename=os.path.join(tmpdirname, "env.pkl"))
    assert len(env.effects_generator.__getstate__()) == 2
    assert env.effects_generator.__getstate__() == [
        {
            'param_malfunction_gen': {'malfunction_rate': 0.0045045045045045045, 'min_duration': 22, 'max_duration': 33},
            'malfunction_cached_random_state': None,
            'malfunction_rand_idx': 0,
        },
        {
            'param_malfunction_gen': {'malfunction_rate': 0.005, 'min_duration': 20, 'max_duration': 30},
            'malfunction_cached_random_state': None,
            'malfunction_rand_idx': 0,
        }
    ]
