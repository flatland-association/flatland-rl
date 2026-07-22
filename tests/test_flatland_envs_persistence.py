import os
import tempfile

import importlib_resources as ir
import numpy as np
import pytest

from flatland.core.effects_generator import EffectsGenerator, find_all_effects_generators, find_effects_generator
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.record_steps_effects_generator import RecordStepsEffectsGenerator
from flatland.envs.rewards import DefaultRewards
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


def test_load_new_overrides():
    """obs_builder/rewards/effects_generator passed to load_new take effect for the restored env, replacing
    (not merging with) any restored or default counterpart - same semantics for all three."""
    rail, rail_map, optionals = make_simple_rail()
    env_initial = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                          line_generator=sparse_line_generator(), number_of_agents=2,
                          malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(min_duration=20, max_duration=30, malfunction_rate=1.0 / 200)))
    env_initial.reset(False, False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "env.pkl")
        RailEnvPersister.save(env_initial, filename)

        custom_obs_builder = DummyObservationBuilder()
        custom_rewards = DefaultRewards()
        custom_effects_generator = MalfunctionEffectsGenerator(
            ParamMalfunctionGen(MalfunctionParameters(min_duration=1, max_duration=2, malfunction_rate=1.0)))

        env_loaded, _ = RailEnvPersister.load_new(filename, obs_builder=custom_obs_builder, rewards=custom_rewards,
                                                  effects_generator=custom_effects_generator)

        assert env_loaded.obs_builder is custom_obs_builder
        assert env_loaded.rewards is custom_rewards
        assert env_loaded.effects_generator is custom_effects_generator


def test_stations_links_default_before_rail_generation():
    """A freshly-constructed RailEnv (before any reset/rail generation) must have `stations_links`
    accessible as None instead of raising AttributeError. `optionals` (the generator's raw,
    undocumented return dict) is intentionally not a RailEnv attribute at all - it's transient,
    local to `_call_rail_generator`."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    assert env.stations_links is None
    assert not hasattr(env, "optionals")


def test_set_full_state_legacy_env_dict_without_stations_links():
    """A legacy env_dict lacking the 'stations_links' key must not leave that attribute unset
    on the restored env - it should fall back to its None default."""
    rail, rail_map, optionals = make_simple_rail()
    env_initial = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                          line_generator=sparse_line_generator(), number_of_agents=2)
    env_initial.reset(False, False)

    env_dict = RailEnvPersister.get_full_state(env_initial)
    env_dict.pop("stations_links", None)

    fresh_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                        line_generator=sparse_line_generator(), number_of_agents=2)
    RailEnvPersister.set_full_state(fresh_env, env_dict)

    assert fresh_env.stations_links is None


def test_optionals_not_persisted():
    """`optionals` (the generator's raw, undocumented return dict) is not a RailEnv attribute and
    must not be persisted - get_full_state must not include it, and set_full_state must not create
    it on the restored env even from a (now-legacy) env_dict that still happens to have the key."""
    rail, rail_map, optionals = make_simple_rail()
    env_initial = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                          line_generator=sparse_line_generator(), number_of_agents=2)
    env_initial.reset(False, False)
    assert not hasattr(env_initial, "optionals")

    env_dict = RailEnvPersister.get_full_state(env_initial)
    assert "optionals" not in env_dict

    # simulate a legacy pickle saved before optionals was dropped from persistence
    env_dict["optionals"] = {"stale": "data"}

    fresh_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                        line_generator=sparse_line_generator(), number_of_agents=2)
    RailEnvPersister.set_full_state(fresh_env, env_dict)

    assert not hasattr(fresh_env, "optionals")


def test_set_full_state_restores_dev_obs_dict_without_dev_pred_dict():
    """A legacy/incomplete env_dict with dev_obs_dict but no dev_pred_dict key must still restore
    dev_obs_dict - previously this assignment was guarded by the wrong variable (dev_pred_dict_
    instead of dev_obs_dict_), a copy-paste bug."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    env.reset(False, False)

    env_dict = RailEnvPersister.get_full_state(env)
    del env_dict["dev_pred_dict"]
    env_dict["dev_obs_dict"] = {0: "some observation payload"}

    RailEnvPersister.set_full_state(env, env_dict)

    assert env.dev_obs_dict == {0: "some observation payload"}


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
    expected = {
        'cls': 'flatland.core.effects_generator.MultiEffectsGeneratorWrapped',
        'specs': {
            'args': [
                {
                    'cls': 'flatland.envs.malfunction_effects_generators.MalfunctionEffectsGenerator',
                    'specs': {
                        'kwargs': {
                            'param_malfunction_gen': {'malfunction_rate': 0.0045045045045045045, 'min_duration': 22, 'max_duration': 33},
                            'malfunction_cached_random_state': None,
                            'malfunction_rand_idx': 0}}
                },
                {
                    'cls': 'flatland.envs.malfunction_effects_generators.MalfunctionEffectsGenerator',
                    'specs': {
                        'kwargs': {
                            'param_malfunction_gen': {'malfunction_rate': 0.005, 'min_duration': 20, 'max_duration': 30},
                            'malfunction_cached_random_state': None,
                            'malfunction_rand_idx': 0
                        }
                    }
                },
                {
                    'cls': 'flatland.envs.record_steps_effects_generator.RecordStepsEffectsGenerator',
                }
            ],
        }
    }


    env = RailEnv(width=50, height=50, number_of_agents=50,
                  malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(min_duration=20, max_duration=30, malfunction_rate=1.0 / 200)),
                  effects_generator=MalfunctionEffectsGenerator(
                      ParamMalfunctionGen(MalfunctionParameters(min_duration=22, max_duration=33, malfunction_rate=1.0 / 222))),
                  record_steps=True,
                  )
    env.reset()
    assert env.effects_generator.__getstate__() == expected
    assert EffectsGenerator.from_state(expected).__getstate__() == expected

    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(env, filename=os.path.join(tmpdirname, "env.pkl"))
        env, _ = RailEnvPersister.load_new(filename=os.path.join(tmpdirname, "env.pkl"))
    assert env.effects_generator.__getstate__() == expected


def _make_env_with_record_steps():
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2, record_steps=True)
    env.reset(False, False)
    return env


def test_record_steps_effects_generator_records_live_action_dict():
    """
    `env.step(action_dict)` must deliver `action_dict` all the way through `effects_generator.on_episode_step_end`
    to the composed `RecordStepsEffectsGenerator`, not just seed `list_actions` directly via `set_state`.
    """
    env = _make_env_with_record_steps()

    action_dict_1 = {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING}
    action_dict_2 = {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD}
    env.step(action_dict_1)
    env.step(action_dict_2)

    record_steps_effects_generator = find_effects_generator(env.effects_generator, RecordStepsEffectsGenerator)
    assert record_steps_effects_generator.list_actions == [action_dict_1, action_dict_2]
    assert len(env.cur_episode) == 2


def test_record_steps_effects_generator_deserialization_new_format():
    """
    Post-refactor episode files have both a "new format" `effects_generator` state (which already embeds a
    `RecordStepsEffectsGenerator`, since it is composed into every `RailEnv`) and a top-level "actions" key
    (written by `save_episode`). Deserialization must reuse the `RecordStepsEffectsGenerator` restored from the
    `effects_generator` state - filling it with `list_actions` from the top-level "actions" key - rather than
    adding a second one.
    """
    env = _make_env_with_record_steps()
    action_dicts = [
        {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING},
        {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD},
    ]
    # seed list_actions directly so the test does not depend on what step() itself passes through the effects_generator chain
    find_effects_generator(env.effects_generator, RecordStepsEffectsGenerator).set_state(action_dicts)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "episode.pkl")
        RailEnvPersister.save_episode(env, filename)
        env_loaded, env_dict = RailEnvPersister.load_new(filename)

    assert env_dict["actions"] == action_dicts
    assert env_dict["effects_generator"] is not None

    found = find_all_effects_generators(env_loaded.effects_generator, RecordStepsEffectsGenerator)
    assert len(found) == 1, f"expected exactly one RecordStepsEffectsGenerator after deserialization, found {len(found)}"
    assert found[0].list_actions == action_dicts


def test_record_steps_effects_generator_deserialization_old_format():
    """
    Legacy episode files (pre-dating `RecordStepsEffectsGenerator`) have a top-level "actions" key from
    `save_episode`, but no `RecordStepsEffectsGenerator` in their (possibly absent) `effects_generator` state.
    Deserialization must add exactly one `RecordStepsEffectsGenerator` and fill it from that legacy "actions" key.
    """
    env = _make_env_with_record_steps()

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "episode.pkl")
        RailEnvPersister.save_episode(env, filename)
        env_dict = RailEnvPersister.load_env_dict(filename)

    # simulate a legacy episode file: no "effects_generator" key at all
    del env_dict["effects_generator"]
    action_dicts = [{0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING}]
    env_dict["actions"] = action_dicts

    env_loaded = _make_env_with_record_steps()
    RailEnvPersister.set_full_state(env_loaded, env_dict)

    found = find_all_effects_generators(env_loaded.effects_generator, RecordStepsEffectsGenerator)
    assert len(found) == 1, f"expected exactly one RecordStepsEffectsGenerator after deserialization, found {len(found)}"
    assert found[0].list_actions == action_dicts
