import os
import tempfile

import pytest

from flatland.env_generation.env_generator import env_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.utils.seeding import random_state_to_hashablestate


@pytest.mark.parametrize(
    'seed',
    [43, 44, 1001, 249385789, 289435789]
)
def test_persistence_reset(seed):
    rail_env, _, _ = env_generator(seed=seed, x_dim=40, y_dim=57, )
    np_random_generated = random_state_to_hashablestate(rail_env.np_random)
    dict_generated = RailEnvPersister.get_full_state(rail_env)

    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(rail_env, os.path.join(tmpdirname, "env.pkl"))
        reloaded, _ = RailEnvPersister.load_new(os.path.join(tmpdirname, "env.pkl"))
    np_random_reloaded = random_state_to_hashablestate(reloaded.np_random)
    dict_reloaded = RailEnvPersister.get_full_state(reloaded)

    assert np_random_generated == np_random_reloaded
    assert dict_generated == dict_reloaded

    rail_env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
    np_random_reset = random_state_to_hashablestate(rail_env.np_random)
    dict_reset = RailEnvPersister.get_full_state(rail_env)

    assert np_random_generated == np_random_reset
    assert dict_generated == dict_reset

    # CAVEAT: if we pass the seed but do not regenerate, this results in a different state!
    rail_env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=seed)
    np_random_reset_no_regenerate_same_seed = random_state_to_hashablestate(rail_env.np_random)
    dict_reset_no_regenerate_same_seed = RailEnvPersister.get_full_state(rail_env)

    assert np_random_generated != np_random_reset_no_regenerate_same_seed
    assert dict_generated != dict_reset_no_regenerate_same_seed

    # however, if we do not regenerate and pass no seed, this results in the same state.
    rail_env.reset(regenerate_rail=False, regenerate_schedule=False)
    np_random_reset_no_regenerate_no_seed = random_state_to_hashablestate(rail_env.np_random)
    dict_reset_no_regenerate_no_seed = RailEnvPersister.get_full_state(rail_env)

    assert np_random_generated != np_random_reset_no_regenerate_no_seed
    assert dict_generated != dict_reset_no_regenerate_no_seed
