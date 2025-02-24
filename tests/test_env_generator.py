import pytest

from flatland.env_generation.env_generator import env_generator
from flatland.utils.seeding import random_state_to_hashablestate


@pytest.mark.parametrize(
    "random_seed",
    [7, 42, 666, 2001]
)
def test_env_generator_deterministic(random_seed):
    gen, _, _ = env_generator(seed=random_seed)
    np_random_gen = random_state_to_hashablestate(gen.np_random)
    gen2, _, _ = env_generator(seed=random_seed)
    np_random_gen2 = random_state_to_hashablestate(gen2.np_random)

    # assert np_random is the same for two different env_generator calls with same seed
    assert np_random_gen == np_random_gen2

    # assert np_random is the same after reset w/ rail/schedule regeneration
    gen.reset(random_seed=random_seed, regenerate_rail=True, regenerate_schedule=True)
    assert np_random_gen == random_state_to_hashablestate(gen.np_random)

    # assert np_random is the differt after reset w/o rail/schedule regeneration
    gen.reset(random_seed=random_seed, regenerate_rail=False, regenerate_schedule=False)
    assert np_random_gen != random_state_to_hashablestate(gen.np_random)

    # TODO https://github.com/flatland-association/flatland-rl/issues/24 fails as hints are not saved:
    # gen.reset(random_seed=random_seed, regenerate_rail=False, regenerate_schedule=True)
    # assert np_random_gen != random_state_to_hashablestate(gen.np_random)

    # regenerate_rail implies regenerate_schedule!
    gen.reset(random_seed=random_seed, regenerate_rail=True, regenerate_schedule=False)
    assert np_random_gen == random_state_to_hashablestate(gen.np_random)
