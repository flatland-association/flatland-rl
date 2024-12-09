import numpy as np
import pytest

from flatland.utils.seeding import random_generator_get_hashablestate


@pytest.mark.parametrize("seed", [4242, 42, 31, None])
def test_random_generator_get_hashablestate(seed: int):
    random_state = np.random.RandomState()

    s = random_generator_get_hashablestate(random_state)

    random_state_deserialized = np.random.RandomState()
    random_state_deserialized.set_state(s)

    assert random_state.rand() == random_state_deserialized.rand()
    assert random_state.rand() == random_state_deserialized.rand()
    assert random_state.rand() == random_state_deserialized.rand()
