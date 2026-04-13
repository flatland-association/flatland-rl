from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters


def test_get_set_state():
    expected = MalfunctionEffectsGenerator(ParamMalfunctionGen(MalfunctionParameters(min_duration=22, max_duration=33, malfunction_rate=1.0 / 222)))
    assert expected.__getstate__() == {
        'param_malfunction_gen': {'malfunction_rate': 0.0045045045045045045, 'max_duration': 33, 'min_duration': 22},
        'malfunction_cached_random_state': None,
        'malfunction_rand_idx': 0,
    }

    actual = MalfunctionEffectsGenerator(None)
    actual.__setstate__(expected.__getstate__())
    assert actual.__getstate__() == expected.__getstate__()
