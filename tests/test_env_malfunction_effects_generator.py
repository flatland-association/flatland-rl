from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator, IntermediateStopMalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters


def test_get_set_state():
    expected = MalfunctionEffectsGenerator(ParamMalfunctionGen(MalfunctionParameters(min_duration=22, max_duration=33, malfunction_rate=1.0 / 222)))
    assert expected.__getstate__() == {
        'cls': 'flatland.envs.malfunction_effects_generators.MalfunctionEffectsGenerator',
        'specs': {
            'kwargs': {
                'param_malfunction_gen': {'malfunction_rate': 0.0045045045045045045, 'max_duration': 33, 'min_duration': 22},
                'malfunction_cached_random_state': None,
                'malfunction_rand_idx': 0,
            }
        }
    }

    actual = MalfunctionEffectsGenerator.from_state(expected.__getstate__())
    assert actual.__getstate__() == expected.__getstate__()


def test_intermediate_stop_malfunction_effects_generator():
    eg = IntermediateStopMalfunctionEffectsGenerator(
        malfunction_rate=55,
        min_duration=56,
        max_duration=57,
        earliest_malfunction=58
    )
    expected = {'cls': 'flatland.envs.malfunction_effects_generators.IntermediateStopMalfunctionEffectsGenerator',
                'specs': {'kwargs': {'malfunction_rate': 55.0, 'min_duration': 56, 'max_duration': 57, 'earliest_malfunction': 58}}}
    assert eg.__getstate__() == expected
    assert MalfunctionEffectsGenerator.from_state(expected).__getstate__() == expected
