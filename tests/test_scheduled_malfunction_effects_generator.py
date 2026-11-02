import sys

import pytest

from flatland.core.effects_generator import EffectsGenerator, find_effects_generator
from flatland.env_generation.env_generator import env_generator
from flatland.envs.malfunction_effects_generators import ScheduledMalfunction, ScheduledMalfunctionEffectsGenerator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState


def _env(malfunctions):
    env, _, _ = env_generator(n_agents=2,
                              seed=42,
                              malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
                              effects_generator=ScheduledMalfunctionEffectsGenerator(malfunctions))
    for agent in env.agents:
        agent.earliest_departure = 0
    return env


def _run(env, num_steps):
    """Steps the env and returns, per elapsed step, the agents' in_malfunction and derived state after the step."""
    in_malfunction = {}
    states = {}
    for _ in range(num_steps):
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD for agent in env.agents})
        in_malfunction[env._elapsed_steps] = {
            agent.handle: agent.malfunction_handler.in_malfunction for agent in env.agents}
        states[env._elapsed_steps] = {agent.handle: agent.derived_state() for agent in env.agents}
    return in_malfunction, states


def test_scheduled_malfunction_breaks_exactly_during_window():
    step, duration = 5, 3
    env = _env([ScheduledMalfunction(handle=0, step=step, duration=duration)])

    in_malfunction, states = _run(env, 12)

    assert [s for s, by_handle in in_malfunction.items() if by_handle[0]] == list(range(step, step + duration))
    assert all(states[s][0] in (TrainState.MALFUNCTION, TrainState.MALFUNCTION_OFF_MAP)
               for s in range(step, step + duration))
    # the other agent is unaffected
    assert not any(by_handle[1] for by_handle in in_malfunction.values())
    assert env.agents[0].malfunction_handler.num_malfunctions == 1


def test_scheduled_malfunction_at_first_step_before_departure_is_off_map():
    env = _env([ScheduledMalfunction(handle=0, step=1, duration=2)])

    in_malfunction, states = _run(env, 4)

    assert [s for s, by_handle in in_malfunction.items() if by_handle[0]] == [1, 2]
    assert [states[s][0] for s in (1, 2)] == [TrainState.MALFUNCTION_OFF_MAP] * 2


def test_scheduled_malfunction_skipped_with_warning_if_already_in_malfunction():
    env = _env([ScheduledMalfunction(handle=0, step=5, duration=3),
                ScheduledMalfunction(handle=0, step=6, duration=10)])

    with pytest.warns(UserWarning, match="already in malfunction"):
        in_malfunction, _ = _run(env, 12)

    assert [s for s, by_handle in in_malfunction.items() if by_handle[0]] == [5, 6, 7]


def test_scheduled_malfunction_accepts_lists():
    eg = ScheduledMalfunctionEffectsGenerator([[1, 7, 2]])

    assert eg.malfunctions == [ScheduledMalfunction(handle=1, step=7, duration=2)]


@pytest.mark.parametrize("malfunction, message", [
    (ScheduledMalfunction(handle=0, step=0, duration=1), "step must be >= 1"),
    (ScheduledMalfunction(handle=0, step=1, duration=0), "duration must be >= 1"),
])
def test_scheduled_malfunction_invalid(malfunction, message):
    with pytest.raises(ValueError, match=message):
        ScheduledMalfunctionEffectsGenerator([malfunction])


def test_scheduled_malfunction_effects_generator_state_round_trip():
    malfunctions = [ScheduledMalfunction(handle=0, step=5, duration=3),
                    ScheduledMalfunction(handle=1, step=7, duration=1)]

    restored = EffectsGenerator.from_state(ScheduledMalfunctionEffectsGenerator(malfunctions).__getstate__())

    assert isinstance(restored, ScheduledMalfunctionEffectsGenerator)
    assert restored.malfunctions == malfunctions


def test_scheduled_malfunction_survives_persistence(tmp_path):
    step, duration = 5, 3
    env = _env([ScheduledMalfunction(handle=0, step=step, duration=duration)])
    RailEnvPersister.save(env, str(tmp_path / "env.pkl"))

    loaded, _ = RailEnvPersister.load_new(str(tmp_path / "env.pkl"))
    loaded.reset()
    for agent in loaded.agents:
        agent.earliest_departure = 0
    in_malfunction, _ = _run(loaded, 12)

    assert find_effects_generator(loaded.effects_generator, ScheduledMalfunctionEffectsGenerator).malfunctions == [
        ScheduledMalfunction(handle=0, step=step, duration=duration)]
    assert [s for s, by_handle in in_malfunction.items() if by_handle[0]] == list(range(step, step + duration))
