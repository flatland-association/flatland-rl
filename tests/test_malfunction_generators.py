import pickle
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pytest

from flatland.core.effects_generator import EffectsGenerator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_effects_generators import ConditionalMalfunctionEffectsGenerator, on_map_state_condition, \
    IntermediateStopMalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import malfunction_from_params, malfunction_from_file, \
    single_malfunction_generator, MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map, sparse_rail_generator
from flatland.utils.simple_rail import make_simple_rail2


def test_malfanction_from_params():
    """
    Test loading malfunction from
    Returns
    -------

    """
    stochastic_data = MalfunctionParameters(malfunction_rate=1000,  # Rate of malfunction occurence
                                            min_duration=2,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )
    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data)
                  )
    env.reset()
    assert env.malfunction_process_data.malfunction_rate == 1000
    assert env.malfunction_process_data.min_duration == 2
    assert env.malfunction_process_data.max_duration == 5


def test_malfanction_to_and_from_file():
    """
    Test loading malfunction from
    Returns
    -------

    """
    stochastic_data = MalfunctionParameters(malfunction_rate=1000,  # Rate of malfunction occurence
                                            min_duration=2,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data)
                  )
    env.reset()
    #env.save("./malfunction_saving_loading_tests.pkl")
    RailEnvPersister.save(env, "./malfunction_saving_loading_tests.pkl")


    malfunction_generator, malfunction_process_data = malfunction_from_file("./malfunction_saving_loading_tests.pkl")
    env2 = RailEnv(width=25,
                   height=30,
                   rail_generator=rail_from_grid_transition_map(rail, optionals),
                   line_generator=sparse_line_generator(),
                   number_of_agents=10,
                   malfunction_generator_and_process_data=malfunction_from_params(stochastic_data)
                   )

    env2.reset()

    assert env2.malfunction_process_data == env.malfunction_process_data
    assert env2.malfunction_process_data.malfunction_rate == 1000
    assert env2.malfunction_process_data.min_duration == 2
    assert env2.malfunction_process_data.max_duration == 5


@pytest.mark.skip("Single malfunction generator is deprecated")
def test_single_malfunction_generator():
    """
    Test single malfunction generator
    Returns
    -------

    """

    rail, rail_map, optionals = make_simple_rail2()
    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=single_malfunction_generator(earlierst_malfunction=3,
                                                                                      malfunction_duration=5)
                  )
    for test in range(10):
        env.reset()
        action_dict = dict()
        tot_malfunctions = 0
        print(test)
        for i in range(10):
            for agent in env.agents:
                # Go forward all the time
                action_dict[agent.handle] = RailEnvActions.MOVE_FORWARD

            _, _, dones, _ = env.step(action_dict)
            if dones['__all__']:
                break
        for agent in env.agents:
            # Go forward all the time
            tot_malfunctions += agent.malfunction_handler.num_malfunctions
        assert tot_malfunctions == 1



@pytest.mark.parametrize("min_duration,max_duration", [(1, 3), (2, 2), (0, 1)])
def test_param_malfunction_gen_draws_durations_in_documented_range(min_duration, max_duration):
    """
    `ParamMalfunctionGen` with a rate high enough that every draw is a malfunction, 10,000 draws: the durations are
    exactly the values of `[min_duration, max_duration]`, both inclusive, each drawn about equally often (#565).
    """
    gen = ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1e9, min_duration=min_duration, max_duration=max_duration))
    np_random = np.random.RandomState(0)
    n = 10_000
    durations = Counter(gen.generate(np_random).num_broken_steps for _ in range(n))

    assert set(durations) == set(range(min_duration, max_duration + 1))
    expected = n / (max_duration - min_duration + 1)
    assert all(abs(count - expected) < 0.05 * n for count in durations.values()), durations


def test_malfunction_blocks_agent_for_drawn_duration():
    """
    Two agents given MOVE_FORWARD in every step; one malfunction of duration 2, drawn as soon as an agent is on the map
    (conditional generator, at most one malfunction).

    - The agent is in MALFUNCTION in the step of the draw and the next one, exactly 2 steps (#565);
      `info['malfunction']` is 2, then 1, then 0.
    - In the step the counter is 0 the agent is MOVING again (starting from speed 0, it doesn't move yet); it leaves
      its cell in the step after.
    """
    env = _env_with_malfunctions(malfunctions=False, effects_generator=ConditionalMalfunctionEffectsGenerator(
        malfunction_rate=1e9, min_duration=2, max_duration=2, max_num_malfunctions=1, condition=on_map_state_condition))
    env.reset(random_seed=42)

    rows = []
    for _ in range(8):
        _, _, _, info = env.step({h: RailEnvActions.MOVE_FORWARD for h in env.get_agent_handles()})
        agent = env.agents[0]
        rows.append((env._elapsed_steps, agent.state.name, info['malfunction'][0], agent.current_entry_point))

    assert [row[:3] for row in rows[2:7]] == [
        (3, "MOVING", 0),  # on the map
        (4, "MALFUNCTION", 2),  # drawn
        (5, "MALFUNCTION", 1),
        (6, "MOVING", 0),  # starts again
        (7, "MOVING", 0),
    ]
    assert rows[2][3] == rows[3][3] == rows[4][3] == rows[5][3] != rows[6][3]
    assert [agent.malfunction_handler.num_malfunctions for agent in env.agents] == [1, 0]


def test_intermediate_stop_malfunctions_default_to_one_step():
    """
    `IntermediateStopMalfunctionEffectsGenerator` with only a malfunction rate: every positive draw is a malfunction of
    1 step, not a duration of 0, which would be no malfunction.
    """
    generator = IntermediateStopMalfunctionEffectsGenerator(malfunction_rate=np.inf)._malfunction_generator
    np_random = np.random.RandomState(0)
    assert {generator.generate(np_random).num_broken_steps for _ in range(100)} == {1}


def _env_with_malfunctions(malfunctions: bool = True, effects_generator: Optional[EffectsGenerator] = None) -> RailEnv:
    """Two agents on a sparse rail, seed 42; with `malfunctions`, the env's own malfunctions have rate 1/5 and durations 1 to 3."""
    return RailEnv(width=30, height=30, number_of_agents=2,
                   rail_generator=sparse_rail_generator(max_num_cities=2, seed=1),
                   line_generator=sparse_line_generator(seed=1),
                   malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 5, min_duration=1, max_duration=3))
                   if malfunctions else None,
                   effects_generator=effects_generator,
                   random_seed=42)


def _run_malfunctions(env: RailEnv, steps: int = 40) -> Tuple[List[str], List[int]]:
    """
    Run the env with MOVE_FORWARD for all agents. Returns, per agent, its malfunction counter after each step as a string of
    digits (one per step), and its number of malfunctions in the episode.
    """
    counters = {h: "" for h in env.get_agent_handles()}
    for _ in range(steps):
        _, _, dones, info = env.step({h: RailEnvActions.MOVE_FORWARD for h in env.get_agent_handles()})
        for h in env.get_agent_handles():
            counters[h] += str(info['malfunction'][h])
        if dones['__all__']:
            break
    return [counters[h] for h in env.get_agent_handles()], [agent.malfunction_handler.num_malfunctions for agent in env.agents]


def test_seeded_reset_reproduces_malfunctions():
    """
    Two agents with malfunction rate 1/5 and durations 1 to 3, both given MOVE_FORWARD in every step.

    - Setup: a new env, reset with seed 42, runs 40 steps; both agents break down.
    - The same env is reset with seed 42 again and runs 40 steps: the malfunction counters are the same in every step,
      and so is the number of malfunctions per agent.
    - Another new env, reset with seed 42, has the same malfunctions too.
    """
    env = _env_with_malfunctions()
    env.reset(random_seed=42)
    first = _run_malfunctions(env)

    env.reset(random_seed=42)
    second = _run_malfunctions(env)

    other = _env_with_malfunctions()
    other.reset(random_seed=42)
    third = _run_malfunctions(other)

    assert first == second == third
    assert all(n > 0 for n in first[1])


# Malfunction counters per agent after each step, and number of malfunctions per agent, of the first episode of
# _env_with_malfunctions() reset with seed 42, as recorded before malfunctions were started over per episode, with the
# durations of #565: the malfunctions start in the same steps as before, each one step shorter (before #565, agent 0 had
# '0000000000043210000004321000004321000032', agent 1 '0213210000000000021000321432121000321000'). Agent 1 has a
# malfunction that starts in the step after the previous one ends ("1" followed by "021").
PINNED_COUNTERS = ['0000000000032100000003210000003210000021', '0102100000000000010000210321210000210000']
PINNED_NUM_MALFUNCTIONS = [4, 7]


def test_first_episode_malfunctions_unchanged():
    """
    Two agents with malfunction rate 1/5 and durations 1 to 3, both given MOVE_FORWARD in every step: the malfunctions
    of the first episode of a new env reset with seed 42.

    - The malfunction counters after each of the 40 steps, and the number of malfunctions per agent, are the same as
      before malfunctions were started over per episode: recorded episodes keep their malfunctions.
    """
    env = _env_with_malfunctions()
    env.reset(random_seed=42)

    assert _run_malfunctions(env) == (PINNED_COUNTERS, PINNED_NUM_MALFUNCTIONS)


def test_seeded_reset_reproduces_conditional_malfunctions():
    """
    Two agents given MOVE_FORWARD in every step, with malfunctions only from a conditional effects generator (rate 1/5,
    durations 1 to 3, no condition).

    - Setup: a new env, reset with seed 42, runs 40 steps; both agents break down.
    - The same env is reset with seed 42 again and runs 40 steps: the malfunction counters are the same in every step.
    - Another new env with the same conditional generator, reset with seed 42, has the same malfunctions too.
    """

    def _env():
        return _env_with_malfunctions(malfunctions=False,
                                      effects_generator=ConditionalMalfunctionEffectsGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=3))

    env = _env()
    env.reset(random_seed=42)
    first = _run_malfunctions(env)

    env.reset(random_seed=42)
    second = _run_malfunctions(env)

    other = _env()
    other.reset(random_seed=42)
    third = _run_malfunctions(other)

    assert first == second == third
    assert all(n > 0 for n in first[1])


def test_max_num_malfunctions_per_episode():
    """
    Two agents given MOVE_FORWARD in every step, with malfunctions only from a conditional effects generator (rate
    1/5, durations 1 to 3, at most 1 malfunction).

    - Setup: a new env, reset with seed 42, runs 40 steps: exactly one malfunction over all agents.
    - The same env is reset with seed 42 again and runs 40 steps: again exactly one malfunction, in the same step,
      as the maximum counts per episode.
    """
    env = _env_with_malfunctions(malfunctions=False,
                                 effects_generator=ConditionalMalfunctionEffectsGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=3,
                                                                                          max_num_malfunctions=1))
    env.reset(random_seed=42)
    first = _run_malfunctions(env)

    env.reset(random_seed=42)
    second = _run_malfunctions(env)

    assert sum(first[1]) == 1
    assert first == second


def test_loaded_env_reset_starts_malfunctions_over(tmp_path):
    """
    Two agents with malfunction rate 1/5 and durations 1 to 3, both given MOVE_FORWARD in every step; the env is saved
    after its seeded reset and again after 3 steps, and both files are loaded and reset with seed 42, as the evaluators
    do.

    - Setup: a new env is reset with seed 42 and saved (episode start); it runs 3 steps and is saved again (mid-episode,
      with the position in its block of random numbers).
    - The episode-start file is loaded, reset with seed 42 and runs 40 steps; reset with seed 42 again, it has the same
      malfunctions.
    - The mid-episode file is loaded and reset with seed 42: the reset starts the malfunctions over, so the 40 steps
      have the same malfunctions as after loading the episode-start file.
    """
    env = _env_with_malfunctions()
    env.reset(random_seed=42)
    start_file = tmp_path / "start.pkl"
    RailEnvPersister.save(env, start_file)
    for _ in range(3):
        env.step({h: RailEnvActions.MOVE_FORWARD for h in env.get_agent_handles()})
    mid_file = tmp_path / "mid.pkl"
    RailEnvPersister.save(env, mid_file)

    loaded, _ = RailEnvPersister.load_new(start_file)
    loaded.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=42)
    first = _run_malfunctions(loaded)
    loaded.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=42)
    second = _run_malfunctions(loaded)

    loaded_mid, _ = RailEnvPersister.load_new(mid_file)
    loaded_mid.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=42)
    third = _run_malfunctions(loaded_mid)

    assert first == second == third
    assert all(n > 0 for n in first[1])


def test_conditional_malfunctions_replay_from_mid_episode_snapshot(tmp_path):
    """
    Two agents given MOVE_FORWARD in every step, with malfunctions only from a conditional effects generator (rate 1/5,
    durations 1 to 3, at most 3 malfunctions, only for agents on the map).

    - Setup: a new env is reset with seed 42 and runs 10 steps, with malfunctions; it is saved (mid-episode) and runs
      30 more steps.
    - The saved env is loaded and runs 30 steps: the malfunction counters are the same in every step as in the 30 steps
      after saving, so the count towards the maximum, the position in the block of random numbers and whether each agent
      was in malfunction carry over.
    - The saved state of the generator holds the random state the block was drawn from, not the block itself: it
      pickles to less than 10 KB (the block is 8 MB), so the saved env doesn't grow considerably.
    """
    env = _env_with_malfunctions(malfunctions=False,
                                 effects_generator=ConditionalMalfunctionEffectsGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=3,
                                                                                          max_num_malfunctions=3, condition=on_map_state_condition))
    env.reset(random_seed=42)
    before = _run_malfunctions(env, steps=10)
    snapshot = tmp_path / "mid.pkl"
    RailEnvPersister.save(env, snapshot)
    expected, _ = _run_malfunctions(env, steps=30)

    loaded, _ = RailEnvPersister.load_new(snapshot)
    actual, _ = _run_malfunctions(loaded, steps=30)

    assert sum(before[1]) > 0
    assert actual == expected
    assert len(pickle.dumps(env.effects_generator.__getstate__())) < 10_000
