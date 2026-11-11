from typing import List, Optional, Tuple

import pytest

from flatland.core.effects_generator import EffectsGenerator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_effects_generators import ConditionalMalfunctionEffectsGenerator
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
# _env_with_malfunctions() reset with seed 42, as recorded before malfunctions were started over per episode. Agent 1
# has malfunctions that start right after the previous one ends ("21" followed by "321").
PINNED_COUNTERS = ['0000000000043210000004321000004321000032', '0213210000000000021000321432121000321000']
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
