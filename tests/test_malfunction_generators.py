from flatland.envs.malfunction_generators import malfunction_from_params, malfunction_from_file, \
    single_malfunction_generator, MalfunctionParameters
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
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
    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
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

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data)
                  )
    env.reset()
    env.save("./malfunction_saving_loading_tests.pkl")

    malfunction_generator, malfunction_process_data = malfunction_from_file("./malfunction_saving_loading_tests.pkl")
    env2 = RailEnv(width=25,
                   height=30,
                   rail_generator=rail_from_grid_transition_map(rail),
                   schedule_generator=random_schedule_generator(),
                   number_of_agents=10,
                   malfunction_generator_and_process_data=malfunction_from_params(stochastic_data)
                   )

    env2.reset()

    assert env2.malfunction_process_data == env.malfunction_process_data
    assert env2.malfunction_process_data.malfunction_rate == 1000
    assert env2.malfunction_process_data.min_duration == 2
    assert env2.malfunction_process_data.max_duration == 5


def test_single_malfunction_generator():
    """
    Test single malfunction generator
    Returns
    -------

    """

    rail, rail_map = make_simple_rail2()
    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=single_malfunction_generator(earlierst_malfunction=10,
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
                action_dict[agent.handle] = RailEnvActions(2)

            env.step(action_dict)
        for agent in env.agents:
            # Go forward all the time
            tot_malfunctions += agent.malfunction_data['nr_malfunctions']
        assert tot_malfunctions == 1
