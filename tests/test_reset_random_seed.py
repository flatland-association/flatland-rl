from flatland.envs.rail_env import RailEnv


def test_reset_random_seed():
    env = RailEnv(width=25, height=25, number_of_agents=5, random_seed=0)
    env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=0)
    initial_pos1 = env.agents[0].initial_position

    env = RailEnv(width=25, height=25, number_of_agents=5, random_seed=0)
    env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=0)
    initial_pos2 = env.agents[0].initial_position

    print('check: ', initial_pos1, initial_pos2)
    assert initial_pos1 == initial_pos2


def test_reset_no_manual_random_seed():
    env = RailEnv(width=25, height=25, number_of_agents=5)
    env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=0)
    initial_pos1 = env.agents[0].initial_position

    env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=0)
    initial_pos2 = env.agents[0].initial_position
    print('check: ', initial_pos1, initial_pos2)
    assert initial_pos1 == initial_pos2
