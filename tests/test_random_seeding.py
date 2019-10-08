from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail2


def test_random_seeding():
    # Set fixed malfunction duration for this test
    rail, rail_map = make_simple_rail2()

    # Move target to unreachable position in order to not interfere with test
    for idx in range(100):
        env = RailEnv(width=25,
                      height=30,
                      rail_generator=rail_from_grid_transition_map(rail),
                      schedule_generator=random_schedule_generator(seed=12),
                      number_of_agents=10
                      )
        env.reset(True, True, False, random_seed=1)
        # Test generation print

        env.agents[0].target = (0, 0)
        for step in range(10):
            actions = {}
            actions[0] = 2
            env.step(actions)
        agent_positions = []
        for a in range(env.get_num_agents()):
            agent_positions += env.agents[a].initial_position
        # print(agent_positions)
        assert agent_positions == [3, 2, 3, 5, 3, 6, 5, 6, 3, 4, 3, 1, 3, 9, 4, 6, 0, 3, 3, 7]
        # Test generation print
        assert env.agents[0].position == (3, 6)
        # print("env.agents[0].initial_position == {}".format(env.agents[0].initial_position))
        #print("assert env.agents[0].position ==  {}".format(env.agents[0].position))


def test_seeding_and_observations():
    print("nothing yet")
