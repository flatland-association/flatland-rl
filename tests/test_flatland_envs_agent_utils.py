import pytest

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.simple_rail import  make_oval_rail


def test_shortest_paths():
    rail, rail_map, optionals = make_oval_rail()

    speed_ratio_map = {1.: 1.0}
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(speed_ratio_map),
                  number_of_agents=2)
    env.reset()

    agent0_shortest_path = env.agents[0].get_shortest_path(env.distance_map)
    agent1_shortest_path = env.agents[1].get_shortest_path(env.distance_map)

    assert len(agent0_shortest_path) == 10
    assert len(agent1_shortest_path) == 10


def test_travel_time_on_shortest_paths():
    rail, rail_map, optionals = make_oval_rail()

    speed_ratio_map = {1.: 1.0}
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(speed_ratio_map),
                  number_of_agents=2)
    env.reset()

    agent0_travel_time = env.agents[0].get_travel_time_on_shortest_path(env.distance_map)
    agent1_travel_time = env.agents[1].get_travel_time_on_shortest_path(env.distance_map)

    assert agent0_travel_time == 10
    assert agent1_travel_time == 10


    speed_ratio_map = {1/2: 1.0}
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(speed_ratio_map),
                  number_of_agents=2)
    env.reset()

    agent0_travel_time = env.agents[0].get_travel_time_on_shortest_path(env.distance_map)
    agent1_travel_time = env.agents[1].get_travel_time_on_shortest_path(env.distance_map)

    assert agent0_travel_time == 20
    assert agent1_travel_time == 20


    speed_ratio_map = {1/3: 1.0}
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(speed_ratio_map),
                  number_of_agents=2)
    env.reset()

    agent0_travel_time = env.agents[0].get_travel_time_on_shortest_path(env.distance_map)
    agent1_travel_time = env.agents[1].get_travel_time_on_shortest_path(env.distance_map)


    assert agent0_travel_time == 30
    assert agent1_travel_time == 30


    speed_ratio_map = {1/4: 1.0}
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(speed_ratio_map),
                  number_of_agents=2)
    env.reset()

    agent0_travel_time = env.agents[0].get_travel_time_on_shortest_path(env.distance_map)
    agent1_travel_time = env.agents[1].get_travel_time_on_shortest_path(env.distance_map)

    assert agent0_travel_time == 40
    assert agent1_travel_time == 40


# def test_latest_arrival_validity():
#     pass


# def test_time_remaining_until_latest_arrival():
#     pass

def main():
    pass

if __name__ == "__main__":
    main()
