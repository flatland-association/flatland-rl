from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.timetable_utils import Line
from flatland.utils.simple_rail import make_oval_rail


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

    speed_ratio_map = {1 / 2: 1.0}
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

    speed_ratio_map = {1 / 3: 1.0}
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

    speed_ratio_map = {1 / 4: 1.0}
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


def test_from_line():
    line = Line(agent_positions=[[(11, 40)], [(38, 8)], [(17, 5)], [(41, 22)], [(11, 40)], [(38, 8)], [(38, 8)], [(31, 26)], [(41, 22)], [(9, 27)]],
                agent_directions=[[Grid4TransitionsEnum(3)], [Grid4TransitionsEnum(1)], [Grid4TransitionsEnum(3)], [Grid4TransitionsEnum(3)],
                                  [Grid4TransitionsEnum(1)], [Grid4TransitionsEnum(3)], [Grid4TransitionsEnum(1)], [Grid4TransitionsEnum(0)],
                                  [Grid4TransitionsEnum(1)], [Grid4TransitionsEnum(3)]],
                agent_targets=[(39, 8), (10, 40), (42, 22), (18, 5), (39, 8), (12, 40), (31, 27), (39, 8), (8, 27), (44, 22)],
                agent_speeds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    env_agents = EnvAgent.from_line(line)
    assert env_agents[0].initial_position == (11, 40)
    assert env_agents[0].initial_direction == 3
    assert env_agents[0].target == (39, 8)
    assert env_agents[1].initial_position == (38, 8)
    assert env_agents[1].initial_direction == 1
    assert env_agents[1].target == (10, 40)
    assert env_agents[2].initial_position == (17, 5)
    assert env_agents[2].initial_direction == 3
    assert env_agents[2].target == (42, 22)
    assert env_agents[3].initial_position == (41, 22)
    assert env_agents[3].initial_direction == 3
    assert env_agents[3].target == (18, 5)
    assert env_agents[4].initial_position == (11, 40)
    assert env_agents[4].initial_direction == 1
    assert env_agents[4].target == (39, 8)
    assert env_agents[5].initial_position == (38, 8)
    assert env_agents[5].initial_direction == 3
    assert env_agents[5].target == (12, 40)
    assert env_agents[6].initial_position == (38, 8)
    assert env_agents[6].initial_direction == 1
    assert env_agents[6].target == (31, 27)
    assert env_agents[7].initial_position == (31, 26)
    assert env_agents[7].initial_direction == 0
    assert env_agents[7].target == (39, 8)
    assert env_agents[8].initial_position == (41, 22)
    assert env_agents[8].initial_direction == 1
    assert env_agents[8].target == (8, 27)
    assert env_agents[9].initial_position == (9, 27)
    assert env_agents[9].initial_direction == 3
    assert env_agents[9].target == (44, 22)
