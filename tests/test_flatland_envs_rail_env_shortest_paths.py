import sys
from typing import List

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_disconnected_simple_rail, make_simple_rail_with_alternatives


def test_get_shortest_paths_unreachable():
    rail, rail_map, optionals = make_disconnected_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv())
    env.reset()

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({}) # DO_NOTHING for all agents

    # set the initial position
    agent = env.agents[0]
    agent.position = (3, 1)  # west dead-end
    agent.initial_position = (3, 1)  # west dead-end
    agent.direction = Grid4TransitionsEnum.WEST
    agent.target = (3, 9)  # east dead-end
    agent.moving = True

    env.reset(False, False)

    actual = env.distance_map.get_shortest_paths()
    expected = {0: None}

    assert actual[0] == expected[0], "actual={},expected={}".format(actual[0], expected[0])


# todo file test_002.pkl has to be generated automatically
def test_get_shortest_paths():
    #env = load_flatland_environment_from_file('test_002.mpk', 'env_data.tests')
    env, env_dict = RailEnvPersister.load_new("test_002.mpk", "env_data.tests")

    #print("env len(agents): ", len(env.agents))
    #print(env.distance_map)
    #print("env number_of_agents:", env.number_of_agents)

    #print("env agents:", env.agents)

    #env.distance_map.reset(env.agents, env.rail)

    #actual = get_shortest_paths(env.distance_map)
    #print("shortest paths:", actual)

    #print(env.distance_map)
    #print("Dist map agents:", env.distance_map.agents)

    #print("\nenv reset()")
    env.reset()
    actual = env.distance_map.get_shortest_paths()
    #print("env agents: ", len(env.agents))
    #print("env number_of_agents: ", env.number_of_agents)



    assert len(actual) == 2, "get_shortest_paths should return a dict of length 2"

    expected = {
        0: [
            Waypoint(position=(1, 1), direction=1),
            Waypoint(position=(1, 2), direction=1),
            Waypoint(position=(1, 3), direction=1),
            Waypoint(position=(2, 3), direction=2),
            Waypoint(position=(2, 4), direction=1),
            Waypoint(position=(2, 5), direction=1),
            Waypoint(position=(2, 6), direction=1),
            Waypoint(position=(2, 7), direction=1),
            Waypoint(position=(2, 8), direction=1),
            Waypoint(position=(2, 9), direction=1),
            Waypoint(position=(2, 10), direction=1),
            Waypoint(position=(2, 11), direction=1),
            Waypoint(position=(2, 12), direction=1),
            Waypoint(position=(2, 13), direction=1),
            Waypoint(position=(2, 14), direction=1),
            Waypoint(position=(2, 15), direction=1),
            Waypoint(position=(2, 16), direction=1),
            Waypoint(position=(2, 17), direction=1),
            Waypoint(position=(2, 18), direction=1)],
        1: [
            Waypoint(position=(3, 18), direction=3),
            Waypoint(position=(3, 17), direction=3),
            Waypoint(position=(3, 16), direction=3),
            Waypoint(position=(2, 16), direction=0),
            Waypoint(position=(2, 15), direction=3),
            Waypoint(position=(2, 14), direction=3),
            Waypoint(position=(2, 13), direction=3),
            Waypoint(position=(2, 12), direction=3),
            Waypoint(position=(2, 11), direction=3),
            Waypoint(position=(2, 10), direction=3),
            Waypoint(position=(2, 9), direction=3),
            Waypoint(position=(2, 8), direction=3),
            Waypoint(position=(2, 7), direction=3),
            Waypoint(position=(2, 6), direction=3),
            Waypoint(position=(2, 5), direction=3),
            Waypoint(position=(2, 4), direction=3),
            Waypoint(position=(2, 3), direction=3),
            Waypoint(position=(2, 2), direction=3),
            Waypoint(position=(2, 1), direction=3)
            ]
    }

    for iA, lWP in expected.items():
        _assert_paths_equal(iA, actual[iA], lWP)

def _assert_paths_equal(iAgent:int, actual:List[Waypoint], expected:List[Waypoint]):
    assert len(actual) == len(expected), f"Lengths differ: actual={len(actual)}, expected={len(expected)}"
    for iWP, (wpA, wpE) in enumerate(zip(actual, expected)):
        assert wpA.position == wpE.position, f"Agent {iAgent} Waypoints at step {iWP} differ: actual={wpA.position}, expected={wpE.position}"
        assert wpA.direction == wpE.direction, f"Agent {iAgent} Waypoint directions at step {iWP} differ:actual={wpA.direction}, expected={wpE.direction}"

# todo file test_002.pkl has to be generated automatically
def test_get_shortest_paths_max_depth():
    #env = load_flatland_environment_from_file('test_002.pkl', 'env_data.tests')
    env, _ = RailEnvPersister.load_new("test_002.mpk", "env_data.tests")
    env.reset()
    actual = env.distance_map.get_shortest_paths(max_depth=2)

    expected = {
        0: [
            Waypoint(position=(1, 1), direction=1),
            Waypoint(position=(1, 2), direction=1)
        ],
        1: [
            Waypoint(position=(3, 18), direction=3),
            Waypoint(position=(3, 17), direction=3),
        ]
    }

    for iA, lWP in expected.items():
        _assert_paths_equal(iA, actual[iA], lWP)



# todo file Level_distance_map_shortest_path.pkl has to be generated automatically
def test_get_shortest_paths_agent_handle():
    #env = load_flatland_environment_from_file('Level_distance_map_shortest_path.pkl', 'env_data.tests')
    env, _ = RailEnvPersister.load_new("Level_distance_map_shortest_path.mpk", "env_data.tests")
    env.reset()
    actual = env.distance_map.get_shortest_paths(agent_handle=6)

    print(actual, file=sys.stderr)

    expected = {6:
                    [Waypoint(position=(5, 5),
                              direction=0),
                     Waypoint(position=(4, 5),
                              direction=0),
                     Waypoint(position=(3, 5),
                              direction=0),
                     Waypoint(position=(2, 5),
                              direction=0),
                     Waypoint(position=(1, 5),
                              direction=0),
                     Waypoint(position=(0, 5),
                              direction=0),
                     Waypoint(position=(0, 6),
                              direction=1),
                     Waypoint(position=(0, 7), direction=1),
                     Waypoint(position=(0, 8),
                              direction=1),
                     Waypoint(position=(0, 9),
                              direction=1),
                     Waypoint(position=(0, 10),
                              direction=1),
                     Waypoint(position=(1, 10),
                              direction=2),
                     Waypoint(position=(2, 10),
                              direction=2),
                     Waypoint(position=(3, 10),
                              direction=2),
                     Waypoint(position=(4, 10),
                              direction=2),
                     Waypoint(position=(5, 10),
                              direction=2),
                     Waypoint(position=(6, 10),
                              direction=2),
                     Waypoint(position=(7, 10),
                              direction=2),
                     Waypoint(position=(8, 10),
                              direction=2),
                     Waypoint(position=(9, 10),
                              direction=2),
                     Waypoint(position=(10, 10),
                              direction=2),
                     Waypoint(position=(11, 10),
                              direction=2),
                     Waypoint(position=(12, 10),
                              direction=2),
                     Waypoint(position=(13, 10),
                              direction=2),
                     Waypoint(position=(14, 10),
                              direction=2),
                     Waypoint(position=(15, 10),
                              direction=2),
                     Waypoint(position=(16, 10),
                              direction=2),
                     Waypoint(position=(17, 10),
                              direction=2),
                     Waypoint(position=(18, 10),
                              direction=2),
                     Waypoint(position=(19, 10),
                              direction=2),
                     Waypoint(position=(20, 10),
                              direction=2),
                     Waypoint(position=(20, 9),
                              direction=3),
                     Waypoint(position=(20, 8),
                              direction=3),
                     Waypoint(position=(21, 8),
                              direction=2),
                     Waypoint(position=(21, 7),
                              direction=3),
                     Waypoint(position=(21, 6),
                              direction=3),
                     Waypoint(position=(21, 5),
                              direction=3)
                     ]}

    for iA, lWP in expected.items():
        _assert_paths_equal(iA, actual[iA], lWP)


def test_get_k_shortest_paths(rendering=False):
    rail, rail_map, optionals = make_simple_rail_with_alternatives()

    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv(),
                  )
    env.reset()

    initial_position = (3, 1)  # west dead-end
    initial_direction = Grid4TransitionsEnum.WEST  # west
    target_position = (3, 9)  # east

    # set the initial position
    agent = env.agents[0]
    agent.position = initial_position
    agent.initial_position = initial_position
    agent.direction = initial_direction
    agent.target = target_position  # east dead-end
    agent.moving = True

    env.reset(False, False)
    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=False)
        input()

    actual = set(get_k_shortest_paths(
        env=env,
        source_position=initial_position,  # west dead-end
        source_direction=int(initial_direction),  # east
        target_position=target_position,
        k=10
    ))

    expected = set([
        (
            Waypoint(position=(3, 1), direction=3),
            Waypoint(position=(3, 0), direction=3),
            Waypoint(position=(3, 1), direction=1),
            Waypoint(position=(3, 2), direction=1),
            Waypoint(position=(3, 3), direction=1),
            Waypoint(position=(2, 3), direction=0),
            Waypoint(position=(1, 3), direction=0),
            Waypoint(position=(0, 3), direction=0),
            Waypoint(position=(0, 4), direction=1),
            Waypoint(position=(0, 5), direction=1),
            Waypoint(position=(0, 6), direction=1),
            Waypoint(position=(0, 7), direction=1),
            Waypoint(position=(0, 8), direction=1),
            Waypoint(position=(0, 9), direction=1),
            Waypoint(position=(1, 9), direction=2),
            Waypoint(position=(2, 9), direction=2),
            Waypoint(position=(3, 9), direction=2)),
        (
            Waypoint(position=(3, 1), direction=3),
            Waypoint(position=(3, 0), direction=3),
            Waypoint(position=(3, 1), direction=1),
            Waypoint(position=(3, 2), direction=1),
            Waypoint(position=(3, 3), direction=1),
            Waypoint(position=(3, 4), direction=1),
            Waypoint(position=(3, 5), direction=1),
            Waypoint(position=(3, 6), direction=1),
            Waypoint(position=(4, 6), direction=2),
            Waypoint(position=(5, 6), direction=2),
            Waypoint(position=(6, 6), direction=2),
            Waypoint(position=(5, 6), direction=0),
            Waypoint(position=(4, 6), direction=0),
            Waypoint(position=(4, 7), direction=1),
            Waypoint(position=(4, 8), direction=1),
            Waypoint(position=(4, 9), direction=1),
            Waypoint(position=(3, 9), direction=0))
    ])

    assert actual == expected, "Sets are different:\nactual={},\nexpected={}".format(actual, expected)

def main():
    test_get_shortest_paths()

if __name__ == "__main__":
    main()
