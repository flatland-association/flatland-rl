import time

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_diamond_crossing_rail


def test_diamond_crossing_without_over_and_underpasses(rendering: bool = False):
    rail, rail_map, optionals = make_diamond_crossing_rail()

    env = RailEnv(
        width=rail_map.shape[1],
        height=rail_map.shape[0],
        rail_generator=rail_from_grid_transition_map(rail, optionals),
        line_generator=sparse_line_generator(),
        number_of_agents=2,
        obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
        record_steps=True
    )

    env.reset()
    env._max_episode_steps = 555

    # set the initial position
    agent_0 = env.agents[0]
    agent_0.initial_position = (3, 0)  # one cell ahead of diamond crossing facing east
    agent_0.position = (3, 0)  # one cell ahead of diamond crossing facing east
    agent_0.direction = 3  # east
    agent_0.initial_direction = 3  # east
    agent_0.target = (3, 9)  # east dead-end
    agent_0.moving = True
    agent_0.latest_arrival = 999
    agent_0._set_state(TrainState.MOVING)

    agent_1 = env.agents[1]
    agent_1.initial_position = (1, 2)  # one cell ahead of diamond crossing facing south
    agent_1.position = (1, 2)  # one cell ahead of diamond crossing facing south
    agent_1.direction = 2  # south
    agent_1.initial_direction = 2  # south
    agent_1.target = (6, 2)  # south dead-end
    agent_1.moving = True
    agent_1.latest_arrival = 999
    agent_1._set_state(TrainState.MOVING)

    env.distance_map._compute(env.agents, env.rail)
    done = False
    env_renderer = None
    if rendering:
        env_renderer = RenderTool(env)
    while not done:
        _, _, dones, _ = env.step({
            0: RailEnvActions.MOVE_FORWARD,
            1: RailEnvActions.MOVE_FORWARD,
        })
        done = dones["__all__"]
        if env_renderer is not None:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            time.sleep(1.2)

    waypoints = []
    for agent_states in env.cur_episode:
        cur = []
        for agent_state in agent_states:
            r, c, d, _, _, _ = agent_state
            cur.append(Waypoint((r, c), d))
        waypoints.append(cur)
    expected = [
        # agent 0 and agent 1 both want to enter the diamond-crossing at (3,2)
        [Waypoint(position=(3, 1), direction=1), Waypoint(position=(2, 2), direction=2)],
        # agent 1 waits until agent 0 has passed the diamond crossing at (3,2)
        [Waypoint(position=(3, 2), direction=1), Waypoint(position=(2, 2), direction=2)],
        [Waypoint(position=(3, 3), direction=1), Waypoint(position=(3, 2), direction=2)],
        [Waypoint(position=(3, 4), direction=1), Waypoint(position=(4, 2), direction=2)],
        [Waypoint(position=(3, 5), direction=1), Waypoint(position=(5, 2), direction=2)],
        [Waypoint(position=(3, 6), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(3, 7), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(3, 8), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(0, 0), direction=1), Waypoint(position=(0, 0), direction=2)]
    ]
    assert expected == waypoints, waypoints


def test_diamond_crossing_with_over_and_underpasses(rendering: bool = False):
    rail, rail_map, optionals = make_diamond_crossing_rail()

    env = RailEnv(
        width=rail_map.shape[1],
        height=rail_map.shape[0],
        rail_generator=rail_from_grid_transition_map(rail, optionals),
        line_generator=sparse_line_generator(),
        number_of_agents=2,
        obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
        record_steps=True
    )

    env.reset()
    env._max_episode_steps = 555

    # set the initial position
    agent_0 = env.agents[0]
    agent_0.initial_position = (3, 0)  # one cell ahead of diamond crossing facing east
    agent_0.position = (3, 0)  # one cell ahead of diamond crossing facing east
    agent_0.direction = 3  # east
    agent_0.initial_direction = 3  # east
    agent_0.target = (3, 9)  # east dead-end
    agent_0.moving = True
    agent_0.latest_arrival = 999
    agent_0._set_state(TrainState.MOVING)

    agent_1 = env.agents[1]
    agent_1.initial_position = (1, 2)  # one cell ahead of diamond crossing facing south
    agent_1.position = (1, 2)  # one cell ahead of diamond crossing facing south
    agent_1.direction = 2  # south
    agent_1.initial_direction = 2  # south
    agent_1.target = (6, 2)  # south dead-end
    agent_1.moving = True
    agent_1.latest_arrival = 999
    agent_1._set_state(TrainState.MOVING)

    env.level_free_positions.add((3, 2))

    env.distance_map._compute(env.agents, env.rail)
    done = False
    env_renderer = None
    if rendering:
        env_renderer = RenderTool(env)
    while not done:
        _, _, dones, _ = env.step({
            0: RailEnvActions.MOVE_FORWARD,
            1: RailEnvActions.MOVE_FORWARD,
        })
        done = dones["__all__"]
        if env_renderer is not None:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            time.sleep(1.2)

    waypoints = []
    for agent_states in env.cur_episode:
        cur = []
        for agent_state in agent_states:
            r, c, d, _, _, _ = agent_state
            cur.append(Waypoint((r, c), d))
        waypoints.append(cur)
    expected = [
        # agent 0 and agent 1 both want to enter the diamond-crossing at (3,2)
        [Waypoint(position=(3, 1), direction=1), Waypoint(position=(2, 2), direction=2)],
        # agent 0 and agent 1 can enter the level-free diamond crossing at (3,2)
        [Waypoint(position=(3, 2), direction=1), Waypoint(position=(3, 2), direction=2)],
        [Waypoint(position=(3, 3), direction=1), Waypoint(position=(4, 2), direction=2)],
        [Waypoint(position=(3, 4), direction=1), Waypoint(position=(5, 2), direction=2)],
        [Waypoint(position=(3, 5), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(3, 6), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(3, 7), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(3, 8), direction=1), Waypoint(position=(0, 0), direction=2)],
        [Waypoint(position=(0, 0), direction=1), Waypoint(position=(0, 0), direction=2)]
    ]
    assert expected == waypoints, waypoints


def test_diamond_crossing_with_over_and_underpasses_head_on(rendering: bool = False):
    rail, rail_map, optionals = make_diamond_crossing_rail()

    env = RailEnv(
        width=rail_map.shape[1],
        height=rail_map.shape[0],
        rail_generator=rail_from_grid_transition_map(rail, optionals),
        line_generator=sparse_line_generator(),
        number_of_agents=2,
        obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
        record_steps=True
    )
    env.reset()
    env._max_episode_steps = 5

    # set the initial position
    agent_0 = env.agents[0]
    agent_0.initial_position = (3, 0)  # one cell ahead of diamond crossing facing east
    agent_0.position = (3, 0)  # one cell ahead of diamond crossing facing east
    agent_0.direction = 3  # east
    agent_0.initial_direction = 3  # east
    agent_0.target = (3, 9)  # east dead-end
    agent_0.moving = True
    agent_0.latest_arrival = 999
    agent_0._set_state(TrainState.MOVING)

    agent_1 = env.agents[1]
    agent_1.initial_position = (3, 4)  # one cell ahead of diamond crossing facing west
    agent_1.position = (3, 4)  # one cell ahead of diamond crossing facing west
    agent_1.direction = 3  # west
    agent_1.initial_direction = 3  # west
    agent_1.target = (3, 0)  # west dead-end
    agent_1.moving = True
    agent_1.latest_arrival = 999
    agent_1._set_state(TrainState.MOVING)

    env.level_free_positions.add((3, 2))

    env.distance_map._compute(env.agents, env.rail)
    done = False
    env_renderer = None
    if rendering:
        env_renderer = RenderTool(env)
    while not done:
        _, _, dones, _ = env.step({
            0: RailEnvActions.MOVE_FORWARD,
            1: RailEnvActions.MOVE_FORWARD,
        })
        done = dones["__all__"]
        if env_renderer is not None:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            time.sleep(1.2)

    waypoints = []
    for agent_states in env.cur_episode:
        cur = []
        for agent_state in agent_states:
            r, c, d, _, _, _ = agent_state
            cur.append(Waypoint((r, c), d))
        waypoints.append(cur)
    expected = [
        # agent 0 and agent 1 both want to enter the diamond-crossing at (3,2)
        [Waypoint(position=(3, 1), direction=1), Waypoint(position=(3, 3), direction=3)],
        # agent 0 and agent 1 are stuck (head-on)
        [Waypoint(position=(3, 2), direction=1), Waypoint(position=(3, 3), direction=3)],
        [Waypoint(position=(3, 2), direction=1), Waypoint(position=(3, 3), direction=3)],
        [Waypoint(position=(3, 2), direction=1), Waypoint(position=(3, 3), direction=3)],
        [Waypoint(position=(3, 2), direction=1), Waypoint(position=(3, 3), direction=3)]]
    assert expected == waypoints, waypoints
