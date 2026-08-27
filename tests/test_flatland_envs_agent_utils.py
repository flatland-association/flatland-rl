from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import (Agent, EnvAgent, _agent_tuple_targets, _filter_valid_target_entry_points,
                                       _sanitize_entry_point, load_env_agent, virtual_entry_point, with_direction)
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.rewards import Rewards
from flatland.envs.step_utils.malfunction_handler import MalfunctionHandler
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState
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
    agent_positions = [[[(11, 40)]], [[(38, 8)]], [[(17, 5)]], [[(41, 22)]], [[(11, 40)]], [[(38, 8)]], [[(38, 8)]], [[(31, 26)]], [[(41, 22)]], [[(9, 27)]]]
    agent_directions = [[[Grid4TransitionsEnum(3)]], [[Grid4TransitionsEnum(1)]], [[Grid4TransitionsEnum(3)]], [[Grid4TransitionsEnum(3)]],
                        [[Grid4TransitionsEnum(1)]], [[Grid4TransitionsEnum(3)]], [[Grid4TransitionsEnum(1)]], [[Grid4TransitionsEnum(0)]],
                        [[Grid4TransitionsEnum(1)]], [[Grid4TransitionsEnum(3)]]]
    agent_targets = [(39, 8), (10, 40), (42, 22), (18, 5), (39, 8), (12, 40), (31, 27), (39, 8), (8, 27), (44, 22)]
    agent_waypoints = {i: [[Waypoint(fpa, fda) for fpa, fda in zip(pa, da)] for pa, da in zip(pas, das)] +
                          [[Waypoint(t, d) for d in Grid4TransitionsEnum]] for i, (pas, das, t) in
                       enumerate(zip(agent_positions, agent_directions, agent_targets))}
    line = Line(agent_waypoints=agent_waypoints, agent_speeds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    env_agents = EnvAgent.from_line(line)
    assert env_agents[0].initial_entry_point[0] == (11, 40)
    assert env_agents[0].initial_entry_point[1] == 3
    assert next(iter(env_agents[0].targets))[0] == (39, 8)
    assert env_agents[1].initial_entry_point[0] == (38, 8)
    assert env_agents[1].initial_entry_point[1] == 1
    assert next(iter(env_agents[1].targets))[0] == (10, 40)
    assert env_agents[2].initial_entry_point[0] == (17, 5)
    assert env_agents[2].initial_entry_point[1] == 3
    assert next(iter(env_agents[2].targets))[0] == (42, 22)
    assert env_agents[3].initial_entry_point[0] == (41, 22)
    assert env_agents[3].initial_entry_point[1] == 3
    assert next(iter(env_agents[3].targets))[0] == (18, 5)
    assert env_agents[4].initial_entry_point[0] == (11, 40)
    assert env_agents[4].initial_entry_point[1] == 1
    assert next(iter(env_agents[4].targets))[0] == (39, 8)
    assert env_agents[5].initial_entry_point[0] == (38, 8)
    assert env_agents[5].initial_entry_point[1] == 3
    assert next(iter(env_agents[5].targets))[0] == (12, 40)
    assert env_agents[6].initial_entry_point[0] == (38, 8)
    assert env_agents[6].initial_entry_point[1] == 1
    assert next(iter(env_agents[6].targets))[0] == (31, 27)
    assert env_agents[7].initial_entry_point[0] == (31, 26)
    assert env_agents[7].initial_entry_point[1] == 0
    assert next(iter(env_agents[7].targets))[0] == (39, 8)
    assert env_agents[8].initial_entry_point[0] == (41, 22)
    assert env_agents[8].initial_entry_point[1] == 1
    assert next(iter(env_agents[8].targets))[0] == (8, 27)
    assert env_agents[9].initial_entry_point[0] == (9, 27)
    assert env_agents[9].initial_entry_point[1] == 3
    assert next(iter(env_agents[9].targets))[0] == (44, 22)


def test_load_env_agent_fallback_waypoints():
    """Regression test: `load_env_agent`'s fallback `waypoints` construction (used when the legacy `Agent`
    NamedTuple carries no `waypoints`) must produce a well-formed `List[List[Waypoint]]`, i.e. every entry -
    including the initial stop - must itself be a list, not a bare `Waypoint`; and the target group must be
    filtered to the arrival directions valid on `rail`."""
    rail, _, _ = make_oval_rail()
    target = (1, 2)  # a straight horizontal track cell: only EAST/WEST are valid entry points here
    agent_tuple = Agent(
        initial_position=(0, 0),
        initial_direction=Grid4TransitionsEnum(0),
        direction=Grid4TransitionsEnum(0),
        targets={(target, d) for d in Grid4TransitionsEnum},
        moving=False,
        earliest_departure=0,
        latest_arrival=100,
        handle=0,
        position=None,
        arrival_time=None,
        old_direction=None,
        old_position=None,
        speed_counter=SpeedCounter(1.0),
        action_saver=None,
        state_machine=TrainStateMachine(initial_state=TrainState.WAITING),
        malfunction_handler=MalfunctionHandler(),
    )
    env_agent = load_env_agent(agent_tuple, rail)

    assert isinstance(env_agent.waypoints[0], list)
    assert env_agent.waypoints[0] == [Waypoint((0, 0), Grid4TransitionsEnum(0))]
    # the four exploded target directions are filtered down to the two rail-valid ones, and `targets` stays in sync
    assert env_agent.waypoints[1] == [Waypoint(target, Grid4TransitionsEnum(1)), Waypoint(target, Grid4TransitionsEnum(3))]
    assert env_agent.targets == {(target, Grid4TransitionsEnum(1)), (target, Grid4TransitionsEnum(3))}

    # must not raise - Rewards._sanitize_waypoints() assumes every entry is iterable.
    Rewards._sanitize_waypoints(env_agent.waypoints)


def test_filter_valid_target_entry_points():
    """`_filter_valid_target_entry_points` must keep only rail-valid directions at a group's position, and
    must explode a legacy `None`-direction placeholder into concrete valid directions instead of dropping it."""
    rail, _, _ = make_oval_rail()
    position = (1, 2)  # a straight horizontal track cell: only EAST/WEST are valid entry points here

    assert _filter_valid_target_entry_points(rail, [
        Waypoint(position, Grid4TransitionsEnum(1)), Waypoint(position, Grid4TransitionsEnum(3)), Waypoint(position, Grid4TransitionsEnum(2))
    ]) == [Waypoint(position, Grid4TransitionsEnum(1)), Waypoint(position, Grid4TransitionsEnum(3))]

    assert _filter_valid_target_entry_points(rail, [Waypoint(position, None)]) == [
        Waypoint(position, Grid4TransitionsEnum(1)), Waypoint(position, Grid4TransitionsEnum(3))]


def test_agent_tuple_targets():
    """`_agent_tuple_targets` must pass a concrete `(position, direction)` set through unchanged, and must
    explode a legacy bare `(row, col)` position (from an env pickled before `Agent.target` became
    `Agent.targets`) into one entry point per direction."""

    def _agent(targets):
        return Agent(
            initial_position=(0, 0), initial_direction=Grid4TransitionsEnum(0), direction=Grid4TransitionsEnum(0),
            targets=targets, moving=False, earliest_departure=0, latest_arrival=100, handle=0, position=None,
            arrival_time=None, old_direction=None, old_position=None, speed_counter=SpeedCounter(1.0),
            action_saver=None, state_machine=TrainStateMachine(initial_state=TrainState.WAITING),
            malfunction_handler=MalfunctionHandler(),
        )

    concrete = {((3, 3), Grid4TransitionsEnum(1)), ((3, 3), Grid4TransitionsEnum(3))}
    assert _agent_tuple_targets(_agent(concrete)) == concrete

    # legacy positional pickle: the `targets` slot holds a bare (row, col) position, not a set
    assert _agent_tuple_targets(_agent((3, 3))) == {((3, 3), d) for d in Grid4TransitionsEnum}


def test_with_direction():
    assert with_direction(((3, 3), 1), 2) == ((3, 3), 2)
    assert with_direction(None, 2) == (None, 2)


def test_virtual_entry_point():
    """
    `virtual_entry_point` must return a real, rail-valid entry point for a `TrainState.DONE` agent
    even after it has been removed from the map (`current_entry_point` is `None`) - unlike
    `current_entry_point` itself, which observations/predictions cannot compute anything useful from
    once the agent is gone.
    """
    rail, rail_map, optionals = make_oval_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    agent = env.agents[0]

    # off map: initial_entry_point
    assert virtual_entry_point(agent) == agent.initial_entry_point

    # on map: current_entry_point
    agent.current_entry_point = agent.initial_entry_point
    agent._set_state(TrainState.MOVING)
    # design: actions applied at cell entry -- keep the current_entry_point/next_entry_point
    # invariant (both set and different while on-map) even for this direct test-harness assignment.
    transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, agent.current_entry_point)
    assert transition is not None
    agent.next_entry_point = _sanitize_entry_point(transition)
    assert virtual_entry_point(agent) == agent.current_entry_point

    # done, agent still has a (now stale) current_entry_point - which must actually be one of the
    # target alternatives, since DONE is only reachable by having arrived at one
    agent.current_entry_point = next(iter(agent.targets))
    agent._set_state(TrainState.DONE)
    # mirrors what AbstractRailEnv.handle_done_state() does before current_entry_point is cleared
    agent.target_entry_point = agent.current_entry_point
    assert virtual_entry_point(agent) in agent.targets
    assert virtual_entry_point(agent) == agent.target_entry_point

    # done and removed from the map (`current_entry_point` is None): still a real entry point
    agent.current_entry_point = None
    entry_point = virtual_entry_point(agent)
    assert entry_point is not None
    assert entry_point in agent.targets
    assert entry_point == agent.target_entry_point
