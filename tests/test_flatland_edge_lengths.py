import math
import pickle
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.env_generation.env_generator import env_generator
from flatland.envs.agent_utils import _sanitize_entry_point
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_state_machine_wrapper import RailEnvStateMachineWrapper
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rewards import DefaultRewards
from flatland.core.transition_map import DEFAULT_EDGE_LENGTH, validated_edge_length
from flatland.envs.step_utils.speed_counter import SpeedCounter, cached_cell_exit
from flatland.envs.step_utils.states import TrainState
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy

EAST = Grid4TransitionsEnum.EAST
NORTH = Grid4TransitionsEnum.NORTH


@pytest.mark.parametrize("edge_length", [Fraction(1), Fraction(2), Fraction(5, 2)])
def test_distance_after_crossing(edge_length):
    # mid-edge: no wrap
    assert SpeedCounter.distance_after_crossing(edge_length - 1, Fraction(1, 2), edge_length) == edge_length - Fraction(1, 2)
    # exact fit: lands at the start of the next edge
    assert SpeedCounter.distance_after_crossing(edge_length - Fraction(1, 2), Fraction(1, 2), edge_length) == 0
    # overshoot carried into the next edge
    assert SpeedCounter.distance_after_crossing(edge_length - Fraction(1, 4), Fraction(3, 4), edge_length) == Fraction(1, 2)
    # banked at the boundary with full speed: wraps to 0, never lands on the next edge's exit
    assert SpeedCounter.distance_after_crossing(edge_length, Fraction(1), edge_length) == 0
    assert SpeedCounter.distance_after_crossing(None, None, edge_length) is None


@pytest.mark.parametrize("edge_length", [Fraction(1), Fraction(2), Fraction(5, 2)])
def test_distance_without_crossing(edge_length):
    assert SpeedCounter.distance_without_crossing(edge_length - Fraction(1, 4), Fraction(3, 4), edge_length) == edge_length
    assert SpeedCounter.distance_without_crossing(edge_length - 1, Fraction(1, 2), edge_length) == edge_length - Fraction(1, 2)
    assert SpeedCounter.distance_without_crossing(None, None, edge_length) is None


@pytest.mark.parametrize("edge_length", [Fraction(1), Fraction(2), Fraction(5, 2)])
def test_cached_cell_exit(edge_length):
    assert cached_cell_exit(Fraction(1), Fraction(1), edge_length - 1, edge_length)
    assert not cached_cell_exit(Fraction(1), Fraction(1, 2), edge_length - 1, edge_length)
    assert cached_cell_exit(Fraction(1), None, None, None)


@pytest.mark.parametrize("length", [Fraction(1, 2), 0, -1, 0.75])
def test_validated_edge_length_rejects_short_edges(length):
    with pytest.raises(ValueError):
        validated_edge_length(length)


def test_validated_edge_length_converts():
    assert validated_edge_length(1.5) == Fraction(3, 2)
    assert validated_edge_length(3) == Fraction(3)


def test_speed_counter_is_cell_exit_uses_edge_length():
    sc = SpeedCounter(max_speed=1)
    sc.set(Fraction(1), Fraction(0), Fraction(3))
    assert sc.edge_length == Fraction(3)
    assert not sc.is_cell_exit()
    sc.set(Fraction(1), Fraction(2), Fraction(3))
    assert sc.is_cell_exit()


def test_speed_counter_is_cell_entry_long_to_short_and_back():
    sc = SpeedCounter(max_speed=1)
    sc.set(Fraction(3, 4), Fraction(0), Fraction(2))
    assert sc.is_cell_entry  # bootstrap onto the map
    sc.set(Fraction(3, 4), Fraction(3, 2), Fraction(2))
    assert not sc.is_cell_entry  # 3/4 + 3/4 < 2: still mid-edge
    # 3/2 + 3/4 >= 2: crossing into a unit-length edge
    sc.set(Fraction(3, 4), SpeedCounter.distance_after_crossing(Fraction(3, 2), Fraction(3, 4), Fraction(2)), Fraction(1))
    assert sc.is_cell_entry
    assert sc.distance == Fraction(1, 4)
    # 1/4 + 3/4 >= 1: crossing into a long edge again
    sc.set(Fraction(3, 4), SpeedCounter.distance_after_crossing(Fraction(1, 4), Fraction(3, 4), Fraction(1)), Fraction(2))
    assert sc.is_cell_entry
    assert sc.distance == Fraction(0)


def test_speed_counter_is_cell_entry_false_when_banked_on_long_edge():
    sc = SpeedCounter(max_speed=1)
    sc.set(Fraction(1), Fraction(3, 2), Fraction(2))
    # denied crossing: banked at the long edge's boundary
    sc.set(Fraction(0), SpeedCounter.distance_without_crossing(Fraction(3, 2), Fraction(1), Fraction(2)), Fraction(2))
    assert sc.distance == Fraction(2)
    assert not sc.is_cell_entry


def test_speed_counter_pickle_round_trip():
    sc = SpeedCounter(max_speed=1)
    sc.set(Fraction(1), Fraction(1, 2), Fraction(5, 2))
    assert pickle.loads(pickle.dumps(sc)) == sc
    assert pickle.loads(pickle.dumps(sc)).edge_length == Fraction(5, 2)


def test_speed_counter_old_pickle_defaults_to_unit_edge_length():
    sc = SpeedCounter(max_speed=1)
    state = {"speed": Fraction(1), "max_speed": Fraction(1), "distance": Fraction(0), "is_cell_entry": True}
    sc.__setstate__(state)
    assert sc.edge_length == DEFAULT_EDGE_LENGTH
    state = {"speed": None, "max_speed": Fraction(1), "distance": None, "is_cell_entry": False}
    sc.__setstate__(state)
    assert sc.edge_length is None


def _straight_rail(n_cells: int):
    """ Straight east-west corridor of `n_cells` cells: dead end - straight* - dead end. """
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end = cells[7]
    straight = transitions.rotate_transition(cells[1], 90)
    opens_east = transitions.rotate_transition(dead_end, 270)
    opens_west = transitions.rotate_transition(dead_end, 90)
    rail = RailGridTransitionMap(width=n_cells, height=1, transitions=transitions)
    rail.grid = np.array([[opens_east] + [straight] * (n_cells - 2) + [opens_west]], dtype=np.uint16)
    agents_hints = {
        'city_positions': [(0, 0), (0, n_cells - 1)],
        'train_stations': [[((0, 0), 0)], [((0, n_cells - 1), 0)]],
        'city_orientations': [0, 0],
    }
    return rail, {'agents_hints': agents_hints}


def _edge(c: int):
    """ The edge through cell (0, c) heading east (c >= 1). """
    return ((0, c), EAST), ((0, c + 1), EAST)


def _with_edge_lengths(optionals, edge_lengths):
    """ Rail generator optionals with `edge_lengths` added (if any). """
    return {**optionals, "edge_lengths": edge_lengths} if edge_lengths else optionals


def _run_straight(edge_lengths, max_speed, n_cells=5, remove_agents_at_target=True):
    """
    Single agent bootstrapped moving at full `max_speed` at the start of dead-end cell (0, 0) (entered heading west,
    hence leaving east), target (0, n_cells - 1),
    always MOVE_FORWARD. Returns the per-step list of (current_entry_point, distance) after each step,
    until the agent is done.
    """
    rail, optionals = _straight_rail(n_cells)
    env = RailEnv(width=n_cells, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, edge_lengths)),
                  line_generator=sparse_line_generator(), number_of_agents=1, remove_agents_at_target=remove_agents_at_target)
    env = RailEnvStateMachineWrapper(env)
    env.reset()
    env._max_episode_steps = 1000
    agent = env.agents[0]
    agent.initial_entry_point = ((0, 0), Grid4TransitionsEnum.WEST)
    agent.current_entry_point = ((0, 0), Grid4TransitionsEnum.WEST)
    agent.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, agent.current_entry_point))
    agent.targets = {((0, n_cells - 1), d) for d in Grid4TransitionsEnum}
    agent._set_state(TrainState.MOVING)
    agent.speed_counter = SpeedCounter(max_speed=max_speed, speed=max_speed)
    agent.speed_counter.set(max_speed, Fraction(0), env.rail.edge_length(agent.current_entry_point, agent.next_entry_point))

    trajectory = []
    for _ in range(100):
        env.step({0: RailEnvActions.MOVE_FORWARD})
        trajectory.append((agent.current_entry_point, agent.speed_counter.distance))
        if agent.state == TrainState.DONE:
            break
    assert agent.state == TrainState.DONE
    return env, trajectory


def _steps_in(trajectory, c):
    return sum(1 for ep, _ in trajectory if ep is not None and ep[0] == (0, c))


@pytest.mark.parametrize("max_speed,length,expected_steps_in_long_cell", [
    (Fraction(1), Fraction(1), 1),
    (Fraction(1), Fraction(3), 3),
    (Fraction(1, 2), Fraction(1), 2),
    (Fraction(1, 2), Fraction(3), 6),
    (Fraction(1), Fraction(5, 2), 3),
])
def test_long_edge_takes_longer(max_speed, length, expected_steps_in_long_cell):
    _, unit_trajectory = _run_straight(None, max_speed)
    _, trajectory = _run_straight({_edge(2): length}, max_speed)
    assert _steps_in(trajectory, 2) == expected_steps_in_long_cell
    assert _steps_in(unit_trajectory, 2) == int(1 / max_speed)
    # other cells unaffected (up to a carried-over overshoot)
    assert _steps_in(trajectory, 1) == _steps_in(unit_trajectory, 1)


def test_long_edge_overshoot_carried_into_next_edge():
    # speed 1, length 5/2: 0 -> 1 -> 2 -> crossing with 1/2 leftover into cell (0, 3)
    _, trajectory = _run_straight({_edge(2): Fraction(5, 2)}, Fraction(1))
    in_long = [d for ep, d in trajectory if ep is not None and ep[0] == (0, 2)]
    assert in_long == [Fraction(0), Fraction(1), Fraction(2)]
    first_in_next = next(d for ep, d in trajectory if ep is not None and ep[0] == (0, 3))
    assert first_in_next == Fraction(1, 2)


def test_long_target_edge_parks_at_edge_length():
    # target cell (0, 4) entered via edge (0, 3) of length 3, not removed at target
    env, trajectory = _run_straight({_edge(3): Fraction(3)}, Fraction(1), remove_agents_at_target=False)
    assert env.agents[0].speed_counter.distance <= Fraction(3)
    assert _steps_in(trajectory, 3) == 3


def test_edge_lengths_default_and_validation():
    rail, optionals = _straight_rail(4)
    env = RailEnv(width=4, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    assert env.rail.edge_lengths is None
    assert not env.rail.has_edge_lengths()
    assert env.rail.edge_length(*_edge(1)) == DEFAULT_EDGE_LENGTH
    env.rail.edge_lengths = {_edge(1): 2.0}
    assert env.rail.has_edge_lengths()
    assert env.rail.edge_length(*_edge(1)) == Fraction(2)
    assert env.rail.edge_length(*_edge(2)) == DEFAULT_EDGE_LENGTH
    with pytest.raises(ValueError):
        env.rail.edge_lengths = {_edge(1): Fraction(1, 2)}


@pytest.mark.parametrize("non_edge", [
    _edge(0),  # dead end (0, 0) is entered heading west only
    (((0, 1), EAST), ((0, 3), EAST)),  # skips a cell
    (((0, 1), EAST), ((0, 1), NORTH)),  # no transition towards north
    (((0, 3), EAST), ((0, 4), EAST)),  # leaves the grid
])
def test_edge_lengths_reject_non_edges(non_edge):
    rail, optionals = _straight_rail(4)
    with pytest.raises(ValueError):
        rail.edge_lengths = {non_edge: Fraction(2)}
    env = RailEnv(width=4, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, {non_edge: Fraction(2)})),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    with pytest.raises(ValueError):
        env.reset()


@pytest.mark.parametrize("seed", [42, 43])
def test_grid_has_edge_matches_graph_edges(seed):
    env, _, _ = env_generator(seed=seed)
    g = GraphTransitionMap.grid_to_digraph(env.rail)
    for r in range(env.height):
        for c in range(env.width):
            for d in range(4):
                for d2 in range(4):
                    (r2, c2) = get_new_position((r, c), d2)
                    assert env.rail.has_edge(((r, c), d), ((r2, c2), d2)) == g.has_edge(f"{r, c, d}", f"{r2, c2, d2}")


@pytest.mark.parametrize("regenerate_rail", [True, False])
def test_edge_lengths_from_rail_generator_optionals_survive_reset(regenerate_rail):
    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, {_edge(1): 3})),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    assert env.rail.edge_lengths == {_edge(1): Fraction(3)}
    env.reset(regenerate_rail=regenerate_rail, regenerate_schedule=regenerate_rail)
    assert env.rail.edge_lengths == {_edge(1): Fraction(3)}
    assert env.distance_map.rail.has_edge_lengths()


@pytest.mark.parametrize("suffix", ["pkl", "mpk"])
def test_edge_lengths_persisted(suffix):
    from flatland.envs.rail_generators import rail_from_file

    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, {_edge(1): Fraction(3)})),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    with tempfile.TemporaryDirectory() as tmpdirname:
        f = Path(tmpdirname) / f"env.{suffix}"
        try:
            RailEnvPersister.save(env, str(f))
        except TypeError:
            pytest.skip("msgpack cannot persist Fractions")
        loaded, _ = RailEnvPersister.load_new(str(f))
        # rail generator from file passes the edge lengths on in its optionals
        _, from_file_optionals = rail_from_file(str(f))(5, 1, 1)
    assert loaded.rail.edge_lengths == {_edge(1): Fraction(3)}
    assert loaded.rail.edge_length(*_edge(1)) == Fraction(3)
    assert from_file_optionals["edge_lengths"] == {_edge(1): Fraction(3)}


def test_graph_edge_length_attribute():
    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, {_edge(1): Fraction(3)})),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    graph_env = GraphRailEnv.from_rail_env(env, DummyObservationBuilder())
    gctgc = GraphTransitionMap.grid_entry_point_to_graph_entry_point
    u, v = gctgc(0, 1, EAST), gctgc(0, 2, EAST)
    assert graph_env.rail.g.edges[u, v]["length"] == Fraction(3)
    assert graph_env.rail.edge_length(u, v) == Fraction(3)
    u2, v2 = gctgc(0, 2, EAST), gctgc(0, 3, EAST)
    assert "length" not in graph_env.rail.g.edges[u2, v2]
    assert graph_env.rail.edge_length(u2, v2) == DEFAULT_EDGE_LENGTH
    # read when constructed: later changes on g are not picked up
    graph_env.rail.g.edges[u2, v2]["length"] = Fraction(2)
    assert graph_env.rail.edge_length(u2, v2) == DEFAULT_EDGE_LENGTH
    # validated when constructed
    graph_env.rail.g.edges[u2, v2]["length"] = 0.5  # below MIN_EDGE_LENGTH (1)
    with pytest.raises(ValueError):
        GraphTransitionMap(graph_env.rail.g)  # constructor validates every `length` attribute


@pytest.mark.parametrize("seed", range(42, 46))
def test_grid_graph_parity_with_edge_lengths(seed):
    grid_env, _, _ = env_generator(seed=seed, rewards=DefaultRewards())
    np_random = np.random.RandomState(seed)
    lengths = [Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)]
    g = GraphTransitionMap.grid_to_digraph(grid_env.rail)
    edge_lengths = {}
    for u, v in g.edges:
        edge_lengths[(GraphTransitionMap.graph_entry_point_to_grid_entry_point(u),
                      GraphTransitionMap.graph_entry_point_to_grid_entry_point(v))] = lengths[np_random.randint(len(lengths))]
    grid_env.rail.edge_lengths = edge_lengths
    grid_env.distance_map.reset(grid_env.agents, grid_env.rail)
    graph_env: GraphRailEnv = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder(), seed=seed, rewards=DefaultRewards())

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        grid_trajectory = PolicyRunner.create_from_policy(env=grid_env, policy=RandomPolicy(), data_dir=data_dir / "one")
        graph_trajectory = PolicyRunner.create_from_policy(env=graph_env, policy=RandomPolicy(), data_dir=data_dir / "two", snapshot_interval=0,
                                                           no_save=True)
        assert len(grid_trajectory.compare_arrived(graph_trajectory)) == 0
        assert len(grid_trajectory.compare_actions(graph_trajectory)) == 0
        graph_trajectory.trains_positions["position"] = graph_trajectory.trains_positions["position"].map(
            GraphTransitionMap.graph_entry_point_to_grid_entry_point)
        assert len(graph_trajectory.trains_positions["position"].compare(grid_trajectory.trains_positions["position"])) == 0
        assert len(graph_trajectory.compare_rewards_dones_infos(grid_trajectory, ignoring_action_required=False)) == 0


def test_weighted_walk_prefers_shorter_length_over_fewer_edges():
    import networkx as nx
    from flatland.core.distance_map_walker import DistanceMapWalker
    from flatland.core.entry_point_distance_map import EntryPointDistanceMap

    g = nx.DiGraph()
    # a -> b -> d: 2 edges, total length 10; a -> c -> e -> d: 3 edges, total length 3
    g.add_edge("a", "b", length=5)
    g.add_edge("b", "d", length=5)
    g.add_edge("a", "c")
    g.add_edge("c", "e")
    g.add_edge("e", "d")
    gtm = GraphTransitionMap(g)
    unweighted_gtm = GraphTransitionMap(nx.DiGraph(list(g.edges)))
    assert gtm.has_edge_lengths() and not unweighted_gtm.has_edge_lengths()

    unweighted = EntryPointDistanceMap(agents=[], waypoint_init=str)
    unweighted.reset([], unweighted_gtm)
    assert DistanceMapWalker(unweighted)._distance_map_walker(unweighted_gtm, ["d"]) == {"a", "b", "c", "d", "e"}
    assert unweighted._get_distance("a", "d") == 2
    assert unweighted._get_distance("b", "d") == 1

    weighted = EntryPointDistanceMap(agents=[], waypoint_init=str)
    weighted.reset([], gtm)
    assert DistanceMapWalker(weighted)._distance_map_walker(gtm, ["d"]) == {"a", "b", "c", "d", "e"}
    assert weighted._get_distance("a", "d") == 3
    assert weighted._get_distance("b", "d") == 5
    assert weighted._get_distance("c", "d") == 2
    assert weighted._get_distance("d", "d") == 0


@pytest.mark.parametrize("seed", [42, 43])
def test_weighted_walk_with_unit_lengths_matches_bfs(seed):
    grid_env, _, _ = env_generator(seed=seed)
    bfs = np.copy(grid_env.distance_map.get())
    # explicit unit edge lengths (Fraction(1)) instead of the default integer unit lengths
    g = GraphTransitionMap.grid_to_digraph(grid_env.rail)
    gtgg = GraphTransitionMap.graph_entry_point_to_grid_entry_point
    grid_env.rail.edge_lengths = {(gtgg(u), gtgg(v)): DEFAULT_EDGE_LENGTH for u, v in g.edges}
    grid_env.distance_map.reset(grid_env.agents, grid_env.rail)
    weighted = grid_env.distance_map.get()
    assert np.array_equal(bfs, weighted)


def test_grid_distance_map_sums_edge_lengths():
    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1,
                  rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, {_edge(1): Fraction(3), _edge(2): Fraction(5, 2)})),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    agent = env.agents[0]
    agent.targets = {((0, 4), EAST)}
    env.distance_map.reset(env.agents, env.rail)
    assert env.distance_map.geometric_distance(((0, 3), EAST), 0) == 1
    assert env.distance_map.geometric_distance(((0, 2), EAST), 0) == 1 + Fraction(5, 2)
    assert env.distance_map.geometric_distance(((0, 1), EAST), 0) == 1 + Fraction(5, 2) + 3


@pytest.mark.parametrize("max_speed", [Fraction(1), Fraction(1, 2), Fraction(1, 3)])
@pytest.mark.parametrize("earliest_departure", [0, 3])
def test_eta_with_edge_lengths(max_speed, earliest_departure):
    """
    Same as test_flatland_envs_rail_env.py's eta consistency tests, on a corridor with non-unit edge lengths:
    the off-map eta prediction is re-confirmed on map every step and matches the actual arrival.
    """
    from tests.test_flatland_envs_rail_env import _make_single_track_corridor

    n_cells = 6
    rail, optionals = _make_single_track_corridor(n_cells)
    g = GraphTransitionMap.grid_to_digraph(rail)
    gtgg = GraphTransitionMap.graph_entry_point_to_grid_entry_point
    edge_lengths = {}
    for u, v in g.edges:
        (r, c), _ = gtgg(u)
        edge_lengths[(gtgg(u), gtgg(v))] = [Fraction(2), Fraction(3), Fraction(1), Fraction(4)][c % 4]
    env = RailEnv(width=n_cells, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, edge_lengths)),
                  line_generator=sparse_line_generator(speed_ratio_map={max_speed: 1.0}), number_of_agents=1,
                  remove_agents_at_target=True)
    env.acceleration_delta = Fraction(1)
    env.reset(random_seed=1)
    env._max_episode_steps = 1000
    agent = env.agents[0]
    agent.earliest_departure = earliest_departure

    eta = env.distance_map.eta(
        entry_point=agent.initial_entry_point, handle=agent.handle, elapsed_steps=env._elapsed_steps,
        earliest_departure=earliest_departure, speed=None, max_speed=max_speed, acceleration_delta=env.acceleration_delta,
    )
    # sanity check: longer than with unit edge lengths
    assert env.distance_map.geometric_distance(agent.initial_entry_point, agent.handle) > n_cells - 1

    for _ in range(1000):
        env.step({0: RailEnvActions.MOVE_FORWARD})
        if agent.current_entry_point is not None:
            on_map_eta = env.distance_map.eta(
                entry_point=agent.current_entry_point, handle=agent.handle, elapsed_steps=env._elapsed_steps,
                earliest_departure=earliest_departure, speed=agent.speed_counter.speed, max_speed=max_speed,
                acceleration_delta=env.acceleration_delta, distance=agent.speed_counter.distance,
            )
            assert on_map_eta == eta
        if env.dones["__all__"]:
            break
    assert agent.arrival_time == eta


def test_graph_distance_map_uses_length_attribute():
    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, {_edge(1): Fraction(3)})),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset()
    graph_env = GraphRailEnv.from_rail_env(env, DummyObservationBuilder())
    agent = graph_env.agents[0]
    gctgc = GraphTransitionMap.grid_entry_point_to_graph_entry_point
    grid_agent = env.agents[0]
    for source in [gctgc(0, 1, EAST), gctgc(0, 2, EAST), gctgc(0, 3, EAST)]:
        assert graph_env.distance_map.geometric_distance(source, agent.handle) == env.distance_map.geometric_distance(
            GraphTransitionMap.graph_entry_point_to_grid_entry_point(source), grid_agent.handle)


def test_decision_point_graph_edge_length():
    from flatland.envs.graph.graph_simplification import DecisionPointGraph
    from flatland.utils.simple_rail import make_simple_rail

    rail, _, _ = make_simple_rail()
    g = GraphTransitionMap.grid_to_digraph(rail)
    unit_dpg = DecisionPointGraph.fromGraphTransitionMap(GraphTransitionMap(g))
    for _, _, d in unit_dpg.g.edges(data="d"):
        assert d.length == d.len - 1
    for u, v in g.edges:
        g.edges[u, v]["length"] = Fraction(5, 2)
    dpg = DecisionPointGraph.fromGraphTransitionMap(GraphTransitionMap(g))
    for _, _, d in dpg.g.edges(data="d"):
        assert d.length == Fraction(5, 2) * (d.len - 1)


def test_weighted_distance_map_recompute():
    grid_env, _, _ = env_generator(seed=42)
    g = GraphTransitionMap.grid_to_digraph(grid_env.rail)
    gtgg = GraphTransitionMap.graph_entry_point_to_grid_entry_point
    grid_env.rail.edge_lengths = {(gtgg(u), gtgg(v)): Fraction(2) for u, v in g.edges}
    grid_env.distance_map.reset(grid_env.agents, grid_env.rail)
    expected = np.copy(grid_env.distance_map.get())
    # re-computing without reset (distances of the previous walk still present) gives the same distances
    grid_env.distance_map._compute(grid_env.agents, grid_env.rail)
    assert np.array_equal(grid_env.distance_map.get(), expected)
    assert np.isfinite(expected).any()


def _straight_env_with_agent(edge_lengths, max_speed=Fraction(1), obs_builder=None, n_cells=5):
    """ Single agent on map at the start of dead-end cell (0, 0), leaving east, target (0, n_cells - 1). """
    from flatland.core.env_observation_builder import DummyObservationBuilder as _Dummy
    rail, optionals = _straight_rail(n_cells)
    env = RailEnv(width=n_cells, height=1, rail_generator=rail_from_grid_transition_map(rail, _with_edge_lengths(optionals, edge_lengths)),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=obs_builder if obs_builder is not None else _Dummy())
    env.reset()
    env._max_episode_steps = 1000
    agent = env.agents[0]
    agent.initial_entry_point = ((0, 0), Grid4TransitionsEnum.WEST)
    agent.current_entry_point = ((0, 0), Grid4TransitionsEnum.WEST)
    agent.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, agent.current_entry_point))
    agent.targets = {((0, n_cells - 1), EAST)}
    agent._set_state(TrainState.MOVING)
    agent.speed_counter = SpeedCounter(max_speed=max_speed, speed=max_speed)
    agent.speed_counter.set(max_speed, Fraction(0), env.rail.edge_length(agent.current_entry_point, agent.next_entry_point))
    env.distance_map.reset(env.agents, env.rail)
    env.obs_builder.reset(env)
    return env


@pytest.mark.parametrize("edge_lengths,expected_dist", [
    (None, 4),
    ({_edge(1): Fraction(3), _edge(2): Fraction(5, 2)}, 1 + 3 + Fraction(5, 2) + 1),
])
def test_tree_obs_distances_sum_edge_lengths(edge_lengths, expected_dist):
    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv

    env = _straight_env_with_agent(edge_lengths, obs_builder=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    obs = env.obs_builder.get_many([0])[0]
    forward = obs.childs["F"]
    assert forward.dist_own_target_encountered == expected_dist
    assert forward.dist_to_next_branch == expected_dist
    assert obs.dist_min_to_target == expected_dist
    if edge_lengths is None:
        assert isinstance(forward.dist_own_target_encountered, int)


@pytest.mark.parametrize("max_speed,edge_lengths,expected_cols", [
    (Fraction(1), None, [0, 1, 2, 3, 4, 4, 4, 4]),
    (Fraction(1), {_edge(1): Fraction(3)}, [0, 1, 1, 1, 2, 3, 4, 4]),
    (Fraction(1, 2), {_edge(1): Fraction(2)}, [0, 0, 1, 1, 1, 1, 2, 2]),
    # overshoot carried over: 3/2 + 3/2 = 3 steps on the two long edges
    (Fraction(1), {_edge(1): Fraction(3, 2), _edge(2): Fraction(3, 2)}, [0, 1, 1, 2, 3, 4, 4, 4]),
])
def test_shortest_path_predictor_with_edge_lengths(max_speed, edge_lengths, expected_cols):
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv

    env = _straight_env_with_agent(edge_lengths, max_speed=max_speed)
    predictor = ShortestPathPredictorForRailEnv(max_depth=len(expected_cols) - 1)
    predictor.reset(env)
    prediction = predictor.get()[0]
    assert [int(p[2]) for p in prediction] == expected_cols


def test_render_agent_progress():
    from flatland.utils.rendertools import RenderTool, RenderLocal

    env = _straight_env_with_agent({_edge(1): Fraction(3)})
    agent = env.agents[0]
    env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point == ((0, 1), EAST)

    plain = RenderTool(env, gl="PILSVG")
    with_progress = RenderTool(env, gl="PILSVG", show_agent_progress=True)
    gl = with_progress.renderer.gl
    n_pix = gl.nPixCell
    color = np.array(gl.get_agent_color(0))
    bar_row = int(0.87 * n_pix)

    def filled(img, col):
        row = img[bar_row, col * n_pix:(col + 1) * n_pix, :3]
        return int(np.all(row == color, axis=1).sum())

    widths = []
    for expected_distance in [Fraction(0), Fraction(1), Fraction(2)]:
        assert agent.speed_counter.distance == expected_distance
        assert RenderLocal._agent_progress(agent) == pytest.approx(float(expected_distance / 3))
        img_plain = plain.render_env(show=False, show_observations=False, return_image=True)
        img = with_progress.render_env(show=False, show_observations=False, return_image=True)
        widths.append(filled(img, 1))
        # default rendering unaffected: no bar drawn
        assert filled(img_plain, 1) == 0
        env.step({0: RailEnvActions.MOVE_FORWARD})
    assert widths[0] == 0
    # bar is 0.8 of the cell width: 1/3 resp. 2/3 of it filled (up to pixel rounding and outline)
    assert widths[1] == pytest.approx(0.8 * n_pix / 3, abs=3)
    assert widths[2] == pytest.approx(2 * 0.8 * n_pix / 3, abs=3)


@pytest.mark.parametrize("max_speed", [Fraction(1), Fraction(1, 2)])
@pytest.mark.parametrize("edge_lengths,expected_length", [
    (None, 5),
    ({_edge(1): Fraction(3), _edge(2): Fraction(5, 2)}, 1 + 3 + Fraction(5, 2) + 1 + 1),
])
def test_travel_time_on_shortest_path_with_edge_lengths(max_speed, edge_lengths, expected_length):
    env = _straight_env_with_agent(edge_lengths, max_speed=max_speed)
    agent = env.agents[0]
    # sum of the edge lengths from (0, 0) to (0, 4), plus 1 as len(shortest_path) counts waypoints, not edges
    assert env.distance_map.shortest_path_length(agent) == expected_length
    assert agent.get_travel_time_on_shortest_path(env.distance_map) == math.ceil(expected_length / max_speed)


def _with_all_edge_lengths(rail_generator, length):
    """ Wraps `rail_generator` to return `length` for every edge of the generated rail in `optionals["edge_lengths"]`. """

    def generator(*args, **kwargs):
        rail, optionals = rail_generator(*args, **kwargs)
        gtgg = GraphTransitionMap.graph_entry_point_to_grid_entry_point
        edge_lengths = {(gtgg(u), gtgg(v)): length for u, v in GraphTransitionMap.grid_to_digraph(rail).edges}
        return rail, {**(optionals or {}), "edge_lengths": edge_lengths}

    return generator


def _env_with_intermediate_waypoints(length=None):
    env, _, _ = env_generator(seed=42, line_length=3, n_cities=5, x_dim=40, y_dim=40)
    if length is not None:
        env.rail_generator = _with_all_edge_lengths(env.rail_generator, length)
    env.reset(random_seed=42)
    assert all(len(a.waypoints) == 3 for a in env.agents)
    return env


def test_timetable_segments_scale_with_edge_lengths():
    unit_env = _env_with_intermediate_waypoints()
    env = _env_with_intermediate_waypoints(Fraction(2))
    assert env.rail.has_edge_lengths()
    for unit_agent, agent in zip(unit_env.agents, env.agents):
        # segment lengths are the sum of the edge lengths plus 1: n unit edges -> n + 1, n edges of length 2 -> 2n + 1
        unit_first = unit_agent.waypoints_earliest_departure[1] - unit_agent.waypoints_earliest_departure[0]
        first = agent.waypoints_earliest_departure[1] - agent.waypoints_earliest_departure[0]
        assert first == 2 * unit_first - 1
        unit_last = unit_agent.waypoints_latest_arrival[2] - unit_agent.waypoints_latest_arrival[1]
        last = agent.waypoints_latest_arrival[2] - agent.waypoints_latest_arrival[1]
        assert last == 2 * unit_last - 1


@pytest.mark.xfail(strict=True, reason="intermediate timetable entries are Fractions from shortest_path_length() - not yet whole steps")
@pytest.mark.parametrize("length", [None, Fraction(3, 2)])
def test_timetable_with_intermediate_waypoints_is_whole_steps(length):
    env = _env_with_intermediate_waypoints(length)
    for agent in env.agents:
        times = [t for t in agent.waypoints_earliest_departure + agent.waypoints_latest_arrival if t is not None]
        assert all(type(t) is int for t in times), times
