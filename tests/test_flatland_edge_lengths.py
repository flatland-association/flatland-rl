import pickle
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
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
from flatland.envs.step_utils.speed_counter import SpeedCounter, DEFAULT_EDGE_LENGTH, cached_cell_exit, validated_edge_length
from flatland.envs.step_utils.states import TrainState
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy

EAST = Grid4TransitionsEnum.EAST


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


def _run_straight(edge_lengths, max_speed, n_cells=5, remove_agents_at_target=True):
    """
    Single agent bootstrapped moving at full `max_speed` at the start of dead-end cell (0, 0) (entered heading west,
    hence leaving east), target (0, n_cells - 1),
    always MOVE_FORWARD. Returns the per-step list of (current_entry_point, distance) after each step,
    until the agent is done.
    """
    rail, optionals = _straight_rail(n_cells)
    env = RailEnv(width=n_cells, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1, remove_agents_at_target=remove_agents_at_target,
                  edge_lengths=edge_lengths)
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
    agent.speed_counter.set(max_speed, Fraction(0), env.edge_length(agent.current_entry_point, agent.next_entry_point))

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
    rail, optionals = _straight_rail(3)
    env = RailEnv(width=3, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    assert env.edge_lengths is None
    assert env.edge_length(*_edge(0)) == DEFAULT_EDGE_LENGTH
    env.edge_lengths = {_edge(0): 2.0}
    assert env.edge_length(*_edge(0)) == Fraction(2)
    assert env.edge_length(*_edge(1)) == DEFAULT_EDGE_LENGTH
    with pytest.raises(ValueError):
        env.edge_lengths = {_edge(0): Fraction(1, 2)}


@pytest.mark.parametrize("suffix", ["pkl", "mpk"])
def test_edge_lengths_persisted(suffix):
    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1, edge_lengths={_edge(1): Fraction(3)})
    env.reset()
    with tempfile.TemporaryDirectory() as tmpdirname:
        f = Path(tmpdirname) / f"env.{suffix}"
        try:
            RailEnvPersister.save(env, str(f))
        except TypeError:
            pytest.skip("msgpack cannot persist Fractions")
        loaded, _ = RailEnvPersister.load_new(str(f))
    assert loaded.edge_lengths == {_edge(1): Fraction(3)}
    assert loaded.edge_length(*_edge(1)) == Fraction(3)


def test_graph_edge_length_attribute():
    rail, optionals = _straight_rail(5)
    env = RailEnv(width=5, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1, edge_lengths={_edge(1): Fraction(3)})
    env.reset()
    graph_env = GraphRailEnv.from_rail_env(env, DummyObservationBuilder())
    gctgc = GraphTransitionMap.grid_entry_point_to_graph_entry_point
    u, v = gctgc(0, 1, EAST), gctgc(0, 2, EAST)
    assert graph_env.rail.g.edges[u, v]["length"] == Fraction(3)
    assert graph_env.edge_length(u, v) == Fraction(3)
    u2, v2 = gctgc(0, 2, EAST), gctgc(0, 3, EAST)
    assert "length" not in graph_env.rail.g.edges[u2, v2]
    assert graph_env.edge_length(u2, v2) == DEFAULT_EDGE_LENGTH
    graph_env.rail.g.edges[u2, v2]["length"] = 0.5
    with pytest.raises(ValueError):
        graph_env.edge_length(u2, v2)


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
    grid_env.edge_lengths = edge_lengths
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
