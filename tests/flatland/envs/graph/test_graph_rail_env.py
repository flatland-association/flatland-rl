import tempfile
from pathlib import Path

import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.env_generation.env_generator import env_generator
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rewards import BaseDefaultRewards, DefaultRewards, PunctualityRewards
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.utils.seeding import random_state_to_hashablestate
from flatland.utils.simple_rail import make_simple_rail
from tests.trajectories.test_policy_runner import RandomPolicy


@pytest.mark.parametrize("rewards_cls", [DefaultRewards, BaseDefaultRewards, PunctualityRewards])
@pytest.mark.parametrize("malfunction_interval", [540, 20])
@pytest.mark.parametrize("seed", range(42, 58))
@pytest.mark.skip
def test_graph_transition_map_from_with_random_policy(seed, malfunction_interval, rewards_cls):
    # N.B. a fresh instance per test invocation - Rewards accumulates mutable state
    # (arrivals/departures/states) over an episode, so instances must never be shared/reused
    # across parametrize cases or between the grid and graph env (see GraphRailEnv.from_rail_env).
    grid_env, _, _ = env_generator(seed=seed, malfunction_interval=malfunction_interval, rewards=rewards_cls())
    graph_env: GraphRailEnv = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder(), seed=seed, rewards=rewards_cls())
    assert random_state_to_hashablestate(grid_env.np_random) == random_state_to_hashablestate(graph_env.np_random)

    for r in range(grid_env.height):
        for c in range(grid_env.width):
            for d in range(4):
                assert (sum(grid_env.rail.get_transitions(((r, c), d))) > 0) == (f"{r, c, d}" in graph_env.rail.g.nodes)
                u = GraphTransitionMap.grid_entry_point_to_graph_entry_point(r, c, d)
                is_grid_entry_point = sum(grid_env.rail.get_transitions(((r, c), d))) > 0
                is_graph_entry_point = u in graph_env.rail.g.nodes
                assert is_graph_entry_point == is_grid_entry_point
                if not is_grid_entry_point:
                    continue

                if "symmetric" in RailEnvTransitionsEnum(grid_env.rail.get_full_transitions(r, c)).name and sum(
                    grid_env.rail.get_transitions(((r, c), d))) == 2:
                    assert RailEnvActions.MOVE_FORWARD in graph_env.rail.g.nodes[u]["prohibited_actions"]
                    assert RailEnvActions.DO_NOTHING in graph_env.rail.g.nodes[u]["prohibited_actions"]
                    # TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: no braking on symmetric switches?
                    assert RailEnvActions.STOP_MOVING in graph_env.rail.g.nodes[u]["prohibited_actions"]
                else:
                    assert RailEnvActions.MOVE_FORWARD not in graph_env.rail.g.nodes[u]["prohibited_actions"]
                    assert RailEnvActions.DO_NOTHING not in graph_env.rail.g.nodes[u]["prohibited_actions"]
                    assert RailEnvActions.STOP_MOVING not in graph_env.rail.g.nodes[u]["prohibited_actions"]

                # verify prohibited actions and edge actions are pairwise disjoint and cover all 5 Flatland actions
                actions = list(graph_env.rail.g.nodes[u]["prohibited_actions"])
                for v in list(graph_env.rail.g.successors(u)):
                    actions.extend(graph_env.rail.g.get_edge_data(u, v)["actions"])
                assert len(actions) == 5

                for a in range(5):
                    actual = graph_env.rail.apply_action_independent(RailEnvActions.from_value(a), f"{r, c, d}")
                    expected_raw = grid_env.rail.apply_action_independent(RailEnvActions.from_value(a), ((r, c), d))
                    if expected_raw is None:
                        assert actual == expected_raw
                    else:
                        (r2, c2), d2 = expected_raw
                        expected = f"{r2, c2, d2}"
                        assert actual == expected

    # use Trajectory API for comparison
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        grid_trajectory = PolicyRunner.create_from_policy(env=grid_env, policy=RandomPolicy(), data_dir=data_dir / "one")
        graph_trajectory = PolicyRunner.create_from_policy(env=graph_env, policy=RandomPolicy(), data_dir=data_dir / "two", snapshot_interval=0, no_save=True)

        def _any_malfunction(trajectory):
            return trajectory.trains_rewards_dones_infos["info"].map(lambda info: info["malfunction"] > 0).any()

        if malfunction_interval <= 20:
            # only guaranteed frequent enough to reliably hit at least once across all seeds at this rate -
            # the default (rarer) malfunction_interval may legitimately produce zero malfunctions for some seeds.
            # N.B. exact malfunction/info equality between grid and graph is already covered below by
            # compare_rewards_dones_infos (which diffs the full info dict, incl. malfunction) - this only
            # additionally guarantees malfunctions weren't trivially absent on both sides.
            assert _any_malfunction(grid_trajectory), "expected at least one malfunction in the grid env's run"
            assert _any_malfunction(graph_trajectory), "expected at least one malfunction in the graph env's run"

        assert len(grid_trajectory.compare_arrived(graph_trajectory)) == 0
        assert len(grid_trajectory.compare_actions(graph_trajectory)) == 0
        graph_trajectory.trains_positions["position"] = graph_trajectory.trains_positions["position"].map(
            GraphTransitionMap.graph_entry_point_to_grid_entry_point)
        assert len(graph_trajectory.trains_positions["position"].compare(grid_trajectory.trains_positions["position"])) == 0

        assert len(graph_trajectory.compare_rewards_dones_infos(grid_trajectory, ignoring_action_required=False)) == 0


@pytest.mark.parametrize("seed", range(42, 58))
def test_apply_timetable_to_agents_waypoints_well_formed(seed):
    """Regression test: `GraphRailEnv._apply_timetable_to_agents` must produce `agent.waypoints` as a
    well-formed `List[List[EntryPoint]]` - every entry, including the exploded target directions,
    must itself be a list of entry points, and only entry points that actually exist in the graph
    (e.g. not a direction blocked by a dead end) may be included."""
    grid_env, _, _ = env_generator(seed=seed)
    graph_env: GraphRailEnv = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder(), seed=seed)

    for agent in graph_env.agents:
        assert all(isinstance(wps, list) for wps in agent.waypoints)
        for wps in agent.waypoints:
            for entry_point in wps:
                assert entry_point in graph_env.rail.g.nodes
                assert isinstance(entry_point, str)
        assert set(agent.waypoints[-1]) == agent.targets


def test_from_graph_defaults():
    """
    Regression test: `GraphRailEnv.from_graph`'s default `timetable_generator`/`agent_speeds`
    branches - never exercised via `from_rail_env`, which always supplies both explicitly - must
    produce a working env: `ttgen_flatland2`'s fixed departure/arrival window and a uniform speed
    of `1.0` for every agent.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    env.reset(False, False)

    g = GraphTransitionMap.grid_to_digraph(env.rail)
    gctgc = GraphTransitionMap.grid_entry_point_to_graph_entry_point
    agent_waypoints = {
        agent.handle: [[gctgc(*wp.position, wp.direction) for wp in group] for group in agent.waypoints]
        for agent in env.agents
    }
    resource_map = {n: n for n in g.nodes}

    # N.B. agent_speeds and timetable_generator deliberately omitted to exercise from_graph's defaults.
    graph_env: GraphRailEnv = GraphRailEnv.from_graph(
        g=g,
        resource_map=resource_map,
        agent_waypoints=agent_waypoints,
        observation_builder=DummyObservationBuilder(),
    )

    assert graph_env._max_episode_steps == 1000
    for agent in graph_env.agents:
        assert agent.speed_counter.max_speed == 1.0
        assert agent.earliest_departure == 0
        assert agent.latest_arrival == 1000

    # smoke test: the resulting env must actually be steppable.
    for _ in range(5):
        graph_env.step({i: RailEnvActions.MOVE_FORWARD for i in range(graph_env.get_num_agents())})
