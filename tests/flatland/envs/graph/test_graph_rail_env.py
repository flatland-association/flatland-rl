import tempfile
from pathlib import Path

import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.env_generation.env_generator import env_generator
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.rail_env_action import RailEnvActions
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.utils.seeding import random_state_to_hashablestate
from tests.trajectories.test_policy_runner import RandomPolicy


@pytest.mark.parametrize("seed", range(42, 58))
def test_graph_transition_map_from_with_random_policy(seed):
    grid_env, _, _ = env_generator(seed=seed)
    graph_env: GraphRailEnv = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder(), seed=seed)
    assert random_state_to_hashablestate(grid_env.np_random) == random_state_to_hashablestate(graph_env.np_random)

    for r in range(grid_env.height):
        for c in range(grid_env.width):
            for d in range(4):
                assert (sum(grid_env.rail.get_transitions(((r, c), d))) > 0) == (f"{r, c, d}" in graph_env.rail.g.nodes)
                u = GraphTransitionMap.grid_configuration_to_graph_configuration(r, c, d)
                is_grid_configuration = sum(grid_env.rail.get_transitions(((r, c), d))) > 0
                is_graph_configuration = u in graph_env.rail.g.nodes
                assert is_graph_configuration == is_grid_configuration
                if not is_grid_configuration:
                    continue

                if "symmetric" in RailEnvTransitionsEnum(grid_env.rail.get_full_transitions(r, c)).name and sum(
                    grid_env.rail.get_transitions(((r, c), d))) == 2:
                    assert RailEnvActions.MOVE_FORWARD in graph_env.rail.g.nodes[u]["prohibited_actions"]
                    assert RailEnvActions.DO_NOTHING in graph_env.rail.g.nodes[u]["prohibited_actions"]
                    # TODO revise design: no braking on symmetric switches?
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
                    # TODO typing
                    actual = graph_env.rail.apply_action_independent(RailEnvActions.from_value(a), f"{r, c, d}")
                    expected_raw = grid_env.rail.apply_action_independent(RailEnvActions.from_value(a), ((r, c), d))
                    if expected_raw is None:
                        assert actual == expected_raw
                    else:
                        ((r2, c2), d2), transition_valid = expected_raw
                        expected = f"{r2, c2, d2}", transition_valid
                        assert actual == expected

    # use Trajectory API for comparison
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        grid_trajectory = PolicyRunner.create_from_policy(env=grid_env, policy=RandomPolicy(), data_dir=data_dir / "one")
        graph_trajectory = PolicyRunner.create_from_policy(env=graph_env, policy=RandomPolicy(), data_dir=data_dir / "two", snapshot_interval=0, no_save=True)

        assert len(grid_trajectory.compare_arrived(graph_trajectory)) == 0
        assert len(grid_trajectory.compare_actions(graph_trajectory)) == 0
        graph_trajectory.trains_positions["position"] = graph_trajectory.trains_positions["position"].map(
            GraphTransitionMap.graph_configuration_to_grid_configuration)
        assert len(graph_trajectory.trains_positions["position"].compare(grid_trajectory.trains_positions["position"])) == 0

        assert len(graph_trajectory.compare_rewards_dones_infos(grid_trajectory)) == 0
