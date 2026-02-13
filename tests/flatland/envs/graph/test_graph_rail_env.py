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
from tests.trajectories.test_policy_runner import RandomPolicy


@pytest.mark.parametrize("seed", range(42, 58))
def test_graph_transition_map_from_with_random_policy(seed):
    # TODO restrictions:
    #   - no malfunction
    #   - mapping level-free/non-level free
    grid_env, _, _ = env_generator(seed=seed, malfunction_interval=9999999999999)
    graph_env: GraphRailEnv = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder())
    graph_env.reset()

    for r in range(grid_env.height):
        for c in range(grid_env.width):
            for d in range(4):
                assert (sum(grid_env.rail.get_transitions(((r, c), d))) > 0) == (f"{r, c, d}" in graph_env.rail.g.nodes)
                if sum(grid_env.rail.get_transitions(((r, c), d))) == 0:
                    continue
                for a in range(5):
                    # TODO typing
                    actual = graph_env.rail.check_action_on_agent(RailEnvActions.from_value(a), f"{r, c, d}")
                    expected_raw = grid_env.rail.check_action_on_agent(RailEnvActions.from_value(a), ((r, c), d))
                    new_cell_valid, ((r2, c2), d2), transition_valid, preprocessed_action = expected_raw

                    expected = new_cell_valid, f"{r2, c2, d2}", transition_valid, preprocessed_action

                    if "symmetric" not in RailEnvTransitionsEnum(grid_env.rail.get_full_transitions(r, c)).name:
                        assert (actual == expected)

                        # TODO new position is derived from grid, not possible on graph alone
                        # TODO maybe add invalid actions on node?
                        u = GraphTransitionMap.grid_configuration_to_graph_configuration(r, c, d)
                        v = GraphTransitionMap.grid_configuration_to_graph_configuration(r2, c2, d2)
                        assert expected_raw in graph_env.rail.g[u][v]["_grid_check_action_on_agent"]

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
