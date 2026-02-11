import tempfile
from pathlib import Path

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.env_generation.env_generator import env_generator
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy


def test_graph_transition_map_from_with_random_policy():
    # TODO restrictions:
    #   - no malfunction
    #   - test multi-speed and dynamic speed
    #   - mapping level-free/non-level free
    grid_env, _, _ = env_generator(seed=42, malfunction_interval=9999999999999, speed_ratios={1.0: 1.0})
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
                    expected = grid_env.rail.check_action_on_agent(RailEnvActions.from_value(a), ((r, c), d))
                    new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action = expected

                    expected = new_cell_valid, f"{new_position[0], new_position[1], new_direction}", transition_valid, preprocessed_action

                    assert (actual == expected)

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
