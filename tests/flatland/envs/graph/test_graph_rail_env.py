import tempfile
from pathlib import Path

from flatland.env_generation.env_generator import env_generator
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy


def test_graph_transition_map_from_with_random_policy():
    # TODO restrictions:
    #   - no malfunction
    #   - homogeneous speed
    #   - L,R,F etc. depending on underlying grid
    env, _, _ = env_generator(malfunction_interval=9999999999999, speed_ratios={1.0: 1.0})
    clone = RailEnv(30, 30)
    clone.clone_from(env)
    clone.rail = GraphTransitionMap.from_rail_env(env)

    for r in range(env.height):
        for c in range(env.width):
            for d in range(4):
                assert (sum(env.rail.get_transitions(((r, c), d))) > 0) == ((r, c, d) in clone.rail.g.nodes)
                if sum(env.rail.get_transitions(((r, c), d))) == 0:
                    continue
                for a in range(5):
                    assert (clone.rail.check_action_on_agent(RailEnvActions.from_value(a), ((r, c), d)) ==
                            env.rail.check_action_on_agent(RailEnvActions.from_value(a), ((r, c), d)))

    # use Trajectory API for comparison
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(env=env, policy=RandomPolicy(), data_dir=data_dir / "one")
        other = PolicyRunner.create_from_policy(env=clone, policy=RandomPolicy(), data_dir=data_dir / "two", snapshot_interval=0, no_save=True)

        assert len(trajectory.compare_arrived(other)) == 0
        assert len(trajectory.compare_actions(other)) == 0
        print(trajectory.compare_positions(other))
        assert len(trajectory.compare_positions(other)) == 0
        assert len(trajectory.compare_rewards_dones_infos(other)) == 0
