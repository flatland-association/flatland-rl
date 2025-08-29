import tempfile
from pathlib import Path

from flatland.env_generation.env_generator import env_generator
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.trajectories.policy_runner import PolicyRunner


def test_shortest_path_policy_no_intermediate_target():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            obs_builder=FullEnvObservation(),
            data_dir=data_dir,
            snapshot_interval=5,
        )
        assert trajectory.trains_arrived_lookup()["success_rate"] == 0.14285714285714285


def test_shortest_path_policy_with_intermediate_targets():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            obs_builder=FullEnvObservation(),
            data_dir=data_dir,
            snapshot_interval=5,
            n_cities=5,
            line_length=3,
        )
        assert trajectory.trains_arrived_lookup()["success_rate"] == 1.0
        env, _, _ = env_generator(
            n_cities=5,
            line_length=3
        )
        for agent in env.agents:
            positions = set(trajectory.trains_positions[trajectory.trains_positions["agent_id"] == agent.handle]["position"].tolist())
            assert (agent.initial_position, agent.direction) in positions
            # N.B. the agent is immediately removed when reaching target (multiple agents can reach the same target in the same time step!)
            for wp in agent.waypoints[:1]:
                assert (wp.position, wp.direction) in positions
