import tempfile
from pathlib import Path

import numpy as np

from flatland.callbacks.generate_movie_callbacks import GenerateMovieCallbacks
from flatland.env_generation.env_generator import env_generator
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.trajectories.policy_runner import PolicyRunner


def test_shortest_path_policy_no_intermediate_target():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            data_dir=data_dir,
            snapshot_interval=5,
            env=env_generator(obs_builder_object=FullEnvObservation(), seed=42, )[0],
        )
        assert np.isclose(trajectory.trains_arrived_lookup()["success_rate"], 1 / 7)


def test_shortest_path_policy_with_intermediate_targets(gen_movies=False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            data_dir=data_dir,
            snapshot_interval=5,
            env=env_generator(n_cities=5, line_length=3, obs_builder_object=FullEnvObservation(), seed=42, )[0],
            callbacks=GenerateMovieCallbacks() if gen_movies else None,
        )
        assert np.isclose(trajectory.trains_arrived_lookup()["success_rate"], 1 / 7)
