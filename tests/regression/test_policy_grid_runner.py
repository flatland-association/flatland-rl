import importlib
import tempfile
from pathlib import Path

import pandas as pd

from flatland.trajectories.policy_grid_runner import generate_trajectories_from_metadata
from flatland.trajectories.trajectories import TRAINS_ARRIVED_FNAME


def test_gen_trajectories_from_metadata():
    metadata_csv_path = importlib.resources.files("env_data.tests.service_test").joinpath("metadata.csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        with importlib.resources.as_file(metadata_csv_path) as metadata_csv:
            tmpdir = Path(tmpdirname)
            generate_trajectories_from_metadata(
                metadata_csv=metadata_csv,
                data_dir=tmpdir,
                policy_pkg="flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
                policy_cls="DeadLockAvoidancePolicy",
                obs_builder_pkg="flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
                obs_builder_cls="FullEnvObservation"
            )
            metadata = pd.read_csv(metadata_csv)
            for sr, t, (k, v) in zip([0.8571428571428571, 1.0, 0.8571428571428571, 1.0], [391, 163, 391, 163], metadata.iterrows()):
                df = pd.read_csv(tmpdir / v["test_id"] / v["env_id"] / TRAINS_ARRIVED_FNAME, sep="\t")
                assert df["success_rate"].to_list() == [sr]
                assert df["env_time"].to_list() == [t]
