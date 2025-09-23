import importlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flatland.evaluators.trajectory_grid_evaluator import evaluate_trajectories_from_metadata
from flatland.trajectories.policy_grid_runner import generate_trajectories_from_metadata
from flatland.trajectories.trajectories import TRAINS_ARRIVED_FNAME


def _dummy_reporter(aggregated_scores):
    report = f"Aggregated scores: {aggregated_scores}"
    print(report)


def _dummy_aggregator(scores):
    print(f"Raw scores: {scores}")
    data = np.array(scores).transpose()
    print(f"Scenario raw punctuality: {data}")
    scenario_punctuality = data[0] / data[1]
    print(f"Scenario punctuality: {scenario_punctuality}")
    return np.mean(scenario_punctuality)


def test_gen_trajectories_from_metadata(capsys):
    metadata_csv_path = importlib.resources.files("env_data.tests.service_test").joinpath("metadata.csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        with importlib.resources.as_file(metadata_csv_path) as metadata_csv:
            tmpdir = Path(tmpdirname)
            with pytest.raises(SystemExit) as e_info:
                generate_trajectories_from_metadata([
                    "--metadata-csv", metadata_csv_path,
                    "--data-dir", tmpdir,
                    "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
                    "--policy-cls", "DeadLockAvoidancePolicy",
                    "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
                    "--obs-builder-cls", "FullEnvObservation",
                    "--rewards-pkg", "flatland.envs.rewards",
                    "--rewards-cls", "PunctualityRewards",
                ])
            assert e_info.value.code == 0
            metadata = pd.read_csv(metadata_csv)
            for sr, t, (k, v) in zip([0.8571428571428571, 1.0, 0.8571428571428571, 1.0], [391, 163, 391, 163], metadata.iterrows()):
                df = pd.read_csv(tmpdir / v["test_id"] / v["env_id"] / TRAINS_ARRIVED_FNAME, sep="\t")
                assert df["success_rate"].to_list() == [sr]
                assert df["env_time"].to_list() == [t]

            with pytest.raises(SystemExit) as e_info:
                evaluate_trajectories_from_metadata([
                    "--metadata-csv", metadata_csv_path,
                    "--data-dir", tmpdir,
                    "--rewards-pkg", "flatland.envs.rewards",
                    "--rewards-cls", "PunctualityRewards",
                    "--report-pkg", "tests.regression.test_policy_grid_runner_evaluator",
                    "--report-cls", "_dummy_reporter",
                    "--aggregator-pkg", "tests.regression.test_policy_grid_runner_evaluator",
                    "--aggregator-cls", "_dummy_aggregator",
                ])
            assert e_info.value.code == 0
            captured = capsys.readouterr()
            assert "Aggregated scores: 0.9285714285714286" in captured.out
            assert "Raw scores: [(12, 14), (14, 14), (12, 14), (14, 14)]" in captured.out
