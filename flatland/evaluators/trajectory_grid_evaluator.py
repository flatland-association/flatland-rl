import importlib
from pathlib import Path

import click
import numpy as np
import pandas as pd

from flatland.envs.rewards import Rewards
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.trajectories import Trajectory


@click.command()
@click.option('--metadata-csv',
              type=click.Path(exists=True, path_type=Path, dir_okay=False),
              help="Path to metadata.csv",
              required=True
              )
@click.option('--data-dir',
              type=click.Path(exists=True, path_type=Path, file_okay=False),
              help="Path to folder containing Flatland episode",
              required=True
              )
@click.option('--rewards-pkg',
              type=str,
              help="Defaults to `flatland.envs.rewards.DefaultRewards`",
              required=False,
              default="flatland.envs.rewards"
              )
@click.option('--rewards-cls',
              type=str,
              help="Defaults to `flatland.envs.rewards.DefaultRewards`",
              required=False,
              default="DefaultRewards"
              )
@click.option('--aggregator-pkg',
              type=str,
              help="Function to aggregate scenario rewards. Defaults to `numpy.sum`",
              required=False,
              default="numpy"
              )
@click.option('--aggregator-cls',
              type=str,
              help="Function to aggregate scenario rewards. Defaults to `numpy.sum`",
              required=False,
              default="sum"
              )
@click.option('--report-pkg',
              type=str,
              help="Function to report aggregated scenario rewards. Defaults to `None`",
              required=False,
              default=None
              )
@click.option('--report-cls',
              type=str,
              help="Function to report aggregated scenario scores. Defaults to `None`",
              required=False,
              default=None
              )
def evaluate_trajectories_from_metadata(
        metadata_csv: Path,
        data_dir: Path,
        rewards_pkg: str = "flatland.envs.rewards",
        rewards_cls: str = "DefaultRewards",
        aggregator_pkg: str = "numpy",
        aggregator_cls: str = "sum",
        report_pkg: str = None,
        report_cls: str = None,
):
    metadata = pd.read_csv(metadata_csv)

    agg = np.sum
    if aggregator_pkg is not None and aggregator_cls is not None:
        module = importlib.import_module(aggregator_pkg)
        agg = getattr(module, aggregator_cls)

    scenario_rewards = []
    for k, v in metadata.iterrows():
        try:
            rewards = None
            if rewards_pkg is not None and rewards_cls is not None:
                module = importlib.import_module(rewards_pkg)
                rewards = getattr(module, rewards_cls)
                rewards: Rewards = rewards()
            test_folder = data_dir / v["test_id"] / v["env_id"]
            trajectory = Trajectory(data_dir=test_folder, ep_id=v["test_id"] + "_" + v["env_id"])
            TrajectoryEvaluator(trajectory).evaluate(rewards=rewards)
            scenario_rewards.append(rewards.cumulate(*trajectory.trains_rewards_dones_infos['reward'].to_list()))
        except SystemExit as exc:
            assert exc.code == 0
    aggregated_scenario_scores = agg(scenario_rewards)
    if report_pkg is not None and report_cls is not None:
        module = importlib.import_module(report_pkg)
        reporter = getattr(module, report_cls)
        reporter(aggregated_scenario_scores)
