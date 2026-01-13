from pathlib import Path

import click
import numpy as np
import pandas as pd

from flatland.envs.rewards import DefaultRewards
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.trajectories import Trajectory
from flatland.utils.cli_utils import resolve_type


@click.command()
@click.option('--metadata-csv',
              type=click.Path(exists=True, path_type=Path, dir_okay=False),
              help="Path to metadata.csv",
              required=True,
              )
@click.option('--data-dir',
              type=click.Path(exists=True, path_type=Path, file_okay=False),
              help="Path to folder containing Flatland episode",
              required=True,
              )
@click.option('--rewards',
              type=str,
              help="Defaults to `flatland.envs.rewards.DefaultRewards`",
              required=False,
              default=None,
              )
@click.option('--rewards-pkg',
              type=str,
              help="DEPRECATED: use --rewards instead. Defaults to `flatland.envs.rewards.DefaultRewards`",
              required=False,
              default="flatland.envs.rewards",
              )
@click.option('--rewards-cls',
              type=str,
              help="DEPRECATED: use --rewards instead. Defaults to `flatland.envs.rewards.DefaultRewards`",
              required=False,
              default="DefaultRewards",
              )
@click.option('--aggregator',
              type=str,
              help="Function to aggregate scenario rewards. Defaults to `numpy.sum`",
              required=False,
              default=None,
              )
@click.option('--aggregator-pkg',
              type=str,
              help="DEPRECATED: use --aggregator instead. Function to aggregate scenario rewards. Defaults to `numpy.sum`",
              required=False,
              default=None,
              )
@click.option('--aggregator-cls',
              type=str,
              help="DEPRECATED: use --aggregator instead. Function to aggregate scenario rewards. Defaults to `numpy.sum`",
              required=False,
              default=None,
              )
@click.option('--report',
              type=str,
              help="Function to report aggregated scenario rewards. Defaults to `None`",
              required=False,
              default=None,
              )
@click.option('--report-pkg',
              type=str,
              help="DEPRECATED: use --report instead. Function to report aggregated scenario rewards. Defaults to `None`",
              required=False,
              default=None
              )
@click.option('--report-cls',
              type=str,
              help="DEPRECATED: use --report instead. Function to report aggregated scenario scores. Defaults to `None`",
              required=False,
              default=None,
              )
def evaluate_trajectories_from_metadata(
    metadata_csv: Path,
    data_dir: Path,
    rewards: str = None,
    rewards_pkg: str = None,
    rewards_cls: str = None,
    aggregator: str = None,
    aggregator_pkg: str = None,
    aggregator_cls: str = None,
    report: str = None,
    report_pkg: str = None,
    report_cls: str = None,
):
    metadata = pd.read_csv(metadata_csv)

    scenario_rewards = []
    for k, v in metadata.iterrows():
        try:
            rewards_ = resolve_type(rewards, rewards_pkg, rewards_cls)
            rewards_ = rewards_ or DefaultRewards
            rewards_ = rewards_()
            test_folder = data_dir / v["test_id"] / v["env_id"]
            trajectory = Trajectory.load_existing(data_dir=test_folder, ep_id=v["test_id"] + "_" + v["env_id"])
            TrajectoryEvaluator(trajectory).evaluate(rewards=rewards_)
            scenario_rewards.append(rewards_.cumulate(*trajectory.trains_rewards_dones_infos['reward'].to_list()))
        except SystemExit as exc:
            assert exc.code == 0

    aggregator = resolve_type(aggregator, aggregator_pkg, aggregator_cls) or np.sum
    aggregated_scenario_scores = aggregator(scenario_rewards)

    reporter = resolve_type(report, report_pkg, report_cls)
    if reporter is not None:
        reporter(aggregated_scenario_scores)
