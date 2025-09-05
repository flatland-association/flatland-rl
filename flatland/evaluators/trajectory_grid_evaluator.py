import importlib
from pathlib import Path

import click
import pandas as pd

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
def evaluate_trajectories_from_metadata(
        metadata_csv: Path,
        data_dir: Path,
        rewards_pkg: str = "flatland.envs.rewards",
        rewards_cls: str = "DefaultRewards"
):
    metadata = pd.read_csv(metadata_csv)
    for k, v in metadata.iterrows():
        try:
            rewards = None
            if rewards_pkg is not None and rewards_cls is not None:
                module = importlib.import_module(rewards_pkg)
                rewards = getattr(module, rewards_cls)
                rewards = rewards()
            test_folder = data_dir / v["test_id"] / v["env_id"]
            TrajectoryEvaluator(Trajectory(data_dir=test_folder, ep_id=v["test_id"] + "_" + v["env_id"])).evaluate(rewards=rewards)
        except SystemExit as exc:
            assert exc.code == 0
