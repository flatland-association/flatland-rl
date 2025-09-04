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
def evaluate_trajectories_from_metadata(
    metadata_csv: Path,
    data_dir: Path,
):
    metadata = pd.read_csv(metadata_csv)
    for k, v in metadata.iterrows():
        try:
            test_folder = data_dir / v["test_id"] / v["env_id"]
            TrajectoryEvaluator(Trajectory(data_dir=test_folder, ep_id=v["test_id"] + "_" + v["env_id"])).evaluate()
        except SystemExit as exc:
            assert exc.code == 0
