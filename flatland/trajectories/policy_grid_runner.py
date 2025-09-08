from pathlib import Path

import click
import pandas as pd

from flatland.trajectories.policy_runner import generate_trajectory_from_policy


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
@click.option('--policy-pkg',
              type=str,
              help="Policy's fully qualified package name.",
              required=True
              )
@click.option('--policy-cls',
              type=str,
              help="Policy class name.",
              required=True
              )
@click.option('--obs-builder-pkg',
              type=str,
              help="Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None
              )
@click.option('--obs-builder-cls',
              type=str,
              help="Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None
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
def generate_trajectories_from_metadata(
    metadata_csv: Path,
    data_dir: Path,
    policy_pkg: str, policy_cls: str,
    obs_builder_pkg: str, obs_builder_cls: str,
    rewards_pkg: str = "flatland.envs.rewards",
    rewards_cls: str = "DefaultRewards",
):
    metadata = pd.read_csv(metadata_csv)
    for k, v in metadata.iterrows():
        try:
            test_folder = data_dir / v["test_id"] / v["env_id"]
            test_folder.mkdir(parents=True, exist_ok=True)
            args = ["--data-dir", test_folder,
                    "--policy-pkg", policy_pkg, "--policy-cls", policy_cls,
                    "--obs-builder-pkg", obs_builder_pkg, "--obs-builder-cls", obs_builder_cls,
                    "--n_agents", v["n_agents"],
                    "--x_dim", v["x_dim"],
                    "--y_dim", v["y_dim"],
                    "--n_cities", v["n_cities"],
                    "--max_rail_pairs_in_city", v["max_rail_pairs_in_city"],
                    "--grid_mode", v["grid_mode"],
                    "--max_rails_between_cities", v["max_rails_between_cities"],
                    "--malfunction_duration_min", v["malfunction_duration_min"],
                    "--malfunction_duration_max", v["malfunction_duration_max"],
                    "--malfunction_interval", v["malfunction_interval"],
                    "--speed_ratios", "1.0", "0.25",
                    "--speed_ratios", "0.5", "0.25",
                    "--speed_ratios", "0.33", "0.25",
                    "--speed_ratios", "0.25", "0.25",
                    "--seed", v["seed"],
                    "--snapshot-interval", 0,
                    "--ep-id", v["test_id"] + "_" + v["env_id"]
                    ]

            if rewards_pkg is not None and rewards_cls is not None:
                args += ["--rewards-pkg", rewards_pkg, "--rewards-cls", rewards_cls, ]
            generate_trajectory_from_policy(args)

        except SystemExit as exc:
            assert exc.code == 0


if __name__ == '__main__':
    metadata_csv = Path("./episodes/trajectories/malfunction_deadlock_avoidance_heuristics/metadata.csv").resolve()
    data_dir = Path("./episodes/trajectories/malfunction_deadlock_avoidance_heuristics").resolve()
    generate_trajectories_from_metadata([
        "--metadata-csv", metadata_csv,
        "--data-dir", data_dir,
        "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
        "--policy-cls", "DeadLockAvoidancePolicy",
        "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
        "--obs-builder-cls", "FullEnvObservation"
    ])
