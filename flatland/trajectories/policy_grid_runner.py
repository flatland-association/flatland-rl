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
@click.option('--policy',
              type=str,
              help=" Policy's fully qualified name. Can also be provided through env var POLICY (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--policy-pkg',
              type=str,
              help="DEPRECATED: use --policy instead. Policy's fully qualified package name. Can also be provided through env var POLICY_PKG (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--policy-cls',
              type=str,
              help="DEPRECATED: use --policy instead. Policy class name. Can also be provided through env var POLICY_CLS  (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--obs-builder',
              type=str,
              help="Can also be provided through env var OBS_BUILDER (command-line option takes priority). Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None,
              )
@click.option('--obs-builder-pkg',
              type=str,
              help="DEPRECATED: use --obs-builder instead. Can also be provided through env var OBS_BUILDER_PKG. Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None,
              )
@click.option('--obs-builder-cls',
              type=str,
              help="DEPRECATED: use --obs-builder instead. Can also be provided through env var OBS_BUILDER_CLS. Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None,
              )
@click.option('--rewards',
              type=str,
              help="Defaults to `flatland.envs.rewards.DefaultRewards`. Can also be provided through env var REWARDS (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--rewards-pkg',
              type=str,
              help="DEPRECATED: use --rewards instead. Defaults to `flatland.envs.rewards.DefaultRewards`. Can also be provided through env var REWARDS_PKG (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--rewards-cls',
              type=str,
              help="DEPRECATED: use --rewards instead. Defaults to `flatland.envs.rewards.DefaultRewards. Can also be provided through env var REWARDS_CLS (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--callbacks',
              type=str,
              help="Defaults to `flatland.envs.callbacks.Defaultcallbacks`. Can also be provided through env var callbacks (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--callbacks-pkg',
              type=str,
              help="DEPRECATED: use --callbacks instead. Defaults to `flatland.envs.callbacks.Defaultcallbacks`. Can also be provided through env var callbacks_PKG (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--callbacks-cls',
              type=str,
              help="DEPRECATED: use --callbacks instead. Defaults to `flatland.envs.callbacks.Defaultcallbacks. Can also be provided through env var callbacks_CLS (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--legacy-env-generator',
              type=bool,
              default=False,
              help="DEPRECATED: use the patched env_generator. Keep only for regression tests. Update tests and drop in separate pr.",
              required=False
              )
def generate_trajectories_from_metadata(
    metadata_csv: Path,
    data_dir: Path,
    policy: str = None,
    policy_pkg: str = None,
    policy_cls: str = None,
    obs_builder: str = None,
    obs_builder_pkg: str = None,
    obs_builder_cls: str = None,
    rewards: str = None,
    rewards_pkg: str = None,
    rewards_cls: str = None,
    callbacks: str = None,
    callbacks_pkg: str = None,
    callbacks_cls: str = None,
    legacy_env_generator: bool = False,
):
    metadata = pd.read_csv(metadata_csv)
    for k, v in metadata.iterrows():
        try:
            test_folder = data_dir / v["test_id"] / v["env_id"]
            test_folder.mkdir(parents=True, exist_ok=True)
            args = ["--data-dir", test_folder,
                    "--n-agents", v["n_agents"],
                    "--x-dim", v["x_dim"],
                    "--y-dim", v["y_dim"],
                    "--n-cities", v["n_cities"],
                    "--max-rail-pairs-in-city", v["max_rail_pairs_in_city"],
                    "--grid-mode", v["grid_mode"],
                    "--max-rails-between-cities", v["max_rails_between_cities"],
                    "--malfunction-duration-min", v["malfunction_duration_min"],
                    "--malfunction-duration-max", v["malfunction_duration_max"],
                    "--malfunction-interval", v["malfunction_interval"],
                    "--speed-ratios", "1.0", "0.25",
                    "--speed-ratios", "0.5", "0.25",
                    "--speed-ratios", "0.33", "0.25",
                    "--speed-ratios", "0.25", "0.25",
                    "--seed", v["seed"],
                    "--snapshot-interval", 0,
                    "--ep-id", v["test_id"] + "_" + v["env_id"]
                    ]
            if policy is not None:
                args += ["--policy", policy]
            if policy_pkg is not None:
                args += ["--policy-pkg", policy_pkg]
            if policy_cls is not None:
                args += ["--policy-cls", policy_cls]

            if obs_builder is not None:
                args += ["--obs-builder", obs_builder]
            if obs_builder_pkg is not None:
                args += ["--obs-builder-pkg", obs_builder_pkg]
            if obs_builder_cls is not None:
                args += ["--obs-builder-cls", obs_builder_cls]

            if rewards is not None:
                args += ["--rewards", rewards]
            if rewards_pkg is not None:
                args += ["--rewards-pkg", rewards_pkg]
            if rewards_cls is not None:
                args += ["--rewards-cls", rewards_cls]
            if legacy_env_generator:
                args += ["--legacy-env-generator", True]

            if callbacks is not None:
                args += ["--callbacks", callbacks]
            if callbacks_pkg is not None:
                args += ["--callbacks-pkg", callbacks_pkg]
            if callbacks_cls is not None:
                args += ["--callbacks-cls", callbacks_cls]

            generate_trajectory_from_policy(args)

        except SystemExit as exc:
            assert exc.code == 0



