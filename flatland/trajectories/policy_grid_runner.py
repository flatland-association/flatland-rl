from pathlib import Path

import pandas as pd

from flatland.trajectories.policy_runner import generate_trajectory_from_policy


def generate_trajectories_from_metadata(
    metadata_csv: Path,
    data_dir: Path,
    policy_pkg: str, policy_cls: str,
    obs_builder_pkg: str, obs_builder_cls: str):
    metadata = pd.read_csv(metadata_csv)
    for k, v in metadata.iterrows():
        try:
            test_folder = data_dir / v["test_id"] / v["env_id"]
            test_folder.mkdir(parents=True, exist_ok=True)
            generate_trajectory_from_policy(
                ["--data-dir", test_folder,
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
                 ])
        except SystemExit as exc:
            assert exc.code == 0


if __name__ == '__main__':
    metadata_csv = Path("./episodes/trajectories/malfunction_deadlock_avoidance_heuristics/metadata.csv").resolve()
    data_dir = Path("./episodes/trajectories/malfunction_deadlock_avoidance_heuristics").resolve()
    generate_trajectories_from_metadata(
        metadata_csv=metadata_csv,
        data_dir=data_dir,
        policy_pkg="flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
        policy_cls="DeadLockAvoidancePolicy",
        obs_builder_pkg="flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
        obs_builder_cls="FullEnvObservation"
    )
