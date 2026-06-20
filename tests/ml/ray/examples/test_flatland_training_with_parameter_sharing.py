import tempfile
import warnings
from pathlib import Path
from zipfile import ZipFile

import pytest
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS
from ray.tune.registry import registry_get_input

from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.examples.flatland_rollout import add_flatland_inference_from_checkpoint, rollout_from_checkpoint
from flatland.ml.ray.examples.flatland_training_with_parameter_sharing import add_flatland_training_with_parameter_sharing_args, \
    _get_algo_config_parameter_sharing, train_with_parameter_sharing_cli


@pytest.mark.parametrize(
    "obid,algo",
    [
        pytest.param(
            obid, algo, id=f"{obid}_{algo}"
        )
        for obid in
        [
            "DummyObservationBuilderGym",
            "GlobalObsForRailEnvGym",
            "FlattenedNormalizedTreeObsForRailEnv_max_depth_3_50",

        ]
        for algo in
        [
            # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
            "PPO",
            "DQN",
            "IMPALA",
            "APPO",
        ]
    ]
)
@pytest.mark.slow
def test_rail_env_wrappers_training_and_rollout(obid: str, algo: str):
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_training_with_parameter_sharing_args()
    evaluation_duration = 2
    stop_iters = 2
    args = parser.parse_args(
        ["--num-agents", "2", "--obs-builder", obid, "--algo", algo, "--stop-iters", f"{stop_iters}", "--train-batch-size-per-learner", "200",
         "--checkpoint-freq", "1", "--evaluation-interval", "1", "--evaluation-duration", f"{evaluation_duration}"])

    results = train_with_parameter_sharing_cli(args)

    best_result = results.get_best_result(
        metric=f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max"
    )
    assert f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/percentage_complete" in best_result.metrics_dataframe.keys()
    assert f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/normalized_reward" in best_result.metrics_dataframe.keys()
    assert f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/trajectory" in best_result.metrics_dataframe.keys()

    if f"{ENV_RUNNER_RESULTS}/percentage_complete" not in best_result.metrics_dataframe.keys():
        warnings.warn(
            f"Key {ENV_RUNNER_RESULTS}/percentage_complete not found in best_result.metrics_dataframe.keys(). Training has never completed an episode.")
    if f"{ENV_RUNNER_RESULTS}/normalized_reward" not in best_result.metrics_dataframe.keys():
        warnings.warn(f"Key {ENV_RUNNER_RESULTS}/normalized_reward not found in best_result.metrics_dataframe.keys(). Training has never completed an episode.")
    assert f"{ENV_RUNNER_RESULTS}/trajectory" not in best_result.metrics_dataframe.keys()
    trajectory_ = best_result.metrics_dataframe[f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/trajectory"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        with (data_dir / "trajectory.zip").open("wb") as f:
            f.write(bytes.fromhex(trajectory_.iloc[0]))
        with ZipFile(data_dir / "trajectory.zip") as myzip:
            assert len([n for n in myzip.namelist() if n.endswith("TrainMovementEvents.trains_positions.tsv")]) == 1

    # TODO why unregistered again?
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_inference_from_checkpoint()
    config = _get_algo_config_parameter_sharing(args=args, obs_builder_class=registry_get_input(args.obs_builder))

    # TODO fix
    rollout_from_checkpoint(parser.parse_args(["--num-agents", "2", "--obs-builder", obid, "--cp", best_result.checkpoint.path, "--policy-id", "p0"]),
                            algo=config.build_algo())
