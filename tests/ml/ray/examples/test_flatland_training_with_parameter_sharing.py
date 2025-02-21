import pytest
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN

from flatland.ml.ray.examples.flatland_inference_with_random_policy import add_flatland_inference_with_random_policy_args, rollout
from flatland.ml.ray.examples.flatland_training_with_parameter_sharing import train, add_flatland_training_with_parameter_sharing_args, \
    register_flatland_ray_cli_observation_builders


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
    results = train(parser.parse_args(
        ["--num-agents", "2", "--obs-builder", obid, "--algo", algo, "--stop-iters", "1", "--train-batch-size-per-learner", "200", "--checkpoint-freq", "1"]))
    best_result = results.get_best_result(
        metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max"
    )
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_inference_with_random_policy_args()
    rollout(parser.parse_args(["--num-agents", "2", "--obs-builder", obid, "--cp", best_result.checkpoint.path, "--policy-id", "p0"]))
