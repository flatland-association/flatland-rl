import pytest

from flatland.ml.ray.examples.flatland_inference_with_random_policy import rollout, add_flatland_ray_cli_example_script_args, \
    add_flatland_ray_cli_observation_builders


@pytest.mark.parametrize(
    "obid",
    [
        pytest.param(
            obid, id=obid
        )
        for obid in
        [
            "FlattenTreeObsForRailEnv_max_depth_3_50",
            "DummyObservationBuilderGym",
            "GlobalObsForRailEnvGym",
        ]
    ],
)
def test_rail_env_wrappers_random_rollout(obid: str):
    add_flatland_ray_cli_observation_builders()
    parser = add_flatland_ray_cli_example_script_args()
    rollout(parser.parse_args(
        ["--algo", "PPO", "--num-agents", "2", "--stop-iters", "1", "--obs_builder", obid]))
