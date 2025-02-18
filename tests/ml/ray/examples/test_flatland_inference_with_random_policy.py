import pytest

from flatland.ml.ray.examples.flatland_inference_with_random_policy import add_flatland_inference_with_random_policy_args
from flatland.ml.ray.examples.flatland_inference_with_random_policy import rollout, register_flatland_ray_cli_observation_builders


@pytest.mark.parametrize(
    "obid",
    [
        pytest.param(
            obid, id=obid
        )
        for obid in
        [
            "FlattenedNormalizedTreeObsForRailEnv_max_depth_3_50",
            "DummyObservationBuilderGym",
            "GlobalObsForRailEnvGym",
        ]
    ],
)
def test_rail_env_wrappers_random_rollout(obid: str):
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_inference_with_random_policy_args()
    rollout(parser.parse_args(
        ["--num-agents", "2", "--obs-builder", obid]))
