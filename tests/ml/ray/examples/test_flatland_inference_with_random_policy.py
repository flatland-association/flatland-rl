import pytest

from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.examples.flatland_rollout import add_flatland_inference_from_checkpoint, random_rollout


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
    parser = add_flatland_inference_from_checkpoint()
    random_rollout(parser.parse_args(["--num-agents", "2", "--obs-builder", obid]))
