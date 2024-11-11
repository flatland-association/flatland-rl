import pytest

from flatland.ml.ray.examples.flatland_training_with_parameter_sharing import train, add_flatland_ray_cli_example_script_args, \
    add_flatland_ray_cli_observation_builders


# TODO takes too long for unit tests -> mark as skip or IT? Or reduce training?
@pytest.mark.parametrize(
    "obid,algo",
    [
        pytest.param(
            obid, algo, id=f"{obid}_{algo}"
        )
        for obid in
        [
            "FlattenTreeObsForRailEnv_max_depth_3_50",
            "DummyObservationBuilderGym",
            "GlobalObsForRailEnvGym",
        ]
        for algo in
        [
            # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html

            "PPO",
            # TODO DQN not working yet - use latest ray with new api stack?
            #   File "/tmp/ray/session_2024-10-28_09-26-41_604177_64833/runtime_resources/conda/54f41acf3bda09e1ccf6469b1e424bd2f43fc0b0/lib/python3.10/site-packages/ray/rllib/utils/replay_buffers/multi_agent_replay_buffer.py", line 224, in add
            #     batch = batch.as_multi_agent()
            # AttributeError: 'list' object has no attribute 'as_multi_agent'
            # "DQN",
            "IMPALA",
            "APPO",
        ]
    ]
)
# TODO PytestUnknownMarkWarning
@pytest.mark.integrationtest
def test_rail_env_wrappers_training(obid: str, algo: str):
    add_flatland_ray_cli_observation_builders()
    parser = add_flatland_ray_cli_example_script_args()
    train(parser.parse_args(
        ["--algo", algo, "--num-agents", "2", "--stop-iters", "1", "--obs_builder", obid]))

# TODO verification of implementation with training? 0.6 bei 1000 Episode
