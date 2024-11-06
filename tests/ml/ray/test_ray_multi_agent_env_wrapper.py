from typing import Callable

import pytest
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import register_input

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym
from flatland.ml.ray.ray_cli import train, add_flatland_ray_cli_example_script_args
from flatland.ml.ray.wrappers import ray_env_creator


@pytest.mark.parametrize(
    "obs_builder_object",
    [
        # pytest.param(DeadLockAvoidancePolicy(), id="DeadLockAvoidancePolicy"),
        pytest.param(
            DummyObservationBuilder(), id="DummyObservationBuilder"
        ),
        pytest.param(
            GlobalObsForRailEnv(), id="GlobalObsForRailEnv"
        ),
        pytest.param(
            FlattenTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), id="FlattenTreeObsForRailEnv_max_depth_2_50"
        ),
        pytest.param(
            FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), id="FlattenTreeObsForRailEnv_max_depth_3_50"
        ),
    ],
)
@pytest.mark.skip(reason="TODO implement random policy")
def test_rail_env_wrappers_random_rollout(obs_builder_object: ObservationBuilder):
    env = ray_env_creator(obs_builder_object=obs_builder_object)
    worker = RolloutWorker(
        env_creator=lambda _: env,
        config=AlgorithmConfig().experimental(_disable_preprocessor_api=True).multi_agent(
            policies={
                # TODO use random policy for now
                # f"main": (DeadLockAvoidancePolicy, env.observation_space["0"], env.action_space["0"], {'env': env})
            },
            policy_mapping_fn=(
                lambda aid, episode, **kwargs: f"main"
            )
        )
    )
    worker.sample()


@pytest.mark.parametrize(
    "obs_builder,expected_shape",
    [
        pytest.param(obs_builder, expected_shape, id=f"{obid}")
        for obs_builder, obid, expected_shape in
        [
            (FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50", (935,)),
            (DummyObservationBuilderGym(), "DummyObservationBuilderGym", (1,)),
            (GlobalObsForRailEnvGym(), "GlobalObsForRailEnvGym", (20700,)),
        ]
    ]
)
def test_obs_builder_gym(obs_builder: ObservationBuilder, expected_shape):
    expected_dtype = float
    expected_agent_ids = ['0', '1', '2', '3', '4', '5', '6']

    env = ray_env_creator(obs_builder_object=obs_builder)

    assert env.agents == expected_agent_ids, env.agents
    for agent_id in env.agents:
        space_shape = env.get_observation_space(agent_id).shape
        assert space_shape == expected_shape, (expected_shape, space_shape)
        space_dtype = env.get_observation_space(agent_id).dtype
        assert space_dtype == expected_dtype
        sample_shape = env.get_observation_space(agent_id).sample().shape
        assert sample_shape == expected_shape, (expected_shape, sample_shape)
    obs, _ = env.reset()
    assert list(obs.keys()) == expected_agent_ids
    for i in range(7):
        assert obs[str(i)].shape == expected_shape
        assert obs[str(i)].dtype == expected_dtype
    obs, _, _, _, _ = env.step({})
    assert list(obs.keys()) == expected_agent_ids
    for i in range(7):
        assert obs[str(i)].shape == expected_shape
        assert obs[str(i)].dtype == expected_dtype


# TODO takes too long for unit tests -> mark as skip or IT? Or reduce training?
@pytest.mark.parametrize(
    "obs_builder,algo",
    [
        pytest.param(
            obs_builder, algo, id=f"{obid}_{algo}"
        )
        for obs_builder, obid in
        [
            (lambda: FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50"),
            (lambda: DummyObservationBuilderGym(), "DummyObservationBuilderGym"),
            (lambda: GlobalObsForRailEnvGym(), "GlobalObsForRailEnvGym"),
        ]
        for algo in [
        # TODO DQN not working yet - use latest ray with new api stack?
        #   File "/tmp/ray/session_2024-10-28_09-26-41_604177_64833/runtime_resources/conda/54f41acf3bda09e1ccf6469b1e424bd2f43fc0b0/lib/python3.10/site-packages/ray/rllib/utils/replay_buffers/multi_agent_replay_buffer.py", line 224, in add
        #     batch = batch.as_multi_agent()
        # AttributeError: 'list' object has no attribute 'as_multi_agent'
        # "DQN",
        "PPO"]
    ]
)
@pytest.mark.integrationtest
def test_rail_env_wrappers_training(obs_builder: Callable[[], ObservationBuilder], algo: str):
    parser = add_flatland_ray_cli_example_script_args()
    # TODO is register_input the right way?
    register_input("test_rail_env_wrappers_training_obs_builder", obs_builder)
    train(parser.parse_args(
        ["--algo", algo, "--num-agents", "2", "--enable-new-api-stack", "--stop-iters", "1", "--obs_builder", "test_rail_env_wrappers_training_obs_builder"]))

# TODO verification of implementation with training? 0.6 bei 1000 Episode
