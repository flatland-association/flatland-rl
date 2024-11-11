"""
Runs random rollout of Flatland env in RLlib, based on a combination of:
- https://github.com/ray-project/ray/blob/master/rllib/examples/inference/policy_inference_after_training.py
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/custom_heuristic_policy.py

See there how to load checkpoint into RlModule.

Take this as starting point to build your own inference cli.
"""
from argparse import Namespace

import numpy as np
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import registry_get_input, register_input

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym
from flatland.ml.ray.wrappers import ray_env_creator


def add_flatland_ray_cli_observation_builders():
    register_input("DummyObservationBuilderGym", lambda: DummyObservationBuilderGym()),
    register_input("GlobalObsForRailEnvGym", lambda: GlobalObsForRailEnvGym()),
    register_input("FlattenTreeObsForRailEnv_max_depth_3_50",
                   lambda: FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))


def add_flatland_ray_cli_example_script_args():
    parser = add_rllib_example_script_args(
        default_iters=200,
        default_timesteps=1000000,
        default_reward=0.0,
    )
    parser.set_defaults(
        enable_new_api_stack=True
    )

    parser.add_argument(
        "--obs_builder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-episodes-during-inference",
        type=int,
        default=10,
        help="Number of episodes to do inference over (after restoring from a checkpoint).",
    )
    return parser


def rollout(args: Namespace):
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    # Create an env to do inference in.
    env = ray_env_creator(n_agents=args.num_agents, obs_builder_object=registry_get_input(args.obs_builder)())
    env.reset()

    num_episodes = 0
    episode_return = 0.0

    while num_episodes < args.num_episodes_during_inference:
        action_dict = {
            str(i): np.random.choice(5) for i in range(args.num_agents)
        }
        _, rewards, terminateds, truncateds, _ = env.step(action_dict)
        for _, v in rewards.items():
            episode_return += v
        # Is the episode `done`? -> Reset.
        if terminateds["__all__"] or truncateds["__all__"]:
            print(f"Episode done: Total reward = {episode_return}")
            env.reset()
            num_episodes += 1
            episode_return = 0.0

    print(f"Done performing action inference through {num_episodes} Episodes")


if __name__ == '__main__':
    add_flatland_ray_cli_observation_builders()
    parser = add_flatland_ray_cli_example_script_args()
    args = parser.parse_args()
    rollout(args)
