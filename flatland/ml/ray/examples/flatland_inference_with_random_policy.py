"""
Runs random rollout of Flatland env in RLlib, based on a combination of:
- https://github.com/ray-project/ray/blob/master/rllib/examples/inference/policy_inference_after_training.py
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/custom_heuristic_policy.py

See there how to load checkpoint into RlModule.

Take this as starting point to build your own inference (cli) script.
"""
import argparse
import tempfile
from argparse import Namespace
from pathlib import Path

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import registry_get_input

from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.wrappers import ray_env_generator, ray_policy_wrapper, ray_policy_wrapper_from_rllib_checkpoint
from flatland.trajectories.policy_runner import PolicyRunner


def add_flatland_inference_with_random_policy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--obs-builder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-episodes-during-inference",
        type=int,
        default=2,
        help="Number of episodes to do inference over (after restoring from a checkpoint).",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default=DEFAULT_MODULE_ID
    )
    parser.add_argument(
        "--cp",
        type=str,
        required=False,
        default=None
    )
    return parser


def rollout(args: Namespace):
    # Create an env to do inference in.
    env = ray_env_generator(n_agents=args.num_agents, obs_builder_object=registry_get_input(args.obs_builder)())
    obs, _ = env.reset()

    num_episodes = 0

    checkpoint_path = args.cp
    if checkpoint_path is not None:
        policy_id = args.policy_id
        policy = ray_policy_wrapper_from_rllib_checkpoint(checkpoint_path, policy_id)
    else:
        rl_module = RandomRLModule(action_space=env.action_space)
        policy = ray_policy_wrapper(rl_module)

    for _ in range(args.num_episodes_during_inference):
        env.reset()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = Path(tmpdirname)
            PolicyRunner.create_from_policy(env=env.wrap(), policy=policy, data_dir=data_dir)
    print(f"Done performing action inference through {num_episodes} Episodes")


if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_inference_with_random_policy_args()
    args = parser.parse_args()
    rollout(args)
