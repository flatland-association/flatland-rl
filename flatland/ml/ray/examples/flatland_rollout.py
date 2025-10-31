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

import numpy as np
from ray.rllib.algorithms import Algorithm
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import registry_get_input

from flatland.ml.ray.wrappers import ray_env_generator, ray_policy_wrapper, ray_policy_wrapper_from_rllib_checkpoint
from flatland.trajectories.policy_runner import PolicyRunner


# TODO drop argparse
def add_flatland_inference_from_checkpoint():
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


# TODO unused?
def random_rollout(args: Namespace):
    num_episodes_during_inference = args.num_episodes_during_inference
    env = ray_env_generator(n_agents=args.num_agents, seed=int(np.random.default_rng().integers(2 ** 32 - 1)),
                            obs_builder_object=registry_get_input(args.obs_builder)())
    obs, _ = env.reset()
    rl_module = RandomRLModule(action_space=env.action_space)
    policy = ray_policy_wrapper(rl_module)

    do_rollout(env, num_episodes_during_inference, policy)


def rollout_from_checkpoint(args: Namespace, algo: Algorithm):
    # Create an env to do inference in.
    env = ray_env_generator(n_agents=args.num_agents, seed=int(np.random.default_rng().integers(2 ** 32 - 1)),
                            obs_builder_object=registry_get_input(args.obs_builder)())
    obs, _ = env.reset()

    checkpoint_path = args.cp
    policy_id = args.policy_id

    policy = ray_policy_wrapper_from_rllib_checkpoint(checkpoint_path, algo, policy_id)

    num_episodes_during_inference = args.num_episodes_during_inference
    do_rollout(env, num_episodes_during_inference, policy)


def do_rollout(env, num_episodes_during_inference, policy):
    for _ in range(num_episodes_during_inference):
        env.reset()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = Path(tmpdirname)
            t = PolicyRunner.create_from_policy(env=env.wrap(), policy=policy, data_dir=data_dir)
            print(t.trains_arrived_lookup())
            print(t.actions)
    # TODO add option for data_dir and analysis on it
    print(f"Done performing action inference through {num_episodes_during_inference} Episodes")
