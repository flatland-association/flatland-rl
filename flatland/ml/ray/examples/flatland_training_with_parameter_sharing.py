"""
Runs Flatland env in RLlib using single policy learning, based on
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.

Take this as starting point to build your own training cli.
"""
import logging
import os.path
from argparse import Namespace

import ray
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env, registry_get_input, register_input

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym
from flatland.ml.ray.wrappers import ray_env_creator


def setup_func():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")


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
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        required=False,
        help="The address of the ray cluster to connect to in the form ray://<head_node_ip_address>:10001. Leave empty to start a new cluster. Passed to ray.init(address=...). See https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html ",
    )
    parser.add_argument("--env_var", "-e",
                        metavar="KEY=VALUE",
                        nargs='*',
                        help="Set ray runtime environment variables like -e RAY_DEBUG=1, passed to ray.init(runtime_env={env_vars: {...}}), see https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference")
    return parser


def add_flatland_ray_cli_observation_builders():
    register_input("DummyObservationBuilderGym", lambda: DummyObservationBuilderGym()),
    register_input("GlobalObsForRailEnvGym", lambda: GlobalObsForRailEnvGym()),
    register_input("FlattenTreeObsForRailEnv_max_depth_3_50",
                   lambda: FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))


def train(args: Namespace):
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    assert (
        args.obs_builder
    ), "Must set --obs_builder <obs builder ID> when running this script!"
    assert os.path.exists("flatland/ml/ray/examples/environment.yml"), "Script must be executed in root folder of checked out flatland-rl."
    assert os.path.exists("requirements-ml.txt"), "Script must be executed in root folder of checkout out flatland-rl."

    setup_func()
    kwargs = {}
    if args.ray_address is not None:
        kwargs['address'] = args.ray_address

    env_vars = set()
    if args.env_var is not None:
        env_vars = args.env_var
    # https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
    ray.init(
        **kwargs,
        # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
        runtime_env={
            # install clean env from environment.yml - important for running in a cluster!
            # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference
            "working_dir": ".",
            "conda": "flatland/ml/ray/examples/environment.yml",
            "excludes": ["notebooks/", ".git/", ".tox/", ".venv/", "docs/", ".idea", "tmp"],
            "env_vars": dict(map(lambda s: s.split('='), env_vars)),
            # https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
            "worker_process_setup_hook": "flatland.ml.ray.examples.flatland_training_with_parameter_sharing.setup_func"
        }
    )
    try:
        env_name = "flatland_env"
        register_env(env_name, lambda _: ray_env_creator(n_agents=args.num_agents, obs_builder_object=registry_get_input(args.obs_builder)()))
        base_config = (
            get_trainable_cls(args.algo)
            .get_default_config()
            .environment("flatland_env")
            .multi_agent(
                policies={"p0"},
                # All agents map to the exact same policy.
                policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
            )
            .training(
                model={
                    "vf_share_layers": True,
                },
                vf_loss_coeff=0.005,
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={"p0": RLModuleSpec()},
                )
            )
        )
        res = run_rllib_example_script_experiment(base_config, args)
        if res.num_errors > 0:
            raise AssertionError(f"{res.errors}")
        ray.shutdown()
    except BaseException as e:
        ray.shutdown()
        raise e


# TODO verification of implementation with a proper model
# TODO https://github.com/flatland-association/flatland-rl/issues/73 get pettingzoo up and running again.
# TODO https://github.com/flatland-association/flatland-rl/issues/75 illustrate algorithm/policy abstraction in ray
# TODO https://github.com/flatland-association/flatland-rl/issues/76 illustrate generic callbacks with ray
# TODO https://github.com/flatland-association/flatland-rl/issues/77 illustrate logging (wandb/tensorflow/custom)...
if __name__ == '__main__':
    add_flatland_ray_cli_observation_builders()
    parser = add_flatland_ray_cli_example_script_args()
    args = parser.parse_args()
    train(args)
