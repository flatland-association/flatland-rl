"""
Runs Flatland env in RLlib using single policy learning, based on
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.

Take this as starting point to build your own training (cli) script.
"""
import logging
from argparse import Namespace
from typing import Union

import ray
from ray import tune
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import get_trainable_cls, register_env, registry_get_input

from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.wrappers import ray_env_creator


def setup_func():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")


def add_flatland_training_with_parameter_sharing_args():
    parser = add_rllib_example_script_args(
        default_iters=200,
        default_timesteps=1000000,
        default_reward=0.0,
    )
    parser.set_defaults(
        enable_new_api_stack=True
    )
    parser.add_argument(
        "--train-batch-size-per-learner",
        type=int,
        default=4000,
        help="See https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training",
    )
    parser.add_argument(
        "--obs-builder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ray-address",
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


# TODO allow for kwargs as well to be accessible from python directly and for easier injection?
def train(args: Namespace) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    assert (
        args.obs_builder
    ), "Must set --obs-builder <obs builder ID> when running this script!"
    # TODO revise whether we can get around this resrction relative path
    # assert os.path.exists("flatland/ml/ray/examples/environment.yml"), "Script must be executed in root folder of checked out flatland-rl."
    # assert os.path.exists("requirements-ml.txt"), "Script must be executed in root folder of checkout out flatland-rl."

    setup_func()
    init_args = {}
    if args.ray_address is not None:
        init_args['address'] = args.ray_address

    env_vars = set()
    if args.env_var is not None:
        env_vars = args.env_var
    # https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
    ray.init(
        **init_args,
        # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
        runtime_env={
            # install clean env from environment.yml - important for running in a cluster!
            # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference
            #"working_dir": ".",
            # TODO conda configurable, default to this only?!
            # "conda": "flatland/ml/ray/examples/environment.yml",
            # "conda": "flatland-rllib-cli",
            #"excludes": ["notebooks/", ".git/", ".tox/", ".venv/", "docs/", ".idea", "tmp"],
            "env_vars": dict(map(lambda s: s.split('='), env_vars)),
            # https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
            "worker_process_setup_hook": "flatland.ml.ray.examples.flatland_training_with_parameter_sharing.setup_func"
        },
        ignore_reinit_error=True,
    )
    # TODO allow for injection?!
    env_name = "flatland_env"
    register_env(env_name, lambda _: ray_env_creator(n_agents=args.num_agents, obs_builder_object=registry_get_input(args.obs_builder)()))
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("flatland_env")
        # TODO make configurable!
        .multi_agent(
            policies={"p0"},
            # All agents map to the exact same policy.
            policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        )
        .training(
            model={
                "vf_share_layers": True,
            },
            train_batch_size=args.train_batch_size_per_learner
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
    return res


# TODO verification of implementation with a proper model
# TODO https://github.com/flatland-association/flatland-rl/issues/73 get pettingzoo up and running again.
# TODO https://github.com/flatland-association/flatland-rl/issues/75 illustrate algorithm/policy abstraction in ray
# TODO https://github.com/flatland-association/flatland-rl/issues/76 illustrate generic callbacks with ray
# TODO https://github.com/flatland-association/flatland-rl/issues/77 illustrate logging (wandb/tensorflow/custom)...
if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_training_with_parameter_sharing_args()
    args = parser.parse_args()
    train(args)
