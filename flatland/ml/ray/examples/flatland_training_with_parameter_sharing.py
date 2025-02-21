"""
Runs Flatland env in RLlib using single policy learning, based on
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.

Take this as starting point to build your own training (cli) script.
"""
import argparse
import logging
from typing import Union, Optional

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
from flatland.ml.ray.wrappers import ray_env_generator


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


def train(args: Optional[argparse.Namespace] = None, init_args=None) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    if args is None:
        parser = add_flatland_training_with_parameter_sharing_args()
        args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    assert (
        args.obs_builder
    ), "Must set --obs-builder <obs builder ID> when running this script!"

    setup_func()
    if init_args is None:
        env_vars = set()
        if args.env_var is not None:
            env_vars = args.env_var
        init_args = {
            # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
            "runtime_env": {
                "env_vars": dict(map(lambda s: s.split('='), env_vars)),
                # https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
                "worker_process_setup_hook": "flatland.ml.ray.examples.flatland_training_with_parameter_sharing.setup_func"
            },
            "ignore_reinit_error": True,
        }
        if args.ray_address is not None:
            init_args['address'] = args.ray_address

    # https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
    ray.init(
        **init_args,
    )
    env_name = "flatland_env"
    register_env(env_name, lambda _: ray_env_generator(n_agents=args.num_agents, obs_builder_object=registry_get_input(args.obs_builder)()))

    # TODO could be extracted to cli - keep it low key as illustration only
    additional_training_config = {}
    if args.algo == "DQN":
        additional_training_config = {"replay_buffer_config": {
            "type": "MultiAgentEpisodeReplayBuffer",
        }}
    base_config = (
        # N.B. the warning `passive_env_checker.py:164: UserWarning: WARN: The obs returned by the `reset()` method was expecting numpy array dtype to be float32, actual type: float64`
        #   comes from ray.tune.registry._register_all() -->  import ray.rllib.algorithms.dreamerv3 as dreamerv3!
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
            train_batch_size=args.train_batch_size_per_learner,
            **additional_training_config
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


# TODO https://github.com/flatland-association/flatland-rl/issues/100 verify implementation
# TODO https://github.com/flatland-association/flatland-rl/issues/73 get pettingzoo up and running again.
# TODO https://github.com/flatland-association/flatland-rl/issues/75 illustrate algorithm/policy abstraction in ray
# TODO https://github.com/flatland-association/flatland-rl/issues/76 illustrate generic callbacks with ray
# TODO https://github.com/flatland-association/flatland-rl/issues/77 illustrate logging (wandb/tensorflow/custom)...
if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_training_with_parameter_sharing_args()
    args = parser.parse_args()
    train(args)
