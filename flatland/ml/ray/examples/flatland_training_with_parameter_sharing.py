"""
Runs Flatland env in RLlib using single policy learning, based on
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.

Take this as starting point to build your own training (cli) script.
"""

import argparse
import importlib
import logging
from typing import Union, Optional, Dict, Any, Type

import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env, registry_get_input

from flatland.ml.observations.gym_observation_builder import GymObservationBuilder
from flatland.ml.ray.wrappers import ray_env_generator


def train_with_parameter_sharing(
    module_class: Type[RLModule] = None,
    obs_builder_class: Type[GymObservationBuilder] = None,
    args: Optional[argparse.Namespace] = None,  # args from add_rllib_example_script_args
    ray_address: str = None,
    init_args=None,
    env_vars=None,
    train_batch_size_per_learner: int = 4000,
    additional_training_config: Dict[str, Any] = None,
    env_config: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    callbacks_pkg: Optional[str] = None,
    callbacks_cls: Optional[str] = None,
    evaluation_callbacks_cls: Optional[str] = None,
    evaluation_callbacks_pkg: Optional[str] = None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    base_config = _get_algo_config_parameter_sharing(
        module_class=module_class,
        obs_builder_class=obs_builder_class,
        args=args,
        ray_address=ray_address,
        init_args=init_args,
        env_vars=env_vars,
        train_batch_size_per_learner=train_batch_size_per_learner,
        additional_training_config=additional_training_config,
        env_config=env_config,
        model_config=model_config,
        callbacks_pkg=callbacks_pkg,
        callbacks_cls=callbacks_cls,
        evaluation_callbacks_cls=evaluation_callbacks_cls,
        evaluation_callbacks_pkg=evaluation_callbacks_pkg,
    )

    # TODO do plain RLlib instead of using run_rllib_example_script_experiment (which config options do we need?) and then add --checkpoint-path to inject resuming training from checkpoint
    res = run_rllib_example_script_experiment(base_config, args)
    if res.num_errors > 0:
        raise AssertionError(f"{res.errors}")
    return res


def _get_algo_config_parameter_sharing(
    module_class: Type[RLModule] = None,
    obs_builder_class: Type[GymObservationBuilder] = None,
    args: Optional[argparse.Namespace] = None,  # args from add_rllib_example_script_args
    ray_address: str = None,
    init_args=None,
    env_vars=None,
    train_batch_size_per_learner: int = 4000,
    additional_training_config: Dict[str, Any] = None,
    env_config: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    callbacks_pkg: Optional[str] = None,
    callbacks_cls: Optional[str] = None,
    evaluation_callbacks_cls: Optional[str] = None,
    evaluation_callbacks_pkg: Optional[str] = None,
) -> AlgorithmConfig:
    setup_func()
    if args is None:
        parser = add_rllib_example_script_args()
        args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    if init_args is None:

        init_args = {
            # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
            "runtime_env": {
                "env_vars": env_vars or {},
                # https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
                "worker_process_setup_hook": "flatland.ml.ray.examples.flatland_training_with_parameter_sharing.setup_func"
            },
            "ignore_reinit_error": True,
        }

        if ray_address is not None:
            init_args['address'] = ray_address
    # https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
    # TODO already done in run experiment!
    ray.init(
        **init_args,
    )
    # TODO pass as env_config
    if env_config is None:
        env_config = {}
    # TODO cli add posibility to use registered input as well
    env_name = "flatland_env"
    register_env(env_name, lambda _: ray_env_generator(
        **env_config,
        n_agents=args.num_agents,
        obs_builder_object=obs_builder_class()
    ))
    # TODO cleanup, should be caller's responsibility
    if args.algo == "DQN":
        additional_training_config = {"replay_buffer_config": {
            "type": "MultiAgentEpisodeReplayBuffer",
        }}
    base_config: AlgorithmConfig = (
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
            train_batch_size=train_batch_size_per_learner,
            **(additional_training_config or {})
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"p0": RLModuleSpec(
                    module_class=module_class,
                    model_config=model_config,
                )},
            )
        )
        # https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html#algorithmconfig-env-runners
        .env_runners(create_env_on_local_worker=True)
    )
    if callbacks_pkg is not None and callbacks_cls is not None:
        module = importlib.import_module(callbacks_pkg)
        callbacks = getattr(module, callbacks_cls)
        base_config = base_config.callbacks(callbacks)
    if evaluation_callbacks_pkg is not None and evaluation_callbacks_cls is not None:
        module = importlib.import_module(evaluation_callbacks_pkg)
        callbacks = getattr(module, evaluation_callbacks_cls)
        base_config = base_config.evaluation(
            evaluation_config=AlgorithmConfig.overrides(callbacks=callbacks),
        )
    return base_config


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
        help="Must be registered input."
    )
    parser.add_argument(
        "--module-class",
        type=str,
        default=None,
        help="Must be registered input."
    )
    parser.add_argument("--model-config",
                        metavar="KEY=VALUE",
                        nargs='*',
                        help="Passed to model_config.")
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        required=False,
        help="The address of the ray cluster to connect to in the form ray://<head_node_ip_address>:10001. Leave empty to start a new cluster. Passed to ray.init(address=...). See https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html ",
    )
    parser.add_argument(
        '--callbacks-pkg',
        type=str,
        help="Defaults to `flatland.ml.ray.flatland_metrics_callback.FlatlandMetricsCallback`",
        required=False,
        default="flatland.ml.ray.flatland_metrics_callback"
    )
    parser.add_argument(
        '--callbacks-cls',
        type=str,
        help="Defaults to `flatland.ml.ray.flatland_metrics_callback.FlatlandMetricsCallback`",
        required=False,
        default="FlatlandMetricsCallback"
    )
    parser.add_argument(
        '--evaluation-callbacks-pkg',
        type=str,
        help="Defaults to `flatland.ml.ray.flatland_metrics_and_trajectory_callback.FlatlandMetricsAndTrajectoryCallback`",
        required=False,
        default="flatland.ml.ray.flatland_metrics_and_trajectory_callback"
    )
    parser.add_argument(
        '--evaluation-callbacks-cls',
        type=str,
        help="Defaults to `flatland.ml.ray.flatland_metrics_and_trajectory_callback.FlatlandMetricsAndTrajectoryCallback`",
        required=False,
        default="FlatlandMetricsAndTrajectoryCallback"
    )
    parser.add_argument("--env-var", "-e",
                        metavar="KEY=VALUE",
                        nargs='*',
                        help="Set ray runtime environment variables like -e RAY_DEBUG=1, passed to ray.init(runtime_env={env_vars: {...}}), see https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference")
    # parser.add_argument("--env-config",
    #                     metavar="KEY=VALUE",
    #                     nargs='*',
    #                     help="Passed to env generator.")  # TODO use literal_eval instead?
    return parser


def train_with_parameter_sharing_cli(args: Optional[argparse.Namespace] = None) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    if args is None:
        parser = add_flatland_training_with_parameter_sharing_args()
        args = parser.parse_args()
    assert (
        args.obs_builder
    ), "Must set --obs-builder <obs builder ID> when running this script!"

    env_vars = {}
    if args.env_var is not None:
        env_vars = dict(map(lambda s: s.split('='), args.env_var))

    model_config = None
    if args.model_config is not None:
        model_config = dict(map(lambda s: s.split('='), model_config))

    return train_with_parameter_sharing(
        module_class=registry_get_input(args.module_class) if args.module_class is not None else None,
        obs_builder_class=registry_get_input(args.obs_builder),
        args=args,
        init_args=None, env_vars=env_vars,
        train_batch_size_per_learner=args.train_batch_size_per_learner,
        additional_training_config={}, env_config=None,
        model_config=model_config,
        callbacks_pkg=args.callbacks_pkg,
        callbacks_cls=args.callbacks_cls,
        evaluation_callbacks_pkg=args.evaluation_callbacks_pkg,
        evaluation_callbacks_cls=args.evaluation_callbacks_cls,
    )
