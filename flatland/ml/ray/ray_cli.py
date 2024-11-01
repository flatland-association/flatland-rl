import logging
from argparse import Namespace

import ray
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.ml.ray.wrappers import ray_env_creator


def setup_func():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)


def train(args: Namespace):
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    setup_func()
    ray.init(runtime_env={
        # TODO cleanup: do without environment file (relative paths), maybe generate ad hoc to inject requirements-ml.txt
        # install clean env fro
        # "conda": "environment.yml",
        # TODO cleanup: pass working dir from cli?
        # "working_dir": f"{Path.cwd().parent.parent.parent.parent}",
        "working_dir": f".",
        # "working_dir": "../../../..",
        "excludes": ["notebooks/", ".git/", ".tox/", ".venv/", "docs/", ".idea", "tmp"],
        "env_vars": {
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1",
            # "RAY_DEBUG": "1",
        },
        "worker_process_setup_hook": setup_func
    })
    try:
        env_name = "flatland_env"
        # TODO inject whole env from cli?
        # TODO can we wrap instead?
        obs_builder_object = FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))
        register_env(env_name, lambda _: ray_env_creator(n_agents=args.num_agents, obs_builder_object=obs_builder_object))
        # Policies are called just like the agents (exact 1:1 mapping).
        policies = {str(i) for i in range(args.num_agents)}
        base_config = (
            get_trainable_cls(args.algo)
            .get_default_config()
            .environment("flatland_env")
            .multi_agent(
                policies=policies,
                # Exact 1:1 mapping from AgentID to ModuleID.
                policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
            )
            .training(
                # vf_loss_coeff=0.005,
            )
            .rl_module(
                model_config_dict={"vf_share_layers": True},
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={p: RLModuleSpec() for p in policies},
                ),
            )

        )
        res = run_rllib_example_script_experiment(base_config, args)
        if res.num_errors > 0:
            raise AssertionError(f"{res.errors}")
        ray.shutdown()
    except BaseException as e:
        ray.shutdown()
        raise e


# TODO convert to full cli (with which options) or convert to example?
# TODO documentation/cli ray cluster?
# TODO https://github.com/flatland-association/flatland-rl/issues/73 get pettingzoo up and running again.
# TODO https://github.com/flatland-association/flatland-rl/issues/75 illustrate algorithm/policy abstraction in ray
# TODO https://github.com/flatland-association/flatland-rl/issues/76 illustrate generic callbacks with ray
# TODO https://github.com/flatland-association/flatland-rl/issues/77 illustrate logging (wandb/tensorflow/custom)...
if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
