import numpy as np
from ray import tune
# from ray.rllib.utils import try_import_tf
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from flatland.contrib.interface import flatland_env
from flatland.contrib.utils import env_generators
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

# Custom observation builder with predictor, uncomment line below if you want to try this one
observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))
seed = 10
np.random.seed(seed)
wandb_log = False
experiment_name = "flatland_pettingzoo"
rail_env = env_generators.small_v0(seed, observation_builder)

# __sphinx_doc_begin__


def env_generator(args):
    env = flatland_env.parallel_env(environment=rail_env, use_renderer=False)
    return env


if __name__ == "__main__":
    env_name = "flatland_pettyzoo"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_generator(config)))

    test_env = ParallelPettingZooEnv(env_generator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    policies = {"policy_0": gen_policy(0)}

    policy_ids = list(policies.keys())

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            # https://github.com/ray-project/ray/issues/10761
            "no_done_at_end": True,
            # "soft_horizon" : True,
            "num_gpus": 0,
            "num_workers": 2,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',
            "clip_rewards": False,
            "vf_clip_param": 500.0,
            "entropy_coeff": 0.01,
            # effective batch_size: train_batch_size * num_agents_in_each_environment [5, 10]
            # see https://github.com/ray-project/ray/issues/4628
            "train_batch_size": 1000,  # 5000
            "rollout_fragment_length": 50,  # 100
            "sgd_minibatch_size": 100,  # 500
            "vf_share_layers": False
            },
    )

# __sphinx_doc_end__
