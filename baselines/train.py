from flatland.envs import rail_env
from flatland.envs.rail_env import random_rail_generator
from baselines.RailEnvRLLibWrapper import RailEnvRLLibWrapper
from flatland.utils.rendertools import RenderTool
import random
import gym

import matplotlib.pyplot as plt

from flatland.envs.generators import complex_rail_generator

import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.dqn.dqn as dqn
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.rllib.models.preprocessors import Preprocessor


import ray
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# RailEnv.__bases__ = (RailEnv.__bases__[0], MultiAgentEnv)


class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (105,)

    def transform(self, observation):
        return observation  # return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
ray.init()

def train(config):
    print('Init Env')
    random.seed(1)
    np.random.seed(1)

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    transition_probability = [0.5,  # empty cell - Case 0
                              1.0,  # Case 1 - straight
                              1.0,  # Case 2 - simple switch
                              0.3,  # Case 3 - diamond drossing
                              0.5,  # Case 4 - single slip
                              0.5,  # Case 5 - double slip
                              0.2,  # Case 6 - symmetrical
                              0.0]  # Case 7 - dead end



    # Example generate a random rail
    env = RailEnvRLLibWrapper(width=15, height=15,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=20, min_dist=12),
                  number_of_agents=10)

    register_env("railenv", lambda _: env)
    # if config['render']:
    #     env_renderer = RenderTool(env, gl="QT")
    # plt.figure(figsize=(5,5))

    obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(105,))
    act_space = gym.spaces.Discrete(4)

    # Dict with the different policies to train
    policy_graphs = {
        "ppo_policy": (PPOPolicyGraph, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id):
        return f"ppo_policy"

    agent_config = ppo.DEFAULT_CONFIG.copy()
    agent_config['model'] = {"fcnet_hiddens": [32, 32], "custom_preprocessor": "my_prep"}
    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                  "policy_mapping_fn": policy_mapping_fn,
                                  "policies_to_train": list(policy_graphs.keys())}
    agent_config["horizon"] = 50
    agent_config["num_workers"] = 0
    agent_config["num_workers"] = 0
    agent_config["num_cpus_per_worker"] = 40
    agent_config["num_gpus"] = 2.0
    agent_config["num_gpus_per_worker"] = 2.0
    agent_config["num_cpus_for_driver"] = 5
    agent_config["num_envs_per_worker"] = 20
    #agent_config["batch_mode"] = "complete_episodes"

    ppo_trainer = PPOAgent(env=f"railenv", config=agent_config)

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # if i % config['save_every'] == 0:
        #     checkpoint = ppo_trainer.save()
        #     print("checkpoint saved at", checkpoint)

train({})






