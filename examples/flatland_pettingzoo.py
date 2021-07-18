
import numpy as np
import os

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import supersuit as ss

import flatland_env
import env_generators

from gym.wrappers import monitor
from flatland.envs.observations import TreeObsForRailEnv,GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
import wandb

# Custom observation builder without predictor
# observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor, uncomment line below if you want to try this one
observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
seed = 10
experiment_name= "flatland_pettingzoo"
# rail_env = env_generators.sparse_env_small(seed, observation_builder)
rail_env = env_generators.random_sparse_env_small(seed, observation_builder)
env = flatland_env.parallel_env(environment = rail_env, use_renderer = False)
run = wandb.init(project="flatland2021", entity="nilabha2007", sync_tensorboard=True, config={}, name=experiment_name, save_code=True)

env = ss.pettingzoo_env_to_vec_env_v0(env)
env.black_death = True
env = ss.concat_vec_envs_v0(env, 3, num_cpus=3, base_class='stable_baselines3')
model = PPO(MlpPolicy, env, tensorboard_log = f"/tmp/{experiment_name}", verbose=3, gamma=0.95, n_steps=100, ent_coef=0.09, learning_rate=0.005, vf_coef=0.04, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=50, clip_range=0.3, batch_size=200)
# wandb.watch(model.policy.action_net,log='all', log_freq = 1)
# wandb.watch(model.policy.value_net, log='all', log_freq = 1)
train_timesteps = 1000000
model.learn(total_timesteps=train_timesteps)
model.save(f"policy_flatland_{train_timesteps}")

env = flatland_env.env(environment = rail_env, use_renderer = True)
env_name="flatland"
monitor.FILE_PREFIX = env_name
monitor.Monitor._after_step = env_generators._after_step
env = monitor.Monitor(env, experiment_name, force=True)
model = PPO.load(f"policy_flatland_{train_timesteps}")


artifact = wandb.Artifact('model', type='model')
artifact.add_file(f'policy_flatland_{train_timesteps}.zip')
run.log_artifact(artifact)

env.reset(random_seed=seed)
step = 0
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    # act = 2
    env.step(act)
    step+=1
    if step % 100 == 0:
       print(act)
       completion = env_generators.perc_completion(env)
       print("Agents Completed:",completion)
env.close()
completion = env_generators.perc_completion(env)
print("Agents Completed:",completion)


import fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


_video_file = f'*0.mp4'
_found_videos = find(_video_file, experiment_name)
print(_found_videos)
for _found_video in _found_videos:
    wandb.log({_found_video:wandb.Video(_found_video, format="mp4")})
run.join()











# from pettingzoo.test.api_test import api_test
# api_test(env)

# env.reset(random_seed=seed)

# action_dict = dict()
# step = 0
# for agent in env.agent_iter(max_iter=2500):
#     if step == 433:
#         print(step)
#     obs, reward, done, info = env.last()
#     action = 2 # controller.act(0)
#     action_dict.update({agent: action})
#     env.step(action)
#     step += 1
#     if step % 50 == 0:
#         print(step)
#     if step > 400:
#         print(step)
#     # env.render()