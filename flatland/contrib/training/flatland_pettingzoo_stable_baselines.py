
import numpy as np
import os
import PIL
import shutil

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO

import supersuit as ss

from flatland.contrib.interface import flatland_env
from flatland.contrib.utils import env_generators

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

import fnmatch
import wandb

"""
https://github.com/PettingZoo-Team/PettingZoo/blob/HEAD/tutorials/13_lines.py
"""

# Custom observation builder without predictor
# observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor
observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))
seed = 10
np.random.seed(seed)
wandb_log = False
experiment_name = "flatland_pettingzoo"

try:
    if os.path.isdir(experiment_name):
        shutil.rmtree(experiment_name)
    os.mkdir(experiment_name)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

# rail_env = env_generators.sparse_env_small(seed, observation_builder)
rail_env = env_generators.small_v0(seed, observation_builder)

# __sphinx_doc_begin__

env = flatland_env.parallel_env(environment=rail_env, use_renderer=False)
# env = flatland_env.env(environment = rail_env, use_renderer = False)

if wandb_log:
    run = wandb.init(project="flatland2021", entity="nilabha2007", sync_tensorboard=True, 
                     config={}, name=experiment_name, save_code=True)

env_steps = 1000  # 2 * env.width * env.height  # Code uses 1.5 to calculate max_steps
rollout_fragment_length = 50
env = ss.pettingzoo_env_to_vec_env_v0(env)
# env.black_death = True
env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

model = PPO(MlpPolicy, env, tensorboard_log=f"/tmp/{experiment_name}", verbose=3, gamma=0.95, 
    n_steps=rollout_fragment_length, ent_coef=0.01, 
    learning_rate=5e-5, vf_coef=1, max_grad_norm=0.9, gae_lambda=1.0, n_epochs=30, clip_range=0.3,
    batch_size=150, seed=seed)
# wandb.watch(model.policy.action_net,log='all', log_freq = 1)
# wandb.watch(model.policy.value_net, log='all', log_freq = 1)
train_timesteps = 100000
model.learn(total_timesteps=train_timesteps)
model.save(f"policy_flatland_{train_timesteps}")

# __sphinx_doc_end__

model = PPO.load(f"policy_flatland_{train_timesteps}")

env = flatland_env.env(environment=rail_env, use_renderer=True)

if wandb_log:
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(f'policy_flatland_{train_timesteps}.zip')
    run.log_artifact(artifact)


# Model Interference

seed = 100
env.reset(random_seed=seed)
step = 0
ep_no = 0
frame_list = []
while ep_no < 1:
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
        step += 1
        if step % 100 == 0:
            print(f"env step:{step} and action taken:{act}")
            completion = env_generators.perc_completion(env)
            print("Agents Completed:", completion)

    completion = env_generators.perc_completion(env)
    print("Final Agents Completed:", completion)
    ep_no += 1
    frame_list[0].save(f"{experiment_name}{os.sep}pettyzoo_out_{ep_no}.gif", save_all=True, 
                       append_images=frame_list[1:], duration=3, loop=0)
    frame_list = []
    env.close()
    env.reset(random_seed=seed+ep_no)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


if wandb_log:
    extn = "gif"
    _video_file = f'*.{extn}'
    _found_videos = find(_video_file, experiment_name)
    print(_found_videos)
    for _found_video in _found_videos:
        wandb.log({_found_video: wandb.Video(_found_video, format=extn)})
    run.join()
