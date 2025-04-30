"""Uses Stable-Baselines3 to train agents to play the Flatland environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Based on https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/SB3/waterworld/sb3_waterworld_vector.py
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenedNormalizedTreeObsForRailEnv
from flatland.ml.pettingzoo.wrappers import PettingzooFlatland


def train_flatland_pettingzoo_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn.parallel_env(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval_flatland_pettingzoo(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env: ParallelEnv = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        obs, _ = env.reset(seed=i)

        done = False
        while not done:
            act = {a: int(model.predict(obs[a], deterministic=True)[0]) for a in env.agents}
            obs, rew, terminations, truncations, infos = env.step(act)
            for a in env.agents:
                rewards[a] += rew[a]
            done = all(terminations.values())

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    raw_env, _, _ = env_generator(obs_builder_object=FlattenedNormalizedTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    env_fn = PettingzooFlatland(raw_env)
    env_kwargs = {}

    # Train a model
    train_flatland_pettingzoo_supersuit(env_fn, steps=196_608, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval_flatland_pettingzoo(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval_flatland_pettingzoo(env_fn, num_games=2, render_mode="human", **env_kwargs)
