import pytest

from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenedNormalizedTreeObsForRailEnv
from flatland.ml.pettingzoo.examples.flatland_pettingzoo_stable_baselines import train_flatland_pettingzoo_supersuit, eval_flatland_pettingzoo
from flatland.ml.pettingzoo.wrappers import PettingzooFlatland


@pytest.mark.slow
def test_train_and_eval_():
    raw_env, _, _ = env_generator(obs_builder_object=FlattenedNormalizedTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    env_fn = PettingzooFlatland(raw_env)
    env_kwargs = {}

    train_flatland_pettingzoo_supersuit(env_fn, steps=1, seed=0, **env_kwargs)

    eval_flatland_pettingzoo(env_fn, num_games=1, render_mode=None, **env_kwargs)
