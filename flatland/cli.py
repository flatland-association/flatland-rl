# -*- coding: utf-8 -*-

"""Console script for flatland."""
import sys
import click
import numpy as np
import time
from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.evaluators.service import FlatlandRemoteEvaluationService
import redis


@click.command()
def demo(args=None):
    """Demo script to check installation"""
    env = RailEnv(
            width=15,
            height=15,
            rail_generator=complex_rail_generator(
                                    nr_start_goal=10,
                                    nr_extra=1,
                                    min_dist=8,
                                    max_dist=99999),
            number_of_agents=5)
    
    env._max_episode_steps = int(15 * (env.width + env.height))
    env_renderer = RenderTool(env)

    while True:
        obs = env.reset()
        _done = False
        # Run a single episode here
        step = 0
        while not _done:
            # Compute Action
            _action = {}
            for _idx, _ in enumerate(env.agents):
                _action[_idx] = np.random.randint(0, 5)
            obs, all_rewards, done, _ = env.step(_action)
            _done = done['__all__']
            step += 1
            env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=False,
                show_predictions=False
            )
            time.sleep(0.3)
    return 0


@click.command()
@click.option('--tests', 
              type=click.Path(exists=True),
              help="Path to folder containing Flatland tests",
              required=True
              )
@click.option('--service_id', 
              default="FLATLAND_RL_SERVICE_ID",
              help="Evaluation Service ID. This has to match the service id on the client.",
              required=False
              )
def evaluator(tests, service_id):
    try:
        redis_connection = redis.Redis()
        redis_connection.ping()
    except redis.exceptions.ConnectionError as e:
        raise Exception(
            "\nRedis server does not seem to be running on your localhost.\n"
            "Please ensure that you have a redis server running on your localhost"
            )
    
    grader = FlatlandRemoteEvaluationService(
                test_env_folder=tests,
                flatland_rl_service_id=service_id,
                visualize=False,
                verbose=False
                )
    grader.run()


if __name__ == "__main__":
    sys.exit(demo())  # pragma: no cover
