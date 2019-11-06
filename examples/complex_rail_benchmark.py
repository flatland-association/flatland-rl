"""Run benchmarks on complex rail flatland."""
import random

import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator


def run_benchmark():
    """Run benchmark on a small number of agents in complex rail environment."""
    random.seed(1)
    np.random.seed(1)

    # Example generate a random rail
    env = RailEnv(width=15, height=15, rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12),
                  schedule_generator=complex_schedule_generator(), number_of_agents=5)
    env.reset()

    n_trials = 20
    action_dict = dict()
    action_prob = [0] * 4

    for trials in range(1, n_trials + 1):

        # Reset environment
        obs, info = env.reset()

        # Run episode
        for step in range(100):
            # Action
            for a in range(env.get_num_agents()):
                action = np.random.randint(0, 4)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            if done['__all__']:
                break
        if trials % 100 == 0:
            action_prob = [1] * 4


if __name__ == "__main__":
    run_benchmark()
