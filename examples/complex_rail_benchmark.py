"""Run benchmarks on complex rail flatland."""
import random

import numpy as np

from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv


def run_benchmark():
    """Run benchmark on a small number of agents in complex rail environment."""
    random.seed(1)
    np.random.seed(1)

    # Example generate a random rail
    env = RailEnv(width=15, height=15,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12),
                  number_of_agents=5)

    n_trials = 20
    action_dict = dict()
    action_prob = [0] * 4

    def max_lt(seq, val):
        """
        Return greatest item in seq for which item < val applies.
        None is returned if seq was empty or all items in seq were >= val.
        """

        idx = len(seq) - 1
        while idx >= 0:
            if seq[idx] < val and seq[idx] >= 0:
                return seq[idx]
            idx -= 1
        return None

    for trials in range(1, n_trials + 1):

        # Reset environment
        obs = env.reset()

        for a in range(env.get_num_agents()):
            norm = max(1, max_lt(obs[a], np.inf))
            obs[a] = np.clip(np.array(obs[a]) / norm, -1, 1)

        # Run episode
        for step in range(100):
            # Action
            for a in range(env.get_num_agents()):
                action = np.random.randint(0, 4)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)
            for a in range(env.get_num_agents()):
                norm = max(1, max_lt(next_obs[a], np.inf))
                next_obs[a] = np.clip(np.array(next_obs[a]) / norm, -1, 1)

            if done['__all__']:
                break
        if trials % 100 == 0:
            action_prob = [1] * 4


if __name__ == "__main__":
    run_benchmark()
