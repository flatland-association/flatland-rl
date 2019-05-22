import random
import time
from collections import deque

import numpy as np
from benchmarker import Benchmarker

from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


def main(render=True, delay=0.0):
    random.seed(1)
    np.random.seed(1)

    # Example generate a random rail
    env = RailEnv(width=15, height=15,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12),
                  number_of_agents=5)

    if render:
        env_renderer = RenderTool(env, gl="QTSVG")

    n_trials = 20
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998
    action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
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

    iFrame = 0
    tStart = time.time()
    for trials in range(1, n_trials + 1):

        # Reset environment
        obs = env.reset()
        if render:
            env_renderer.set_new_rail()

        for a in range(env.get_num_agents()):
            norm = max(1, max_lt(obs[a], np.inf))
            obs[a] = np.clip(np.array(obs[a]) / norm, -1, 1)

        # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)

        score = 0
        env_done = 0

        # Run episode
        for step in range(100):
            # if trials > 114:
            # env_renderer.renderEnv(show=True)
            # print(step)
            # Action
            for a in range(env.get_num_agents()):
                action = np.random.randint(0, 4)
                action_prob[action] += 1
                action_dict.update({a: action})

            if render:
                env_renderer.renderEnv(show=True, frames=True, iEpisode=trials, iStep=step, action_dict=action_dict)
                if delay > 0:
                    time.sleep(delay)

            iFrame += 1

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)
            for a in range(env.get_num_agents()):
                norm = max(1, max_lt(next_obs[a], np.inf))
                next_obs[a] = np.clip(np.array(next_obs[a]) / norm, -1, 1)
            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                # agent.step(obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a])
                score += all_rewards[a]

            obs = next_obs.copy()
            if done['__all__']:
                env_done = 1
                break
        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        done_window.append(env_done)
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(('\rTraining {} Agents.\tEpisode {}\tAverage Score: {:.0f}\tDones: {:.2f}%' +
               '\tEpsilon: {:.2f} \t Action Probabilities: \t {}').format(
            env.get_num_agents(),
            trials,
            np.mean(scores_window),
            100 * np.mean(done_window),
            eps, action_prob / np.sum(action_prob)),
            end=" ")
        if trials % 100 == 0:
            tNow = time.time()
            rFps = iFrame / (tNow - tStart)
            print(('\rTraining {} Agents.\tEpisode {}\tAverage Score: {:.0f}\tDones: {:.2f}%' +
                   '\tEpsilon: {:.2f} fps: {:.2f} \t Action Probabilities: \t {}').format(
                env.get_num_agents(),
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, rFps, action_prob / np.sum(action_prob)))
            action_prob = [1] * 4


if __name__ == "__main__":
    with Benchmarker(cycle=20, extra=1) as bench:
        @bench("Everything")
        def _(bm):
            main(render=False, delay=0)
