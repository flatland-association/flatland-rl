# import torch
import random
import time
# from flatland.baselines.dueling_double_dqn import Agent
from collections import deque

import numpy as np

from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


class Player(object):
    def __init__(self, env):
        self.env = env
        self.handle = env.get_agent_handles()

        self.state_size = 105
        self.action_size = 4
        self.n_trials = 9999
        self.eps = 1.
        self.eps_end = 0.005
        self.eps_decay = 0.998
        self.action_dict = dict()
        self.scores_window = deque(maxlen=100)
        self.done_window = deque(maxlen=100)
        self.scores = []
        self.dones_list = []
        self.action_prob = [0] * 4

        # Removing refs to a real agent for now.
        # self.agent = Agent(self.state_size, self.action_size, "FC", 0)
        # self.agent.qnetwork_local.load_state_dict(torch.load('../flatland/baselines/Nets/avoid_checkpoint9900.pth'))
        # self.agent.qnetwork_local.load_state_dict(torch.load(
        #    '../flatland/flatland/baselines/Nets/avoid_checkpoint15000.pth'))

        self.iFrame = 0
        self.tStart = time.time()

        # Reset environment
        # self.obs = self.env.reset()
        self.env.obs_builder.reset()
        self.obs = self.env._get_observations()
        for envAgent in range(self.env.get_num_agents()):
            norm = max(1, max_lt(self.obs[envAgent], np.inf))
            self.obs[envAgent] = np.clip(np.array(self.obs[envAgent]) / norm, -1, 1)

        # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)

        self.score = 0
        self.env_done = 0

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self):
        env = self.env

        # Pass the (stored) observation to the agent network and retrieve the action
        for handle in env.get_agent_handles():
            # Real Agent
            # action = self.agent.act(np.array(self.obs[handle]), eps=self.eps)
            # Random actions
            # action = random.randint(0, 3)
            action = np.random.choice([0, 1, 2, 3], 1, p=[0.2, 0.1, 0.6, 0.1])[0]
            # Numpy version uses single random sequence
            # action = np.random.randint(0, 4, size=1)
            self.action_prob[action] += 1
            self.action_dict.update({handle: action})

        # Environment step - pass the agent actions to the environment,
        # retrieve the response - observations, rewards, dones
        next_obs, all_rewards, done, _ = self.env.step(self.action_dict)

        for handle in env.get_agent_handles():
            norm = max(1, max_lt(next_obs[handle], np.inf))
            next_obs[handle] = np.clip(np.array(next_obs[handle]) / norm, -1, 1)

        # Update replay buffer and train agent
        if False:
            for handle in self.env.get_agent_handles():
                self.agent.step(self.obs[handle], self.action_dict[handle],
                                all_rewards[handle], next_obs[handle], done[handle],
                                train=False)
                self.score += all_rewards[handle]

        self.iFrame += 1

        self.obs = next_obs.copy()
        if done['__all__']:
            self.env_done = 1


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


def main(render=True, delay=0.0, n_trials=3, n_steps=50, sGL="PILSVG"):
    random.seed(1)
    np.random.seed(1)

    # Example generate a random rail
    env = RailEnv(width=15, height=15,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12),
                  number_of_agents=5)

    if render:
        env_renderer = RenderTool(env, gl=sGL, show=True)

    oPlayer = Player(env)

    for trials in range(1, n_trials + 1):

        # Reset environment
        oPlayer.reset()
        env_renderer.set_new_rail()

        # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)

        # score = 0
        # env_done = 0

        # Run episode
        for step in range(n_steps):
            oPlayer.step()
            if render:
                env_renderer.renderEnv(show=True, frames=True, iEpisode=trials, iStep=step)
                # time.sleep(10)
                if delay > 0:
                    time.sleep(delay)

    env_renderer.gl.close_window()


if __name__ == "__main__":
    main(render=True, delay=0)
