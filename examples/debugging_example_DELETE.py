import random
import time

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

random.seed(1)
np.random.seed(1)


class SingleAgentNavigationObs(TreeObsForRailEnv):
    """
    We derive our bbservation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
    the minimum distances from each grid node to each agent's target.

    We then build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__(max_depth=0)
        self.observation_space = [3]

    def reset(self):
        # Recompute the distance map, if the environment has changed.
        super().reset()

    def get(self, handle):
        agent = self.env.agents[handle]

        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = self._new_position(agent.position, direction)
                    min_distances.append(self.distance_map[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


env = RailEnv(width=14,
              height=14,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999, seed=0),
              schedule_generator=complex_schedule_generator(),
              number_of_agents=2,
              obs_builder_object=SingleAgentNavigationObs())

obs = env.reset()
env_renderer = RenderTool(env, gl="PILSVG")
env_renderer.render_env(show=True, frames=True, show_observations=False)
for step in range(100):
    actions = {}
    for i in range(len(obs)):
        actions[i] = np.argmax(obs[i]) + 1

    if step % 5 == 0:
        print("Agent halts")
        actions[0] = 4  # Halt

    obs, all_rewards, done, _ = env.step(actions)
    if env.agents[0].malfunction_data['malfunction'] > 0:
        print("Agent 0 broken-ness: ", env.agents[0].malfunction_data['malfunction'])

    env_renderer.render_env(show=True, frames=True, show_observations=False)
    time.sleep(0.5)
    if done["__all__"]:
        break
env_renderer.close_window()
