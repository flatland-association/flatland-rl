import getopt
import random
import sys
import time
from typing import List

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool

random.seed(100)
np.random.seed(100)


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

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
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", ""])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    sleep_for_animation = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        else:
            assert False, "unhandled option"

    env = RailEnv(width=7, height=7,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
                                                        seed=1), schedule_generator=complex_schedule_generator(),
                  number_of_agents=1, obs_builder_object=SingleAgentNavigationObs())

    obs, info = env.reset()
    env_renderer = RenderTool(env)
    env_renderer.render_env(show=True, frames=True, show_observations=True)
    for step in range(100):
        action = np.argmax(obs[0]) + 1
        obs, all_rewards, done, _ = env.step({0: action})
        print("Rewards: ", all_rewards, "  [done=", done, "]")
        env_renderer.render_env(show=True, frames=True, show_observations=True)
        if sleep_for_animation:
            time.sleep(0.1)
        if done["__all__"]:
            break
    env_renderer.close_window()


if __name__ == '__main__':
    if 'argv' in globals():
        main(argv)
    else:
        main(sys.argv[1:])
