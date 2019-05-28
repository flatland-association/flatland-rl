import os
import time
import random

import numpy as np

from flatland.envs.generators import complex_rail_generator
# from flatland.envs.generators import rail_from_list_of_saved_GridTransitionMap_generator
from flatland.envs.generators import random_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

# ensure that every demo run behave constantly equal
random.seed(1)
np.random.seed(1)


class Scenario_Generator:
    @staticmethod
    def generate_random_scenario(number_of_agents=3):
        # Example generate a rail given a manual specification,
        # a map of tuples (cell_type, rotation)
        transition_probability = [15,  # empty cell - Case 0
                                  5,  # Case 1 - straight
                                  5,  # Case 2 - simple switch
                                  1,  # Case 3 - diamond crossing
                                  1,  # Case 4 - single slip
                                  1,  # Case 5 - double slip
                                  1,  # Case 6 - symmetrical
                                  0,  # Case 7 - dead end
                                  1,  # Case 1b (8)  - simple turn right
                                  1,  # Case 1c (9)  - simple turn left
                                  1]  # Case 2b (10) - simple switch mirrored

        # Example generate a random rail

        env = RailEnv(width=20,
                      height=20,
                      rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
                      number_of_agents=number_of_agents)

        return env

    @staticmethod
    def generate_complex_scenario(number_of_agents=3):
        env = RailEnv(width=15,
                      height=15,
                      rail_generator=complex_rail_generator(nr_start_goal=6, nr_extra=30, min_dist=10,
                                                            max_dist=99999, seed=0),
                      number_of_agents=number_of_agents)

        return env

    @staticmethod
    def load_scenario(filename, number_of_agents=3):
        env = RailEnv(width=2 * (1 + number_of_agents),
                      height=1 + number_of_agents)

        """
        env = RailEnv(width=20,
                      height=20,
                      rail_generator=rail_from_list_of_saved_GridTransitionMap_generator(
                          [filename]),
                      number_of_agents=number_of_agents)
        """
        if os.path.exists(filename):
            print("load file: ", filename)
            env.load(filename)
            env.reset(False, False)
        else:
            print("File does not exist:", filename, " Working directory: ", os.getcwd())

        return env


def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_lt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] > val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    max_obs = max(1, max_lt(obs, 1000))
    min_obs = max(0, min_lt(obs, 0))
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    if norm == 0:
        norm = 1.
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


class Demo:

    def __init__(self, env):
        self.env = env
        self.create_renderer()
        self.action_size = 4

    def create_renderer(self):
        self.renderer = RenderTool(self.env, gl="PILSVG")
        handle = self.env.get_agent_handles()
        return handle

    def run_demo(self, max_nbr_of_steps=30):
        action_dict = dict()

        # Reset environment
        _ = self.env.reset(False, False)

        time.sleep(0.0001)  # to satisfy lint...

        for step in range(max_nbr_of_steps):

            # time.sleep(.1)

            # Action
            for iAgent in range(self.env.get_num_agents()):
                action = 2

                if True:
                    if not ((step) % 2 == 0):
                        if iAgent == 3:
                            action = 0

                    if not ((step) % 2 == 0):
                        if iAgent == 4:
                            action = 0

                if False:
                    agent = self.env.agents[iAgent]
                    trial = 0
                    while not self.env.check_action(agent, action)[1]:
                        action = np.random.choice(self.action_size)
                        trial += 1
                        if trial > 10:
                            break
                action_dict.update({iAgent: action})

            self.renderer.renderEnv(show=True, action_dict=action_dict)

            # Environment step
            next_obs, all_rewards, done, _ = self.env.step(action_dict)

            if done['__all__']:
                break
        self.renderer.close_window()


if True:
    demo_000 = Demo(Scenario_Generator.generate_random_scenario())
    demo_000.run_demo()
    demo_000 = None

    demo_001 = Demo(Scenario_Generator.generate_complex_scenario())
    demo_001.run_demo()
    demo_001 = None

    demo_000 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_network_000.pkl'))
    demo_000.run_demo()
    demo_000 = None

    demo_001 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_network_001.pkl'))
    demo_001.run_demo()
    demo_001 = None

    demo_002 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_network_002.pkl'))
    demo_002.run_demo()
    demo_002 = None

    demo_flatland_000 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_flatland_000.pkl'))
    demo_flatland_000.renderer.resize()
    demo_flatland_000.run_demo(300)
    demo_flatland_000 = None

    demo_flatland_000 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_flatland_001.pkl'))
    demo_flatland_000.renderer.resize()
    demo_flatland_000.run_demo(300)
    demo_flatland_000 = None


demo_flatland_000 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_network_003.pkl'))
demo_flatland_000.renderer.resize()
demo_flatland_000.run_demo(1800)
demo_flatland_000 = None
