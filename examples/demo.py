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

class Demo:

    def __init__(self, env):
        self.env = env
        self.create_renderer()
        self.action_size = 4
        self.max_frame_rate = 60
        self.record_frames = None

    def set_record_frames(self,record_frames):
        self.record_frames = record_frames

    def create_renderer(self):
        self.renderer = RenderTool(self.env, gl="PILSVG")
        handle = self.env.get_agent_handles()
        return handle

    def set_max_framerate(self,max_frame_rate):
        self.max_frame_rate = max_frame_rate

    def run_demo(self, max_nbr_of_steps=30):
        action_dict = dict()

        # Reset environment
        _ = self.env.reset(False, False)

        time.sleep(0.0001)  # to satisfy lint...

        for step in range(max_nbr_of_steps):

            # Action
            for iAgent in range(self.env.get_num_agents()):

                # allways walk straight forward
                action = 2

                # update the actions
                action_dict.update({iAgent: action})

            # environment step (apply the actions to all agents)
            next_obs, all_rewards, done, _ = self.env.step(action_dict)

            # render
            self.renderer.renderEnv(show=True,show_observations=False)

            if done['__all__']:
                break



            if self.record_frames is not None:
                self.renderer.gl.saveImage(self.record_frames.format(step))



        self.renderer.close_window()


if False:
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
    demo_flatland_000.run_demo(60)
    demo_flatland_000 = None


    demo_flatland_000 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_network_003.pkl'))
    demo_flatland_000.renderer.resize()
    demo_flatland_000.set_max_framerate(5)
    demo_flatland_000.run_demo(30)
    demo_flatland_000 = None


    demo_flatland_000 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_flatland_001.pkl'))
    demo_flatland_000.renderer.resize()
    demo_flatland_000.set_record_frames('./rendering/frame_{:04d}.bmp')
    demo_flatland_000.run_demo(60)
    demo_flatland_000 = None



demo_001 = Demo(Scenario_Generator.load_scenario('./env-data/railway/temp.pkl'))
demo_001.run_demo(10)
demo_001 = None

if True:
    demo_001 = Demo(Scenario_Generator.load_scenario('./env-data/railway/example_flatland_001.pkl'))
    demo_001.set_record_frames('./rendering/frame_{:04d}.bmp')
    demo_001.run_demo(60)
    demo_001 = None
