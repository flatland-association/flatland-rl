import getopt
import random
import sys
import time
from typing import Optional, List, Dict

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.misc import str2bool
from flatland.utils.ordered_set import OrderedSet
from flatland.utils.rendertools import RenderTool

random.seed(100)
np.random.seed(100)


class ObservePredictions(ObservationBuilder):
    """
    We use the provided ShortestPathPredictor to illustrate the usage of predictors in your custom observation.
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        '''
        Because we do not want to call the predictor seperately for every agent we implement the get_many function
        Here we can call the predictor just ones for all the agents and use the predictions to generate our observations
        :param handles:
        :return:
        '''

        self.predictions = self.predictor.get()

        self.predicted_pos = {}

        if handles is None:
            handles = []

        for t in range(len(self.predictions[0])):
            pos_list = []
            for a in handles:
                pos_list.append(self.predictions[a][t][1:3])
            # We transform (x,y) coodrinates to a single integer number for simpler comparison
            self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> np.ndarray:
        '''
        Lets write a simple observation which just indicates whether or not the own predicted path
        overlaps with other predicted paths at any time. This is useless for the task of navigation but might
        help when looking for conflicts. A more complex implementation can be found in the TreeObsForRailEnv class

        Each agent recieves an observation of length 10, where each element represents a prediction step and its value
        is:
         - 0 if no overlap is happening
         - 1 where n i the number of other paths crossing the predicted cell

        :param handle: handeled as an index of an agent
        :return: Observation of handle
        '''

        observation = np.zeros(10)

        # We are going to track what cells where considered while building the obervation and make them accesible
        # For rendering

        visited = OrderedSet()
        for _idx in range(10):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))
            if self.predicted_pos[_idx][handle] in np.delete(self.predicted_pos[_idx], handle, 0):
                # We detect if another agent is predicting to pass through the same cell at the same predicted time
                observation[handle] = 1

        # This variable will be access by the renderer to visualize the observation
        self.env.dev_obs_dict[handle] = visited

        return observation

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)


def create_env(custom_obs_builder):
    nAgents = 3
    n_cities = 2
    max_rails_between_cities = 4
    max_rails_in_city = 2
    seed = 0
    env = RailEnv(
        width=30,
        height=30,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=nAgents,
        obs_builder_object=custom_obs_builder
    )
    return env


def custom_observation_example_03_ObservePredictions(sleep_for_animation, do_rendering):
    # Initiate the Predictor
    custom_predictor = ShortestPathPredictorForRailEnv(10)

    # Pass the Predictor to the observation builder
    custom_obs_builder = ObservePredictions(custom_predictor)

    # Initiate Environment
    env = create_env(custom_obs_builder)
    obs, info = env.reset()

    env_renderer = None
    if do_rendering:
        env_renderer = RenderTool(env)
        # We render the initial step and show the obsered cells as colored boxes
        env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)

    action_dict = {}
    for step in range(100):
        for a in range(env.get_num_agents()):
            action = np.random.randint(0, 5)
            action_dict[a] = action
        obs, all_rewards, done, _ = env.step(action_dict)
        print("Rewards: ", all_rewards, "  [done=", done, "]")
        if env_renderer is not None:
            env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)
        if sleep_for_animation:
            time.sleep(0.5)

        if done["__all__"]:
            print("All done!")
            break

    if env_renderer is not None:
        env_renderer.close_window()


def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", "do_rendering=", ""])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    sleep_for_animation = True
    do_rendering = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        elif o in ("--do_rendering"):
            do_rendering = str2bool(a)
        else:
            assert False, "unhandled option"

    # execute example
    custom_observation_example_03_ObservePredictions(sleep_for_animation, do_rendering)

if __name__ == '__main__':
    if 'argv' in globals():
        main(argv)
    else:
        main(sys.argv[1:])
