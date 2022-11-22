from typing import Union

import numpy as np

from flatland.utils.Timer import Timer
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.rnd_size = 10000000
        self.random_actions = np.random.choice(np.arange(self.action_size), size=self.rnd_size)
        self.rnd_cnt = 0

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        self.rnd_cnt += 1
        return self.random_actions[self.rnd_cnt % self.rnd_size]


def get_rail_env(nAgents=70, use_dummy_obs=False, width=300, height=300):
    # Rail Generator:

    num_cities = 5  # Number of cities to place on the map
    seed = 1  # Random seed
    max_rails_between_cities = 2  # Maximum number of rails connecting 2 cities
    max_rail_pairs_in_cities = 2  # Maximum number of pairs of tracks within a city
    # Even tracks are used as start points, odd tracks are used as endpoints)

    rail_generator = sparse_rail_generator(
        max_num_cities=num_cities,
        seed=seed,
        max_rails_between_cities=max_rails_between_cities,
        max_rail_pairs_in_city=max_rail_pairs_in_cities,
    )

    # Line Generator

    # sparse_line_generator accepts a dictionary which maps speeds to probabilities.
    # Different agent types (trains) with different speeds.
    speed_probability_map = {
        1.: 0.25,  # Fast passenger train
        1. / 2.: 0.25,  # Fast freight train
        1. / 3.: 0.25,  # Slow commuter train
        1. / 4.: 0.25  # Slow freight train
    }

    line_generator = sparse_line_generator(speed_probability_map)

    # Malfunction Generator:

    stochastic_data = MalfunctionParameters(
        malfunction_rate=1 / 10000,  # Rate of malfunction occurence
        min_duration=15,  # Minimal duration of malfunction
        max_duration=50  # Max duration of malfunction
    )

    malfunction_generator = ParamMalfunctionGen(stochastic_data)

    # Observation Builder

    # tree observation returns a tree of possible paths from the current position.
    max_depth = 3  # Max depth of the tree
    predictor = ShortestPathPredictorForRailEnv(
        max_depth=50)  # (Specific to Tree Observation - read code)

    observation_builder = TreeObsForRailEnv(
        max_depth=max_depth,
        predictor=predictor
    )

    if use_dummy_obs:
        observation_builder = DummyObservationBuilder()

    number_of_agents = nAgents  # Number of trains to create
    seed = 1  # Random seed

    env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=number_of_agents,
        random_seed=seed,
        obs_builder_object=observation_builder,
        malfunction_generator=malfunction_generator
    )
    return env


def run_simulation(env_fast: RailEnv, do_rendering):
    agent = RandomAgent(action_size=5)
    max_steps = 200

    env_renderer = None
    if do_rendering:
        env_renderer = RenderTool(env_fast,
                                  gl="PGL",
                                  show_debug=True,
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS)
        env_renderer.set_new_rail()
        env_renderer.reset()

    action_dict = {}
    for step in range(max_steps):

        # Chose an action for each agent in the environment
        for handle in range(env_fast.get_num_agents()):
            action = agent.act(handle)
            action_dict.update({handle: action})

        next_obs, all_rewards, done, _ = env_fast.step(action_dict)
        if env_renderer is not None:
            env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=True,
                show_predictions=False
            )

    if env_renderer is not None:
        env_renderer.close_window()


USE_TIME_PROFILER = True

RUN_SIMULATION = True
DO_RENDERING = False

USE_DUMMY_OBS = True

N_AGENTS = 200
WIDTH = 100
HEIGHT = 100


def start_timer() -> Union[Timer, None]:
    if USE_TIME_PROFILER:
        time_profiler = Timer()
        time_profiler.start()
        return time_profiler
    return None


def end_timer(label: str, time_profiler: Timer):
    if time_profiler is None:
        return
    print('{:>20} \t {:7.5f}ms'.format(label, time_profiler.end()))


def execute_standard_flatland_application():
    time_profiler = start_timer()
    env_fast = get_rail_env(nAgents=N_AGENTS, use_dummy_obs=USE_DUMMY_OBS, width=WIDTH, height=HEIGHT)
    end_timer('Create env', time_profiler)

    time_profiler = start_timer()
    env_fast.reset(random_seed=1)
    end_timer('Reset env', time_profiler)

    time_profiler = start_timer()
    action_dict = {agent.handle: 0 for agent in env_fast.agents}
    end_timer('Build actions', time_profiler)

    time_profiler = start_timer()
    for i in range(1):
        env_fast.step(action_dict)
    end_timer('Step env', time_profiler)

    time_profiler = start_timer()
    obs = env_fast._get_observations()
    end_timer('get observations', time_profiler)

    if RUN_SIMULATION:
        time_profiler = start_timer()
        run_simulation(env_fast, DO_RENDERING)
        end_timer('run simulation', time_profiler)


if __name__ == "__main__":
    print("Start ...")
    execute_standard_flatland_application()
    print("... end.")
