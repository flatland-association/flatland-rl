import getopt
import sys
import time

import numpy as np

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent instead
class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return 2  # np.random.choice(np.arange(self.action_size))

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


def create_env():
    # Use the new sparse_rail_generator to generate feasible network configurations with corresponding tasks
    # Training on simple small tasks is the best way to get familiar with the environment

    # Use a the malfunction generator to break agents from time to time
    stochastic_data = MalfunctionParameters(malfunction_rate=30,  # Rate of malfunction occurence
                                            min_duration=3,  # Minimal duration of malfunction
                                            max_duration=20  # Max duration of malfunction
                                            )
    # Custom observation builder
    TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
    nAgents = 3
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    env = RailEnv(
        width=20,
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
        obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
    )
    return env


def flatland_3_0_example(sleep_for_animation, do_rendering):
    np.random.seed(1)

    env = create_env()
    env.reset()

    env_renderer = None
    if do_rendering:
        env_renderer = RenderTool(env, gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                                  show_debug=True,
                                  screen_height=1000,
                                  screen_width=1000)

    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    # Set action space to 4 to remove stop action
    agent = RandomAgent(218, 4)

    # Empty dictionary for all agent action
    action_dict = dict()

    print("Start episode...")

    # Reset environment and get initial observations for all agents
    start_reset = time.time()
    obs, info = env.reset()
    end_reset = time.time()
    print(end_reset - start_reset)
    print(env.get_num_agents(), )

    # Reset the rendering sytem
    if env_renderer is not None:
        env_renderer.reset()

    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository

    score = 0
    # Run episode
    frame_step = 0
    for step in range(500):
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(obs[a])
            action_dict.update({a: action})

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        if env_renderer is not None:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        frame_step += 1
        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
            score += all_rewards[a]

        obs = next_obs.copy()
        if done['__all__']:
            break

    if env_renderer is not None:
        env_renderer.close_window()

    print('Episode: Steps {}\t Score = {}'.format(step, score))
    RailEnvPersister.save(env, "saved_episode_2.pkl")


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
    flatland_3_0_example(sleep_for_animation, do_rendering)


if __name__ == '__main__':
    if 'argv' in globals():
        main(argv)
    else:
        main(sys.argv[1:])
