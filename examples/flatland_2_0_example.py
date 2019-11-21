import time

import numpy as np

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

np.random.seed(1)

# Use the new sparse_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment

# Use a the malfunction generator to break agents from time to time
stochastic_data = MalfunctionParameters(malfunction_rate=30,  # Rate of malfunction occurence
                                        min_duration=3,  # Minimal duration of malfunction
                                        max_duration=20  # Max duration of malfunction
                                        )
# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

env = RailEnv(width=100,
              height=100,
              rail_generator=sparse_rail_generator(max_num_cities=30,
                                                   # Number of cities in map (where train stations are)
                                                   seed=14,  # Random seed
                                                   grid_mode=False,
                                                   max_rails_between_cities=2,
                                                   max_rails_in_city=8,
                                                   ),
              schedule_generator=sparse_schedule_generator(speed_ration_map),
              number_of_agents=100,
              malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
              # Malfunction data generator
              obs_builder_object=GlobalObsForRailEnv(),
              remove_agents_at_target=True,
              record_steps=True
              )

# RailEnv.DEPOT_POSITION = lambda agent, agent_handle : (agent_handle % env.height,0)

env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          screen_height=1000,
                          screen_width=1000)


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
    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    frame_step += 1
    # Update replay buffer and train agent
    for a in range(env.get_num_agents()):
        agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        score += all_rewards[a]

    obs = next_obs.copy()
    if done['__all__']:
        break

print('Episode: Steps {}\t Score = {}'.format(step, score))
env.save_episode("saved_episode_2.mpk")
