import numpy as np

from flatland.envs.generators import sparse_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

np.random.seed(1)

# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment

TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
env = RailEnv(width=50,
              height=50,
              rail_generator=sparse_rail_generator(num_cities=10,  # Number of cities in map
                                                   num_intersections=3,  # Number of interesections in map
                                                   num_trainstations=40,  # Number of possible start/targets on map
                                                   min_node_dist=10,  # Minimal distance of nodes
                                                   node_radius=2,  # Proximity of stations to city center
                                                   num_neighb=4,  # Number of connections to other cities
                                                   seed=15,  # Random seed
                                                   ),
              number_of_agents=35,
              obs_builder_object=TreeObservation)

env_renderer = RenderTool(env, gl="PILSVG", )


# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent here


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice(np.arange(self.action_size))

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
agent = RandomAgent(218, 4)
n_trials = 5

# Empty dictionary for all agent action
action_dict = dict()
print("Starting Training...")

for trials in range(1, n_trials + 1):

    # Reset environment and get initial observations for all agents
    obs = env.reset()
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        speed = (idx % 4) + 1
        tmp_agent.speed_data["speed"] = 1 / speed
    env_renderer.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository

    score = 0
    # Run episode
    for step in range(500):
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(obs[a])
            action_dict.update({a: action})

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
            score += all_rewards[a]

        obs = next_obs.copy()
        if done['__all__']:
            break
    print('Episode Nr. {}\t Score = {}'.format(trials, score))
