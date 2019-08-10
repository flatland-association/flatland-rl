import numpy as np

from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv

np.random.seed(1)

# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment
#

env = RailEnv(width=50,
              height=50,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999, seed=0),
              number_of_agents=5)


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


def test_multi_speed_init():
    # Reset environment and get initial observations for all agents
    obs = env.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository
    for i_agent in range(env.get_num_agents()):
        env.agents[i_agent].speed_data['speed'] = 1. / np.random.randint(1, 10)
    score = 0
    # Run episode
    for step in range(100):
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(obs[a])
            action_dict.update({a: action})

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
            score += all_rewards[a]

        obs = next_obs.copy()
        if done['__all__']:
            break
