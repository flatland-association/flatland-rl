import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_rail_generator_agents_placer

np.random.seed(1)


# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment
#


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice([1, 2, 3])

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


def test_multi_speed_init():
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999,
                                                        seed=0),
                  agent_generator=complex_rail_generator_agents_placer(),
                  number_of_agents=5)
    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    agent = RandomAgent(218, 4)

    # Empty dictionary for all agent action
    action_dict = dict()

    # Set all the different speeds
    # Reset environment and get initial observations for all agents
    env.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository
    old_pos = []
    for i_agent in range(env.get_num_agents()):
        env.agents[i_agent].speed_data['speed'] = 1. / (i_agent + 1)
        old_pos.append(env.agents[i_agent].position)

    # Run episode
    for step in range(100):

        # Choose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(0)
            action_dict.update({a: action})

            # Check that agent did not move in between its speed updates
            assert old_pos[a] == env.agents[a].position

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether they are done
        _, _, _, _ = env.step(action_dict)

        # Update old position whenever an agent was allowed to move
        for i_agent in range(env.get_num_agents()):
            if (step + 1) % (i_agent + 1) == 0:
                print(step, i_agent, env.agents[i_agent].position)
                old_pos[i_agent] = env.agents[i_agent].position
