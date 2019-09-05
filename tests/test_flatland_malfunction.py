import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator


class SingleAgentNavigationObs(TreeObsForRailEnv):
    """
    We derive our bbservation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
    the minimum distances from each grid node to each agent's target.

    We then build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__(max_depth=0)
        self.observation_space = [3]

    def reset(self):
        # Recompute the distance map, if the environment has changed.
        super().reset()

    def get(self, handle):
        agent = self.env.agents[handle]

        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
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
                    new_position = self._new_position(agent.position, direction)
                    min_distances.append(self.distance_map[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def test_malfunction_process():
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 1000,
                       'min_duration': 3,
                       'max_duration': 3}
    np.random.seed(5)

    env = RailEnv(width=20,
                  height=20,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
                                                        seed=0),
                  schedule_generator=complex_schedule_generator(),
                  number_of_agents=2,
                  obs_builder_object=SingleAgentNavigationObs(),
                  stochastic_data=stochastic_data)

    obs = env.reset()

    # Check that a initial duration for malfunction was assigned
    assert env.agents[0].malfunction_data['next_malfunction'] > 0

    agent_halts = 0
    total_down_time = 0
    agent_malfunctioning = False
    agent_old_position = env.agents[0].position
    for step in range(100):
        actions = {}
        for i in range(len(obs)):
            actions[i] = np.argmax(obs[i]) + 1

        if step % 5 == 0:
            # Stop the agent and set it to be malfunctioning
            env.agents[0].malfunction_data['malfunction'] = -1
            env.agents[0].malfunction_data['next_malfunction'] = 0
            agent_halts += 1

        obs, all_rewards, done, _ = env.step(actions)

        if env.agents[0].malfunction_data['malfunction'] > 0:
            agent_malfunctioning = True
        else:
            agent_malfunctioning = False

        if agent_malfunctioning:
            # Check that agent is not moving while malfunctioning
            assert agent_old_position == env.agents[0].position

        agent_old_position = env.agents[0].position
        total_down_time += env.agents[0].malfunction_data['malfunction']

    # Check that the appropriate number of malfunctions is achieved
    assert env.agents[0].malfunction_data['nr_malfunctions'] == 21

    # Check that 20 stops where performed
    assert agent_halts == 20

    # Check that malfunctioning data was standing around
    assert total_down_time > 0


def test_malfunction_process_statistically():
    """Tests hat malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 2,
                       'min_duration': 3,
                       'max_duration': 3}
    np.random.seed(5)

    env = RailEnv(width=20,
                  height=20,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
                                                        seed=0),
                  schedule_generator=complex_schedule_generator(),
                  number_of_agents=2,
                  obs_builder_object=SingleAgentNavigationObs(),
                  stochastic_data=stochastic_data)

    env.reset()
    nb_malfunction = 0
    for step in range(100):
        action_dict = {}
        for agent in env.agents:
            if agent.malfunction_data['malfunction'] > 0:
                nb_malfunction += 1
            # We randomly select an action
            action_dict[agent.handle] = np.random.randint(4)

        env.step(action_dict)

    # check that generation of malfunctions works as expected
    # results are different in py36 and py37, therefore no exact test on nb_malfunction
    assert nb_malfunction > 150
