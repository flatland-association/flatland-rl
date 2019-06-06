"""
Collection of environment-specific PredictionBuilder.
"""

import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.rail_env import RailEnvActions


class DummyPredictorForRailEnv(PredictionBuilder):
    """
    DummyPredictorForRailEnv object.

    This object returns predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def get(self, handle=None):
        """
        Called whenever step_prediction is called on the environment.

        Parameters
        -------
        handle : int (optional)
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            Returns a dictionary index by the agent handle and for each agent a vector of 5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here
        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]

        prediction_dict = {}

        for agent in agents:
            action_priorities = [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]
            _agent_initial_position = agent.position
            _agent_initial_direction = agent.direction
            prediction = np.zeros(shape=(self.max_depth, 5))
            prediction[0] = [0, _agent_initial_position[0], _agent_initial_position[1], _agent_initial_direction, 0]
            for index in range(1, self.max_depth):
                action_done = False
                for action in action_priorities:
                    cell_isFree, new_cell_isValid, new_direction, new_position, transition_isValid = \
                        self.env._check_action_on_agent(action, agent)
                    if all([new_cell_isValid, transition_isValid]):
                        # move and change direction to face the new_direction that was
                        # performed
                        agent.position = new_position
                        agent.direction = new_direction
                        prediction[index] = [index, new_position[0], new_position[1], new_direction, action]
                        action_done = True
                        break
                if not action_done:
                    print("Cannot move further.")
            prediction_dict[agent.handle] = prediction
            agent.position = _agent_initial_position
            agent.direction = _agent_initial_direction
        return prediction_dict
