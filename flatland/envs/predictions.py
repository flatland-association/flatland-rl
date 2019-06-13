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

    def get(self, distancemap, handle=None):
        """
        Called whenever predict is called on the environment.

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
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, _agent_initial_position[0], _agent_initial_position[1], _agent_initial_direction, 0]
            for index in range(1, self.max_depth + 1):
                action_done = False
                # if we're at the target, stop moving...
                if agent.position == agent.target:
                    prediction[index] = [index, agent.target[0], agent.target[1], agent.direction,
                                         RailEnvActions.STOP_MOVING]

                    continue
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
                    raise Exception("Cannot move further. Something is wrong")
            prediction_dict[agent.handle] = prediction
            agent.position = _agent_initial_position
            agent.direction = _agent_initial_direction
        return prediction_dict


class ShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    DummyPredictorForRailEnv object.

    This object returns predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def get(self, distancemap, handle=None):
        """
        Called whenever predict is called on the environment.

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
        agent_idx = 0
        for agent in agents:
            action_priorities = [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]
            _agent_initial_position = agent.position
            _agent_initial_direction = agent.direction
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, _agent_initial_position[0], _agent_initial_position[1], _agent_initial_direction, 0]
            for index in range(1, self.max_depth + 1):
                action_done = False
                # if we're at the target, stop moving...
                if agent.position == agent.target:
                    prediction[index] = [index, agent.target[0], agent.target[1], agent.direction,
                                         RailEnvActions.STOP_MOVING]

                    continue
                # Take shortest possible path
                cell_transitions = self.env.rail.get_transitions((*agent.position, agent.direction))

                if np.sum(cell_transitions) == 1:
                    new_direction = np.argmax(cell_transitions)
                    new_position = self._new_position(agent.position, new_direction)
                elif np.sum(cell_transitions) > 1:
                    min_dist = np.inf
                    for direct in range(4):
                        if cell_transitions[direct] == 1:
                            target_dist = distancemap[agent_idx, agent.position[0], agent.position[1], direct]
                            if target_dist < min_dist:
                                min_dist = target_dist
                                new_direction = direct
                    new_position = self._new_position(agent.position, new_direction)

                agent.position = new_position
                agent.direction = new_direction
                prediction[index] = [index, new_position[0], new_position[1], new_direction, 0]
                action_done = True
                if not action_done:
                    raise Exception("Cannot move further. Something is wrong")
            prediction_dict[agent.handle] = prediction
            agent.position = _agent_initial_position
            agent.direction = _agent_initial_direction
            agent_idx += 1

        return prediction_dict

    def _new_position(self, position, movement):
        """
        Utility function that converts a compass movement over a 2D grid to new positions (r, c).
        """
        if movement == 0:  # NORTH
            return (position[0] - 1, position[1])
        elif movement == 1:  # EAST
            return (position[0], position[1] + 1)
        elif movement == 2:  # SOUTH
            return (position[0] + 1, position[1])
        elif movement == 3:  # WEST
            return (position[0], position[1] - 1)
