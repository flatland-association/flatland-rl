"""
Collection of environment-specific PredictionBuilder.
"""

import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.step_utils.states import TrainState
from flatland.envs.step_utils import transition_utils


class DummyPredictorForRailEnv(PredictionBuilder):
    """
    DummyPredictorForRailEnv object.

    This object returns predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def get(self, handle: int = None):
        """
        Called whenever get_many in the observation build is called.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here
            The prediction at 0 is the current position, direction etc.

        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]

        prediction_dict = {}

        for agent in agents:
            if not agent.state.is_on_map_state():
                # TODO make this generic
                continue
            action_priorities = [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]
            agent_virtual_position = agent.position
            agent_virtual_direction = agent.direction
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]
            for index in range(1, self.max_depth + 1):
                action_done = False
                # if we're at the target, stop moving...
                if agent.position == agent.target:
                    prediction[index] = [index, *agent.target, agent.direction, RailEnvActions.STOP_MOVING]

                    continue
                for action in action_priorities:
                    new_cell_isValid, new_direction, new_position, transition_isValid = \
                        transition_utils.check_action_on_agent(action, self.env.rail, agent.position, agent.direction)
                    if all([new_cell_isValid, transition_isValid]):
                        # move and change direction to face the new_direction that was
                        # performed
                        agent.position = new_position
                        agent.direction = new_direction
                        prediction[index] = [index, *new_position, new_direction, action]
                        action_done = True
                        break
                if not action_done:
                    raise Exception("Cannot move further. Something is wrong")
            prediction_dict[agent.handle] = prediction
            agent.position = agent_virtual_position
            agent.direction = agent_virtual_direction
        return prediction_dict


class ShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    ShortestPathPredictorForRailEnv object.

    This object returns shortest-path predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def __init__(self, max_depth: int = 20):
        super().__init__(max_depth)

    def get(self, handle: int = None):
        """
        Called whenever get_many in the observation build is called.
        Requires distance_map to extract the shortest path.
        Does not take into account future positions of other agents!

        If there is no shortest path, the agent just stands still and stops moving.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here (not implemented yet)
            The prediction at 0 is the current position, direction etc.
        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]
        distance_map: DistanceMap = self.env.distance_map

        shortest_paths = get_shortest_paths(distance_map, max_depth=self.max_depth)

        prediction_dict = {}
        for agent in agents:
            if agent.state.is_off_map_state():
                agent_virtual_position = agent.initial_position
            elif agent.state.is_on_map_state():
                agent_virtual_position = agent.position
            elif agent.state == TrainState.DONE:
                agent_virtual_position = agent.target
            else:

                prediction = np.zeros(shape=(self.max_depth + 1, 5))
                for i in range(self.max_depth):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_counter.speed
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # if there is a shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(1, self.max_depth + 1):
                # if we're at the target, stop moving until max_depth is reached
                if new_position == agent.target or not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction

                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.add((*new_position, new_direction))

            # TODO: very bady side effects for visualization only: hand the dev_pred_dict back instead of setting on env!
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict
