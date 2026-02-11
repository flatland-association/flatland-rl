import math
from typing import Dict, List, Optional, Generic, TypeVar

import numpy as np

from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map_walker import DistanceMapWalker
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState

UnderlyingTransitionMapType = TypeVar('UnderlyingTransitionMapType', bound=TransitionMap)
UnderlyingDistanceMapType = TypeVar('UnderlyingDistanceMapType')
UnderlyingConfigurationType = TypeVar('UnderlyingConfigurationType')


class AbstractDistanceMap(Generic[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType]):
    def __init__(self, agents: List[EnvAgent]):
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None

    def set(self, distance_map: np.ndarray):
        """
        Set the distance map
        """
        self.distance_map = distance_map

    def get(self) -> np.ndarray:
        """
        Get the distance map
        """
        if self.reset_was_called:
            self.reset_was_called = False

            compute_distance_map = True
            # Don't compute the distance map if it was loaded
            if self.agents_previous_computation is None and self.distance_map is not None:
                compute_distance_map = False

            if compute_distance_map:
                self._compute(self.agents, self.rail)

        elif self.distance_map is None:
            self._compute(self.agents, self.rail)

        return self.distance_map

    def reset(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        """
        Reset the distance map
        """
        self.reset_was_called = True
        self.agents: List[EnvAgent] = agents
        self.rail = rail

    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[Waypoint]]]:
        raise NotImplementedError()

    def _compute(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        raise NotImplementedError()


class DistanceMap(AbstractDistanceMap[RailGridTransitionMap, np.ndarray, Waypoint]):
    def __init__(self, agents: List[EnvAgent], env_height: int, env_width: int):
        super().__init__(agents=agents)
        self.env_height = env_height
        self.env_width = env_width

    def reset(self, agents: List[EnvAgent], rail: RailGridTransitionMap):
        """
        Reset the distance map
        """
        super().reset(agents=agents, rail=rail)
        self.env_height = rail.height
        self.env_width = rail.width

    def _compute(self, agents: List[EnvAgent], rail: RailGridTransitionMap):
        """
        This function computes the distance maps for each unique target. Thus, if several targets are the same
        we only compute the distance for them once and copy to all targets with same position.
        :param agents: All the agents in the environment, independent of their current status
        :param rail: The rail transition map

        """
        self.agents_previous_computation = self.agents
        self.distance_map = np.full(shape=(len(agents),
                                           self.env_height,
                                           self.env_width,
                                           4),
                                    fill_value=np.inf
                                    )
        distance_map_walker = DistanceMapWalker(self.distance_map)
        computed_targets = []
        for i, agent in enumerate(agents):
            if agent.target not in computed_targets:
                distance_map_walker._distance_map_walker(rail, agent.target, i)
            else:
                # just copy the distance map form other agent with same target (performance)
                self.distance_map[i, :, :, :] = np.copy(
                    self.distance_map[computed_targets.index(agent.target), :, :, :])
            computed_targets.append(agent.target)

    # N.B. get_shortest_paths is not part of distance_map since it refers to RailEnvActions (would lead to circularity!)
    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[Waypoint]]]:
        """
        Computes the shortest path for each agent to its target and the action to be taken to do so.
        The paths are derived from a `DistanceMap`.

        If there is no path (rail disconnected), the path is given as None.
        The agent state (moving or not) and its speed are not taken into account

        example:
                agent_fixed_travel_paths = get_shortest_paths(env.distance_map, None, agent.handle)
                path = agent_fixed_travel_paths[agent.handle]

        Parameters
        ----------
        self : reference to the distance_map
        max_depth : max path length, if the shortest path is longer, it will be cutted
        agent_handle : if set, the shortest for agent.handle will be returned , otherwise for all agents

        Returns
        -------
            Dict[int, Optional[List[WalkingElement]]]

        """
        shortest_paths = dict()

        def _shortest_path_for_agent(agent: EnvAgent):
            if agent.state.is_off_map_state():
                configuration = agent.initial_configuration
            elif agent.state.is_on_map_state():
                configuration = agent.current_configuration
            elif agent.state == TrainState.DONE:
                shortest_paths[agent.handle] = None
                return
            else:
                shortest_paths[agent.handle] = None
                return


            shortest_paths[agent.handle] = []
            distance = math.inf
            depth = 0
            while configuration not in agent.targets and (max_depth is None or depth < max_depth):
                next_actions = self.rail.get_valid_move_actions(configuration)
                best_next_action = None
                for next_action in next_actions:
                    next_action_distance = self.get()[agent.handle, next_action.next_position[0], next_action.next_position[1], next_action.next_direction]
                    if next_action_distance < distance:
                        best_next_action = next_action
                        distance = next_action_distance
                shortest_paths[agent.handle].append(Waypoint(*configuration))
                depth += 1

                # if there is no way to continue, the rail must be disconnected!
                # (or distance map is incorrect)
                if best_next_action is None:
                    shortest_paths[agent.handle] = None
                    return
                configuration = (best_next_action.next_position, best_next_action.next_direction)
            if max_depth is None or depth < max_depth:
                shortest_paths[agent.handle].append(Waypoint(*configuration))

        if agent_handle is not None:
            _shortest_path_for_agent(self.agents[agent_handle])
        else:
            for agent in self.agents:
                _shortest_path_for_agent(agent)

        return shortest_paths
