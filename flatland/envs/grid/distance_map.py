import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from flatland.core.distance_map import AbstractDistanceMap
from flatland.core.distance_map_walker import DistanceMapWalker
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


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
        distance_map_walker = DistanceMapWalker[DistanceMap, RailGridTransitionMap, Tuple[Tuple[int, int], int], int](self)
        computed_targets = []
        for i, agent in enumerate(agents):
            # TODO safe?
            if agent.target not in computed_targets:
                targets = [target for target in agent.targets if rail.is_valid_configuration(target)]
                distance_map_walker._distance_map_walker(rail, i, targets)
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

    def _set_distance(self, configuration: Tuple[Tuple[int, int], int], target_nr: int, new_distance: int):
        position, direction = configuration
        self.distance_map[target_nr, *position, direction] = new_distance

    def _get_distance(self, configuration: Tuple[Tuple[int, int], int], target_nr: int):
        position, direction = configuration
        return self.distance_map[target_nr, *position, direction]
