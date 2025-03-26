import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


class DistanceMap:
    def __init__(self, agents: List[EnvAgent], env_height: int, env_width: int):
        self.env_height = env_height
        self.env_width = env_width
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

    def reset(self, agents: List[EnvAgent], rail: RailGridTransitionMap):
        """
        Reset the distance map
        """
        self.reset_was_called = True
        self.agents: List[EnvAgent] = agents
        self.rail = rail
        self.env_height = rail.height
        self.env_width = rail.width

    def _compute(self, agents: List[EnvAgent], rail: RailGridTransitionMap):
        """
        This function computes the distance maps for each unique target. Thus if several targets are the same
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

        computed_targets = []
        for i, agent in enumerate(agents):
            if agent.target not in computed_targets:
                self._distance_map_walker(rail, agent.target, i)
            else:
                # just copy the distance map form other agent with same target (performance)
                self.distance_map[i, :, :, :] = np.copy(
                    self.distance_map[computed_targets.index(agent.target), :, :, :])
            computed_targets.append(agent.target)

    def _distance_map_walker(self, rail: RailGridTransitionMap, position, target_nr: int):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map
        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least a possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(rail, position, target_nr, 0, enforce_target_direction=-1))

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = {(position[0], position[1], 0), (position[0], position[1], 1), (position[0], position[1], 2),
                   (position[0], position[1], 3)}

        max_distance = 0

        while nodes_queue:
            node = nodes_queue.popleft()

            node_id = (node[0], node[1], node[2])

            if node_id not in visited:
                visited.add(node_id)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to direction node[2]
                valid_neighbors = self._get_and_update_neighbors(rail, (node[0], node[1]), target_nr, node[3], node[2])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, node[3] + 1)

        return max_distance

    def _get_and_update_neighbors(self, rail: RailGridTransitionMap, position, target_nr, current_distance,
                                  enforce_target_direction=-1):
        """
        Utility function used by _distance_map_walker to perform a BFS walk over the rail, filling in the
        minimum distances from each target cell.
        """
        neighbors = []

        possible_directions = [0, 1, 2, 3]
        if enforce_target_direction >= 0:
            # The agent must land into the current cell with orientation `enforce_target_direction'.
            # This is only possible if the agent has arrived from the cell in the opposite direction!
            possible_directions = [(enforce_target_direction + 2) % 4]

        for neigh_direction in possible_directions:
            new_cell = get_new_position(position, neigh_direction)

            if new_cell[0] >= 0 and new_cell[0] < self.env_height and new_cell[1] >= 0 and new_cell[1] < self.env_width:

                desired_movement_from_new_cell = (neigh_direction + 2) % 4

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    is_valid = rail.get_transition((new_cell[0], new_cell[1], agent_orientation),
                                                   desired_movement_from_new_cell)

                    if is_valid:
                        """
                        # TODO: check that it works with deadends! -- still bugged!
                        movement = desired_movement_from_new_cell
                        if isNextCellDeadEnd:
                            movement = (desired_movement_from_new_cell+2) % 4
                        """
                        new_distance = min(self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation],
                                           current_distance + 1)
                        neighbors.append((new_cell[0], new_cell[1], agent_orientation, new_distance))
                        self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation] = new_distance

        return neighbors

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

        def _shortest_path_for_agent(agent):
            if agent.state.is_off_map_state():
                position = agent.initial_position
            elif agent.state.is_on_map_state():
                position = agent.position
            elif agent.state == TrainState.DONE:
                position = agent.target
            else:
                shortest_paths[agent.handle] = None
                return
            direction = agent.direction
            shortest_paths[agent.handle] = []
            distance = math.inf
            depth = 0
            while (position != agent.target and (max_depth is None or depth < max_depth)):
                next_actions = self.rail.get_valid_move_actions_(direction, position)
                best_next_action = None
                for next_action in next_actions:
                    next_action_distance = self.get()[
                        agent.handle, next_action.next_position[0], next_action.next_position[
                            1], next_action.next_direction]
                    if next_action_distance < distance:
                        best_next_action = next_action
                        distance = next_action_distance

                shortest_paths[agent.handle].append(Waypoint(position, direction))
                depth += 1

                # if there is no way to continue, the rail must be disconnected!
                # (or distance map is incorrect)
                if best_next_action is None:
                    shortest_paths[agent.handle] = None
                    return

                position = best_next_action.next_position
                direction = best_next_action.next_direction
            if max_depth is None or depth < max_depth:
                shortest_paths[agent.handle].append(Waypoint(position, direction))

        if agent_handle is not None:
            _shortest_path_for_agent(self.agents[agent_handle])
        else:
            for agent in self.agents:
                _shortest_path_for_agent(agent)

        return shortest_paths
