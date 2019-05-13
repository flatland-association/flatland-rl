"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ Reset() is called after each environment reset, to allow for pre-computing relevant data.

+ Get() is called whenever an observation has to be computed, potentially for each agent independently in
case of multi-agent environments.
"""

import numpy as np

from collections import deque


class ObservationBuilder:
    """
    ObservationBuilder base class.
    """

    def __init__(self):
        pass

    def _set_env(self, env):
        self.env = env

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get(self, handle=0):
        """
        Called whenever an observation has to be computed for the `env' environment, possibly
        for each agent independently (agent id `handle').

        Parameters
        -------
        handle : int (optional)
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            An observation structure, specific to the corresponding environment.
        """
        raise NotImplementedError()


class TreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the tree structure of the rail
    network to simplify the representation of the state of the environment for each agent.
    """

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def reset(self):
        agents = self.env.agents
        nAgents = len(agents)
        self.distance_map = np.inf * np.ones(shape=(nAgents,  # self.env.number_of_agents,
                                                    self.env.height,
                                                    self.env.width,
                                                    4))
        self.max_dist = np.zeros(nAgents)

        # for i in range(nAgents):
        #     self.max_dist[i] = self._distance_map_walker(self.env.agents_target[i], i)
        self.max_dist = [self._distance_map_walker(agent.target, i) for i, agent in enumerate(agents)]

        # Update local lookup table for all agents' target locations
        self.location_has_target = {}
        # for loc in self.env.agents_target:
        #    self.location_has_target[(loc[0], loc[1])] = 1
        self.location_has_target = {tuple(agent.target): 1 for agent in agents}

    def _distance_map_walker(self, position, target_nr):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map

        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
        # nodes_queue = []  # list of tuples (row, col, direction, distance);
        # direction is the direction of movement, meaning that at least a possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(position, target_nr, 0, enforce_target_direction=-1))

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = set([(position[0], position[1], 0),
                       (position[0], position[1], 1),
                       (position[0], position[1], 2),
                       (position[0], position[1], 3)])

        max_distance = 0

        while nodes_queue:
            node = nodes_queue.popleft()

            node_id = (node[0], node[1], node[2])

            if node_id not in visited:
                visited.add(node_id)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to direction node[2]
                valid_neighbors = self._get_and_update_neighbors((node[0], node[1]), target_nr, node[3], node[2])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, node[3] + 1)

        return max_distance

    def _get_and_update_neighbors(self, position, target_nr, current_distance, enforce_target_direction=-1):
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
            new_cell = self._new_position(position, neigh_direction)

            if new_cell[0] >= 0 and new_cell[0] < self.env.height and \
                new_cell[1] >= 0 and new_cell[1] < self.env.width:

                desired_movement_from_new_cell = (neigh_direction + 2) % 4

                """
                # Is the next cell a dead-end?
                isNextCellDeadEnd = False
                nbits = 0
                tmp = self.env.rail.get_transitions((new_cell[0], new_cell[1]))
                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1
                if nbits == 1:
                    # Dead-end!
                    isNextCellDeadEnd = True
                """

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    isValid = self.env.rail.get_transition((new_cell[0], new_cell[1], agent_orientation),
                                                           desired_movement_from_new_cell)

                    if isValid:
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

    def get(self, handle):
        """
        Computes the current observation for agent `handle' in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is:
            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']





        Each branch data is organized as:
            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Finally, each node information is composed of 5 floating point values:

        #1:

        #2: 1 if a target of another agent is detected between the previous node and the current one.

        #3: 1 if another agent is detected between the previous node and the current one.

        #4: distance of agent to the current branch node

        #5: minimum distance from node to the agent's target (when landing to the node following the corresponding
            branch.

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target].
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        # Update local lookup table for all agents' positions
        # self.location_has_agent = {}
        # for loc in self.env.agents_position:
        #    self.location_has_agent[(loc[0], loc[1])] = 1
        self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents}

        agent = self.env.agents[handle]  # TODO: handle being treated as index
        # position = self.env.agents_position[handle]
        # orientation = self.env.agents_direction[handle]
        possible_transitions = self.env.rail.get_transitions((*agent.position, agent.direction))
        num_transitions = np.count_nonzero(possible_transitions)
        # Root node - current position
        # observation = [0, 0, 0, 0, self.distance_map[handle, position[0], position[1], orientation]]
        observation = [0, 0, 0, 0, self.distance_map[(handle, *agent.position, agent.direction)]]
        root_observation = observation[:]

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        # TODO: Test if this works as desired!
        orientation = agent.direction
        if num_transitions == 1:
            orientation == np.argmax(possible_transitions)

        # for branch_direction in [(orientation + 4 + i) % 4 for i in range(-1, 3)]:
        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_cell = self._new_position(agent.position, branch_direction)

                branch_observation = self._explore_branch(handle, new_cell, branch_direction, root_observation, 1)
                observation = observation + branch_observation
            else:
                num_cells_to_fill_in = 0
                pow4 = 1
                for i in range(self.max_depth):
                    num_cells_to_fill_in += pow4
                    pow4 *= 4
                observation = observation + [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf] * num_cells_to_fill_in
        return observation

    def _explore_branch(self, handle, position, direction, root_observation, depth):
        """
        Utility function to compute tree-based observations.
        """
        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_isSwitch = False
        last_isDeadEnd = False
        last_isTerminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_isTarget = False

        visited = set()

        # other_agent_encountered = False
        # other_target_encountered = False
        other_agent_encountered = np.inf
        other_target_encountered = np.inf

        num_steps = 1
        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                # other_agent_encountered = True
                if num_steps < other_agent_encountered:
                    other_agent_encountered = num_steps

            if position in self.location_has_target:
                # other_target_encountered = True
                if num_steps < other_target_encountered:
                    other_target_encountered = num_steps
            # #############################
            # #############################

            if (position[0], position[1], direction) in visited:
                last_isTerminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            # if position[0] == self.env.agents_target[handle][0] and position[1] == self.env.agents_target[handle][1]:
            if np.array_equal(position, self.env.agents[handle].target):
                last_isTarget = True
                break

            cell_transitions = self.env.rail.get_transitions((*position, direction))
            num_transitions = np.count_nonzero(cell_transitions)
            exploring = False
            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = 0
                tmp = self.env.rail.get_transitions(tuple(position))
                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1
                if nbits == 1:
                    # Dead-end!
                    last_isDeadEnd = True

                if not last_isDeadEnd:
                    # Keep walking through the tree along `direction'
                    exploring = True
                    direction = np.argmax(cell_transitions)
                    position = self._new_position(position, direction)
                    num_steps += 1

            elif num_transitions > 0:
                # Switch detected
                last_isSwitch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_isTerminal = True
                break

        # `position' is either a terminal node or a switch

        observation = []

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!
        """
        if last_isTarget:
            observation = [0,
                           1 if other_target_encountered else 0,
                           1 if other_agent_encountered else 0,
                           root_observation[3] + num_steps,
                           0]

        elif last_isTerminal:
            observation = [0,
                           1 if other_target_encountered else 0,
                           1 if other_agent_encountered else 0,
                           np.inf,
                           np.inf]
        else:
            observation = [0,
                           1 if other_target_encountered else 0,
                           1 if other_agent_encountered else 0,
                           root_observation[3] + num_steps,
                           self.distance_map[handle, position[0], position[1], direction]]
        """
        if last_isTarget:
            observation = [0,
                           other_target_encountered,
                           other_agent_encountered,
                           root_observation[3] + num_steps,
                           0]

        elif last_isTerminal:
            observation = [0,
                           other_target_encountered,
                           other_agent_encountered,
                           np.inf,
                           np.inf]
        else:
            observation = [0,
                           other_target_encountered,
                           other_agent_encountered,
                           root_observation[3] + num_steps,
                           self.distance_map[handle, position[0], position[1], direction]]
        # #############################
        # #############################

        new_root_observation = observation[:]
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions((*position, direction))
        for branch_direction in [(direction + 4 + i) % 4 for i in range(-1, 3)]:
            if last_isDeadEnd and self.env.rail.get_transition((*position, direction),
                                                               (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = self._new_position(position, (branch_direction + 2) % 4)
                branch_observation = self._explore_branch(handle,
                                                          new_cell,
                                                          (branch_direction + 2) % 4,
                                                          new_root_observation,
                                                          depth + 1)
                observation = observation + branch_observation

            elif last_isSwitch and possible_transitions[branch_direction]:
                new_cell = self._new_position(position, branch_direction)
                branch_observation = self._explore_branch(handle,
                                                          new_cell,
                                                          branch_direction,
                                                          new_root_observation,
                                                          depth + 1)
                observation = observation + branch_observation

            else:
                num_cells_to_fill_in = 0
                pow4 = 1
                for i in range(self.max_depth - depth):
                    num_cells_to_fill_in += pow4
                    pow4 *= 4
                observation = observation + [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf] * num_cells_to_fill_in

        return observation

    def util_print_obs_subtree(self, tree, num_features_per_node=5, prompt='', current_depth=0):
        """
        Utility function to pretty-print tree observations returned by this object.
        """
        if len(tree) < num_features_per_node:
            return

        depth = 0
        tmp = len(tree) / num_features_per_node - 1
        pow4 = 4
        while tmp > 0:
            tmp -= pow4
            depth += 1
            pow4 *= 4

        prompt_ = ['L:', 'F:', 'R:', 'B:']

        print("  " * current_depth + prompt, tree[0:num_features_per_node])
        child_size = (len(tree) - num_features_per_node) // 4
        for children in range(4):
            child_tree = tree[(num_features_per_node + children * child_size):
                              (num_features_per_node + (children + 1) * child_size)]
            self.util_print_obs_subtree(child_tree,
                                        num_features_per_node,
                                        prompt=prompt_[children],
                                        current_depth=current_depth + 1)

    def split_tree(self, tree, num_features_per_node=5, current_depth=0):
        """

        :param tree:
        :param num_features_per_node:
        :param prompt:
        :param current_depth:
        :return:
        """

        if len(tree) < num_features_per_node:
            return [], []

        depth = 0
        tmp = len(tree) / num_features_per_node - 1
        pow4 = 4
        while tmp > 0:
            tmp -= pow4
            depth += 1
            pow4 *= 4
        child_size = (len(tree) - num_features_per_node) // 4
        tree_data = tree[0:num_features_per_node - 1].tolist()
        distance_data = [tree[num_features_per_node - 1]]
        for children in range(4):
            child_tree = tree[(num_features_per_node + children * child_size):
                              (num_features_per_node + (children + 1) * child_size)]
            tmp_tree_data, tmp_distance_data = self.split_tree(child_tree,
                                                               num_features_per_node,
                                                               current_depth=current_depth + 1)
            if len(tmp_tree_data) > 0:
                tree_data.extend(tmp_tree_data)
                distance_data.extend(tmp_distance_data)
        return tree_data, distance_data


class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),
          assuming 16 bits encoding of transitions.

        - Four 2D arrays containing respectively the position of the given agent,
          the position of its target, the positions of the other agents and of
          their target.

        - A 4 elements array with one of encoding of the direction.
    """

    def __init__(self):
        super(GlobalObsForRailEnv, self).__init__()

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                self.rail_obs[i, j] = np.array(
                    list(f'{self.env.rail.get_transitions((i, j)):016b}')).astype(int)

        # self.targets = np.zeros(self.env.height, self.env.width)
        # for target_pos in self.env.agents_target:
        #     self.targets[target_pos] += 1

    def get(self, handle):
        obs = np.zeros((4, self.env.height, self.env.width))
        agents = self.env.agents
        agent = agents[handle]

        agent_pos = agents[handle].position
        obs[0][agent_pos] += 1
        obs[1][agent.target] += 1

        for i in range(len(agents)):
            if i != handle:  # TODO: handle used as index...?
                agent2 = agents[i]
                obs[3][agent2.position] += 1
                obs[2][agent2.target] += 1

        direction = np.zeros(4)
        direction[agent.direction] = 1

        return self.rail_obs, obs, direction
