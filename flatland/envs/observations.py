"""
Collection of environment-specific ObservationBuilder.
"""
from collections import deque

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import coordinate_to_position


class TreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the tree structure of the rail
    network to simplify the representation of the state of the environment for each agent.
    """

    def __init__(self, max_depth, predictor=None):
        self.max_depth = max_depth

        # Compute the size of the returned observation vector
        size = 0
        pow4 = 1
        for i in range(self.max_depth + 1):
            size += pow4
            pow4 *= 4
        self.observation_dim = 8
        self.observation_space = [size * self.observation_dim]
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.agents_previous_reset = None

    def reset(self):
        agents = self.env.agents
        nAgents = len(agents)

        compute_distance_map = True
        if self.agents_previous_reset is not None:
            if nAgents == len(self.agents_previous_reset):
                compute_distance_map = False
                for i in range(nAgents):
                    if agents[i].target != self.agents_previous_reset[i].target:
                        compute_distance_map = True
        self.agents_previous_reset = agents

        if compute_distance_map:
            self._compute_distance_map()

    def _compute_distance_map(self):
        agents = self.env.agents
        nAgents = len(agents)
        self.distance_map = np.inf * np.ones(shape=(nAgents,  # self.env.number_of_agents,
                                                    self.env.height,
                                                    self.env.width,
                                                    4))
        self.max_dist = np.zeros(nAgents)
        self.max_dist = [self._distance_map_walker(agent.target, i) for i, agent in enumerate(agents)]
        # Update local lookup table for all agents' target locations
        self.location_has_target = {tuple(agent.target): 1 for agent in agents}

    def _distance_map_walker(self, position, target_nr):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map

        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
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

            if new_cell[0] >= 0 and new_cell[0] < self.env.height and new_cell[1] >= 0 and new_cell[1] < self.env.width:

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
        if movement == Grid4TransitionsEnum.NORTH:
            return (position[0] - 1, position[1])
        elif movement == Grid4TransitionsEnum.EAST:
            return (position[0], position[1] + 1)
        elif movement == Grid4TransitionsEnum.SOUTH:
            return (position[0] + 1, position[1])
        elif movement == Grid4TransitionsEnum.WEST:
            return (position[0], position[1] - 1)

    def get_many(self, handles=[]):
        """
        Called whenever an observation has to be computed for the `env' environment, for each agent with handle
        in the `handles' list.
        """

        if self.predictor:
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get(custom_args={'distance_map': self.distance_map})
            for t in range(len(self.predictions[0])):
                pos_list = []
                dir_list = []
                for a in handles:
                    pos_list.append(self.predictions[a][t][1:3])
                    dir_list.append(self.predictions[a][t][3])
                self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                self.predicted_dir.update({t: dir_list})
            self.max_prediction_depth = len(self.predicted_pos)
        observations = {}
        for h in handles:
            observations[h] = self.get(h)
        return observations

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

        #1: 1 if own target lies on the explored branch

        #2: distance toa target of another agent is detected between the previous node and the current one.

        #3: distance to another agent is detected between the previous node and the current one.

        #4: distance of agent to the current branch node

        #5: minimum distance from node to the agent's target (when landing to the node following the corresponding
            branch.

        #6: agent in the same direction
            1 = agent present same direction
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #7: agent in the opposite drection
            1 = agent present other direction than myself (so conflict)
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #8: possible conflict detected
            1 = Other agent predicts to pass along this cell at the same time as the agent

            0 = No other agent reserve the same cell at similar time


        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target].
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        # Update local lookup table for all agents' positions
        self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents}
        self.location_has_agent_direction = {tuple(agent.position): agent.direction for agent in self.env.agents}
        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index
        possible_transitions = self.env.rail.get_transitions((*agent.position, agent.direction))
        num_transitions = np.count_nonzero(possible_transitions)

        # Root node - current position
        observation = [0, 0, 0, 0, self.distance_map[(handle, *agent.position, agent.direction)], 0, 0, 0]

        root_observation = observation[:]
        visited = set()
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        # TODO: Test if this works as desired!
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_cell = self._new_position(agent.position, branch_direction)
                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, root_observation, 0, 1)
                observation = observation + branch_observation
                visited = visited.union(branch_visited)
            else:
                num_cells_to_fill_in = 0
                pow4 = 1
                for i in range(self.max_depth):
                    num_cells_to_fill_in += pow4
                    pow4 *= 4
                observation = observation + ([-np.inf] * self.observation_dim) * num_cells_to_fill_in
        self.env.dev_obs_dict[handle] = visited
        return observation

    def _explore_branch(self, handle, position, direction, root_observation, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        """
        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_isSwitch = False
        last_isDeadEnd = False
        last_isTerminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_isTarget = False

        visited = set()
        agent = self.env.agents[handle]
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        potential_conflict = 0
        num_steps = 1
        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                if num_steps < other_agent_encountered:
                    other_agent_encountered = num_steps

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                if self.location_has_agent_direction[position] != direction:
                    # Cummulate the number of agents on branch with other direction
                    other_agent_opposite_direction += 1

            # Register possible conflict
            if self.predictor and num_steps < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:
                    pre_step = max(0, tot_dist - 1)
                    post_step = min(self.max_prediction_depth - 1, tot_dist + 1)

                    # Look for opposing paths at distance num_step
                    if int_position in np.delete(self.predicted_pos[tot_dist], handle):
                        conflicting_agent = np.where(np.delete(self.predicted_pos[tot_dist], handle) == int_position)
                        for ca in conflicting_agent:
                            if direction != self.predicted_dir[tot_dist][ca[0]]:
                                potential_conflict = 1
                    # Look for opposing paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent:
                            if direction != self.predicted_dir[pre_step][ca[0]]:
                                potential_conflict = 1
                    # Look for opposing paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle):
                        conflicting_agent = np.where(np.delete(self.predicted_pos[post_step], handle) == int_position)
                        for ca in conflicting_agent:
                            if direction != self.predicted_dir[post_step][ca[0]]:
                                potential_conflict = 1

            if position in self.location_has_target and position != agent.target:
                if num_steps < other_target_encountered:
                    other_target_encountered = num_steps

            if position == agent.target:
                if num_steps < own_target_encountered:
                    own_target_encountered = num_steps

            # #############################
            # #############################

            if (position[0], position[1], direction) in visited:
                last_isTerminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
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
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = self._new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
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
        other_agent_same_direction = \
            1 if position in self.location_has_agent and self.location_has_agent_direction[position] == direction else 0
        other_agent_opposite_direction = \
            1 if position in self.location_has_agent and self.location_has_agent_direction[position] != direction else 0

        if last_isTarget:
            observation = [0,
                           other_target_encountered,
                           other_agent_encountered,
                           root_observation[3] + num_steps,
                           0,
                           other_agent_same_direction,
                           other_agent_opposite_direction
                           ]

        elif last_isTerminal:
            observation = [0,
                           other_target_encountered,
                           other_agent_encountered,
                           np.inf,
                           np.inf,
                           other_agent_same_direction,
                           other_agent_opposite_direction
                           ]
        else:
            observation = [0,
                           other_target_encountered,
                           other_agent_encountered,
                           root_observation[3] + num_steps,
                           self.distance_map[handle, position[0], position[1], direction],
                           other_agent_same_direction,
                           other_agent_opposite_direction
                           ]
        """

        if last_isTarget:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           root_observation[3] + num_steps,
                           0,
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           potential_conflict
                           ]

        elif last_isTerminal:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           np.inf,
                           np.inf,
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           potential_conflict
                           ]
        else:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           root_observation[3] + num_steps,
                           self.distance_map[handle, position[0], position[1], direction],
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           potential_conflict
                           ]
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
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          (branch_direction + 2) % 4,
                                                                          new_root_observation, tot_dist + 1,
                                                                          depth + 1)
                observation = observation + branch_observation
                if len(branch_visited) != 0:
                    visited = visited.union(branch_visited)
            elif last_isSwitch and possible_transitions[branch_direction]:
                new_cell = self._new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          branch_direction,
                                                                          new_root_observation, tot_dist + 1,
                                                                          depth + 1)
                observation = observation + branch_observation
                if len(branch_visited) != 0:
                    visited = visited.union(branch_visited)
            else:
                num_cells_to_fill_in = 0
                pow4 = 1
                for i in range(self.max_depth - depth):
                    num_cells_to_fill_in += pow4
                    pow4 *= 4
                observation = observation + ([-np.inf] * self.observation_dim) * num_cells_to_fill_in

        return observation, visited

    def util_print_obs_subtree(self, tree, num_features_per_node=8, prompt='', current_depth=0):
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

    def split_tree(self, tree, num_features_per_node=8, current_depth=0):
        """

        :param tree:
        :param num_features_per_node:
        :param prompt:
        :param current_depth:
        :return:
        """

        if len(tree) < num_features_per_node:
            return [], [], []

        depth = 0
        tmp = len(tree) / num_features_per_node - 1
        pow4 = 4
        while tmp > 0:
            tmp -= pow4
            depth += 1
            pow4 *= 4
        child_size = (len(tree) - num_features_per_node) // 4
        tree_data = tree[:4].tolist()
        distance_data = [tree[4]]
        agent_data = tree[5:num_features_per_node].tolist()
        for children in range(4):
            child_tree = tree[(num_features_per_node + children * child_size):
                              (num_features_per_node + (children + 1) * child_size)]
            tmp_tree_data, tmp_distance_data, tmp_agent_data = self.split_tree(child_tree,
                                                                               num_features_per_node,
                                                                               current_depth=current_depth + 1)
            if len(tmp_tree_data) > 0:
                tree_data.extend(tmp_tree_data)
                distance_data.extend(tmp_distance_data)
                agent_data.extend(tmp_agent_data)
        return tree_data, distance_data, agent_data

    def _set_env(self, env):
        self.env = env
        if self.predictor:
            self.predictor._set_env(self.env)


class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),
          assuming 16 bits encoding of transitions.

        - Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent
         target and the positions of the other agents targets.

        - A 3D array (map_height, map_width, 8) with the 4 first channels containing the one hot encoding
          of the direction of the given agent and the 4 second channels containing the positions
          of the other agents at their position coordinates.
    """

    def __init__(self):
        self.observation_space = ()
        super(GlobalObsForRailEnv, self).__init__()

    def _set_env(self, env):
        super()._set_env(env)

        self.observation_space = [4, self.env.height, self.env.width]

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_transitions((i, j)))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle):
        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 8))
        agents = self.env.agents
        agent = agents[handle]

        direction = np.zeros(4)
        direction[agent.direction] = 1
        agent_pos = agents[handle].position
        obs_agents_state[agent_pos][:4] = direction
        obs_targets[agent.target][0] += 1

        for i in range(len(agents)):
            if i != handle:  # TODO: handle used as index...?
                agent2 = agents[i]
                obs_agents_state[agent2.position][4 + agent2.direction] = 1
                obs_targets[agent2.target][1] += 1

        direction = self._get_one_hot_for_agent_direction(agent)

        return self.rail_obs, obs_agents_state, obs_targets, direction


class GlobalObsForRailEnvDirectionDependent(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),
          assuming 16 bits encoding of transitions, flipped in the direction of the agent
          (the agent is always heading north on the flipped view).

        - Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent
         target and the positions of the other agents targets, also flipped depending on the agent's direction.

        - A 3D array (map_height, map_width, 5) containing the one hot encoding of the direction of the other
          agents at their position coordinates, and the last channel containing the position of the given agent.

        - A 4 elements array with one hot encoding of the direction.
    """

    def __init__(self):
        self.observation_space = ()
        super(GlobalObsForRailEnvDirectionDependent, self).__init__()

    def _set_env(self, env):
        super()._set_env(env)

        self.observation_space = [4, self.env.height, self.env.width]

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_transitions((i, j)))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle):
        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 5))
        agents = self.env.agents
        agent = agents[handle]
        direction = agent.direction

        idx = np.tile(np.arange(16), 2)

        rail_obs = self.rail_obs[:, :, idx[direction * 4: direction * 4 + 16]]

        if direction == 1:
            rail_obs = np.flip(rail_obs, axis=1)
        elif direction == 2:
            rail_obs = np.flip(rail_obs)
        elif direction == 3:
            rail_obs = np.flip(rail_obs, axis=0)

        agent_pos = agents[handle].position
        obs_agents_state[agent_pos][0] = 1
        obs_targets[agent.target][0] += 1

        idx = np.tile(np.arange(4), 2)
        for i in range(len(agents)):
            if i != handle:  # TODO: handle used as index...?
                agent2 = agents[i]
                obs_agents_state[agent2.position][1 + idx[4 + (agent2.direction - direction)]] = 1
                obs_targets[agent2.target][1] += 1

        direction = self._get_one_hot_for_agent_direction(agent)

        return rail_obs, obs_agents_state, obs_targets, direction


class LocalObsForRailEnv(ObservationBuilder):
    """
    Gives a local observation of the rail environment around the agent.
    The observation is composed of the following elements:

        - transition map array of the local environment around the given agent,
          with dimensions (2*view_radius + 1, 2*view_radius + 1, 16),
          assuming 16 bits encoding of transitions.

        - Two 2D arrays containing respectively, if they are in the agent's vision range,
          its target position, the positions of the other targets.

        - A 3D array (map_height, map_width, 4) containing the one hot encoding of directions
          of the other agents at their position coordinates, if they are in the agent's vision range.

        - A 4 elements array with one hot encoding of the direction.
    """

    def __init__(self, view_radius):
        """
        :param view_radius:
        """
        super(LocalObsForRailEnv, self).__init__()
        self.view_radius = view_radius

    def reset(self):
        # We build the transition map with a view_radius empty cells expansion on each side.
        # This helps to collect the local transition map view when the agent is close to a border.

        self.rail_obs = np.zeros((self.env.height + 2 * self.view_radius,
                                  self.env.width + 2 * self.view_radius, 16))
        for i in range(self.env.height):
            for j in range(self.env.width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_transitions((i, j)))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i + self.view_radius, j + self.view_radius] = np.array(bitlist)
                # self.rail_obs[i+self.view_radius, j+self.view_radius] = np.array(
                #     list(f'{self.env.rail.get_transitions((i, j)):016b}')).astype(int)

    def get(self, handle):
        agents = self.env.agents
        agent = agents[handle]

        local_rail_obs = self.rail_obs[agent.position[0]: agent.position[0] + 2 * self.view_radius + 1,
                         agent.position[1]:agent.position[1] + 2 * self.view_radius + 1]

        obs_map_state = np.zeros((2 * self.view_radius + 1, 2 * self.view_radius + 1, 2))

        obs_other_agents_state = np.zeros((2 * self.view_radius + 1, 2 * self.view_radius + 1, 4))

        def relative_pos(pos):
            return [agent.position[0] - pos[0], agent.position[1] - pos[1]]

        def is_in(rel_pos):
            return (abs(rel_pos[0]) <= self.view_radius) and (abs(rel_pos[1]) <= self.view_radius)

        target_rel_pos = relative_pos(agent.target)
        if is_in(target_rel_pos):
            obs_map_state[self.view_radius + np.array(target_rel_pos)][0] += 1

        for i in range(len(agents)):
            if i != handle:  # TODO: handle used as index...?
                agent2 = agents[i]

                agent_2_rel_pos = relative_pos(agent2.position)
                if is_in(agent_2_rel_pos):
                    obs_other_agents_state[self.view_radius + agent_2_rel_pos[0],
                                           self.view_radius + agent_2_rel_pos[1]][agent2.direction] += 1

                target_rel_pos_2 = relative_pos(agent2.position)
                if is_in(target_rel_pos_2):
                    obs_map_state[self.view_radius + np.array(target_rel_pos_2)][1] += 1

        direction = self._get_one_hot_for_agent_direction(agent)

        return local_rail_obs, obs_map_state, obs_other_agents_state, direction
