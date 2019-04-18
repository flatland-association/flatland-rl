import numpy as np
from collections import deque

# TODO: add docstrings, pylint, etc...


class ObservationBuilder:
    def __init__(self, env):
        self.env = env

    def reset(self):
        raise NotImplementedError()

    def get(self, handle):
        raise NotImplementedError()


class TreeObsForRailEnv(ObservationBuilder):
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.distance_map = np.inf * np.ones(shape=(self.env.number_of_agents,
                                                    self.env.height,
                                                    self.env.width))
        self.max_dist = np.zeros(self.env.number_of_agents)

        for i in range(self.env.number_of_agents):
            self.max_dist[i] = self._distance_map_walker(self.env.agents_target[i], i)


    def _distance_map_walker(self, position, target_nr):
        # Returns max distance to target, from the farthest away node, while filling in distance_map

        for ori in range(4):
            self.distance_map[target_nr, position[0], position[1]] = 0

        # Fill in the (up to) 4 neighboring nodes
        # nodes_queue = []  # list of tuples (row, col, direction, distance);
        # direction is the direction of movement, meaning that at least a possible orientation
        # of an agent in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(position,
                                                           target_nr, 0, enforce_target_direction=-1))

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = set([(position[0], position[1], 0), (position[0], position[1], 1),
                       (position[0], position[1], 2), (position[0], position[1], 3)])

        max_distance = 0

        while nodes_queue:
            node = nodes_queue.popleft()

            node_id = (node[0], node[1], node[2])

            #print(node_id, visited, (node_id in visited))
            #print(nodes_queue)

            if node_id not in visited:
                visited.add(node_id)

                # From the list of possible neighbors that have at least a path to the
                # current node, only keep those whose new orientation in the current cell
                # would allow a transition to direction node[2]
                valid_neighbors = self._get_and_update_neighbors(
                    (node[0], node[1]), target_nr, node[3], node[2])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors)>0:
                    max_distance = max(max_distance, node[3]+1)

        return max_distance


    def _get_and_update_neighbors(self, position, target_nr, current_distance, enforce_target_direction=-1):
        neighbors = []

        for direction in range(4):
            new_cell = self._new_position(position, (direction+2)%4)

            if new_cell[0]>=0 and new_cell[0]<self.env.height and\
                new_cell[1]>=0 and new_cell[1]<self.env.width:
                # Check if the two cells are connected by a valid transition
                transitionValid = False
                for orientation in range(4):
                    moves = self.env.rail.get_transitions((new_cell[0], new_cell[1], orientation))
                    if moves[direction]:
                        transitionValid = True
                        break

                if not transitionValid:
                    continue

                # Check if a transition in direction node[2] is possible if an agent
                # lands in the current cell with orientation `direction'; this only
                # applies to cells that are not dead-ends!
                directionMatch = True
                if enforce_target_direction>=0:
                    directionMatch = self.env.rail.get_transition(
                        (new_cell[0], new_cell[1], direction), enforce_target_direction)

                # If transition is found to invalid, check if perhaps it
                # is a dead-end, in which case the direction of movement is rotated
                # 180 degrees (moving forward turns the agents and makes it step in the previous cell)
                if not directionMatch:
                    # If cell is a dead-end, append previous node with reversed
                    # orientation!
                    nbits = 0
                    tmp = self.env.rail.get_transitions((new_cell[0], new_cell[1]))
                    while tmp > 0:
                        nbits += (tmp & 1)
                        tmp = tmp >> 1
                    if nbits == 1:
                        # Dead-end!
                        # Check if transition is possible in new_cell
                        # with orientation (direction+2)%4 in direction `direction'
                        directionMatch = directionMatch or self.env.rail.get_transition(
                            (new_cell[0], new_cell[1], (direction+2)%4), direction)

                if transitionValid and directionMatch:
                    new_distance = min(self.distance_map[target_nr,
                                                         new_cell[0], new_cell[1]], current_distance+1)
                    neighbors.append((new_cell[0], new_cell[1], direction, new_distance))
                    self.distance_map[target_nr, new_cell[0], new_cell[1]] = new_distance

        return neighbors

    def _new_position(self, position, movement):
        if movement == 0:    # NORTH
            return (position[0]-1, position[1])
        elif movement == 1:  # EAST
            return (position[0], position[1] + 1)
        elif movement == 2:  # SOUTH
            return (position[0]+1, position[1])
        elif movement == 3:  # WEST
            return (position[0], position[1] - 1)


    def get(self, handle):
        # TODO: compute the observation for agent `handle'
        return []



"""

    def get_observation(self, agent):
        # Get the current observation for an agent
        current_position = self.internal_position[agent]
        #target_heading = self._compass(agent).tolist()
        coordinate = tuple(np.transpose(self._position_to_coordinate([current_position])))
        agent_distance = self.distance_map[agent][coordinate][0]
        # Start tree search
        if current_position == self.target[agent]:
            agent_tree = Node(current_position, [-np.inf, -np.inf, -np.inf, -np.inf, -1])
        else:
            agent_tree = Node(current_position, [0, 0, 0, 0, agent_distance])

        initial_tree_state = Tree_State(agent, current_position, -1, 0, 0)
        self._tree_search(initial_tree_state, agent_tree, agent)
        observation = []
        distance_data = []

        self._flatten_tree(agent_tree, observation, distance_data,  self.max_depth+1)
        # This is probably very slow!!!!
        #max_obs = np.max([i for i in observation if i < np.inf])
        #if max_obs != 0:
        #    observation = np.array(observation)/ max_obs

        #print([i for i in distance_data if i >= 0])
        observation = np.concatenate((observation, distance_data))
        #observation = np.concatenate((observation, np.identity(5)[int(self.last_action[agent])]))
        #return np.clip(observation / self.max_dist[agent], -1, 1)
        return np.clip(observation / 15., -1, 1)




    def _tree_search(self, in_tree_state, parent_node, agent):
        if in_tree_state.depth >= self.max_depth:
            return
        target_distance = np.inf
        other_target = np.inf
        other_agent = np.inf
        coordinate = tuple(np.transpose(self._position_to_coordinate([in_tree_state.position])))
        curr_target_dist = self.distance_map[agent][coordinate][0]
        forbidden_action = (in_tree_state.direction + 2) % 4
        # Update the position
        failed_move = 0
        leaf_distance = in_tree_state.distance
        for child_idx in range(4):
            if child_idx != forbidden_action or in_tree_state.direction == -1:
                tree_state = copy.deepcopy(in_tree_state)
                tree_state.direction = child_idx
                current_position, invalid_move = self._detect_path(
                tree_state.position, tree_state.direction)
                if tree_state.initial_direction == None:
                    tree_state.initial_direction = child_idx
                if not invalid_move:
                    coordinate = tuple(np.transpose(self._position_to_coordinate([current_position])))
                    curr_target_dist = self.distance_map[agent][coordinate][0]
                    #if tree_state.initial_direction == None:
                    #    tree_state.initial_direction = child_idx
                    tree_state.position = current_position
                    tree_state.distance += 1


                    # Collect information at the current position
                    detection_distance = tree_state.distance
                    if current_position == self.target[tree_state.agent]:
                        target_distance = detection_distance

                    elif current_position in self.target:
                        other_target = detection_distance

                    if current_position in self.internal_position:
                        other_agent = detection_distance

                    tree_state.data[0] = self._min_greater_zero(target_distance, tree_state.data[0])
                    tree_state.data[1] = self._min_greater_zero(other_target, tree_state.data[1])
                    tree_state.data[2] = self._min_greater_zero(other_agent, tree_state.data[2])
                    tree_state.data[3] = tree_state.distance
                    tree_state.data[4] = self._min_greater_zero(curr_target_dist, tree_state.data[4])

                    if self._switch_detection(tree_state.position):
                        tree_state.depth += 1
                        new_tree_state = copy.deepcopy(tree_state)
                        new_node = parent_node.insert(tree_state.position,
                         tree_state.data, tree_state.initial_direction)
                        new_tree_state.initial_direction = None
                        new_tree_state.data = [np.inf, np.inf, np.inf, np.inf, np.inf]
                        self._tree_search(new_tree_state, new_node, agent)
                    else:
                        self._tree_search(tree_state, parent_node, agent)
                else:
                    failed_move += 1
            if failed_move == 3 and in_tree_state.direction != -1:
                tree_state.data[4] = self._min_greater_zero(curr_target_dist, tree_state.data[4])
                parent_node.insert(tree_state.position, tree_state.data, tree_state.initial_direction)
                return
        return

    def _flatten_tree(self, node, observation_vector, distance_sensor, depth):
        if depth <= 0:
            return
        if node != None:
            observation_vector.extend(node.data[:-1])
            distance_sensor.extend([node.data[-1]])
        else:
            observation_vector.extend([-np.inf, -np.inf, -np.inf, -np.inf])
            distance_sensor.extend([-np.inf])
        for child_idx in range(4):
            if node != None:
                child = node.children[child_idx]
            else:
                child = None
            self._flatten_tree(child, observation_vector, distance_sensor,  depth -1)



    def _switch_detection(self, position):
        # Hack to detect switches
        # This can later directly be derived from the transition matrix
        paths = 0
        for i in range(4):
            _, invalid_move = self._detect_path(position, i)
            if not invalid_move:
                paths +=1
            if paths >= 3:
                return True
        return False




    def _min_greater_zero(self, x, y):
        if x <= 0 and y <= 0:
            return 0
        if x < 0:
            return y
        if y < 0:
            return x
        return min(x, y)



"""


class Tree_State:
    """
    Keep track of the current state while building the tree
    """
    def __init__(self, agent, position, direction, depth, distance):
        self.agent = agent
        self.position = position
        self.direction = direction
        self.depth = depth
        self.initial_direction = None
        self.distance = distance
        self.data = [np.inf, np.inf, np.inf, np.inf, np.inf]

class Node():
    """
    Define a tree node to get populated during search
    """
    def __init__(self, position, data):
        self.n_children = 4
        self.children = [None]*self.n_children
        self.data = data
        self.position = position

    def insert(self, position, data, child_idx):
        """
        Insert new node with data

        @param data node data object to insert
        """
        new_node = Node(position, data)
        self.children[child_idx] = new_node
        return new_node

    def print_tree(self, i=0, depth = 0):
        """
        Print tree content inorder
        """
        current_i = i
        curr_depth = depth+1
        if i < self.n_children:
            if self.children[i] != None:
                self.children[i].print_tree(depth=curr_depth)
            current_i += 1
            if self.children[i] != None:
                self.children[i].print_tree(i, depth=curr_depth)


