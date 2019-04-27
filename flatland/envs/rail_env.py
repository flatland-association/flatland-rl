"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import TreeObsForRailEnv

from flatland.core.transitions import Grid8Transitions, RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


def rail_from_manual_specifications_generator(rail_spec):
    """
    Utility to convert a rail given by manual specification as a map of tuples
    (cell_type, rotation), to a transition map with the correct 16-bit
    transitions specifications.

    Parameters
    -------
    rail_spec : list of list of tuples
        List (rows) of lists (columns) of tuples, each specifying a cell for
        the RailEnv environment as (cell_type, rotation), with rotation being
        clock-wise and in [0, 90, 180, 270].

    Returns
    -------
    function
        Generator function that always returns a GridTransitionMap object with
        the matrix of correct 16-bit bitmaps for each cell.
    """
    def generator(width, height, num_resets=0):
        t_utils = RailEnvTransitions()

        height = len(rail_spec)
        width = len(rail_spec[0])
        rail = GridTransitionMap(width=width, height=height, transitions=t_utils)

        for r in range(height):
            for c in range(width):
                cell = rail_spec[r][c]
                if cell[0] < 0 or cell[0] >= len(t_utils.transitions):
                    print("ERROR - invalid cell type=", cell[0])
                    return []
                rail.set_transitions((r, c), t_utils.rotate_transition(t_utils.transitions[cell[0]], cell[1]))

        return rail

    return generator


def rail_from_GridTransitionMap_generator(rail_map):
    """
    Utility to convert a rail given by a GridTransitionMap map with the correct
    16-bit transitions specifications.

    Parameters
    -------
    rail_map : GridTransitionMap object
        GridTransitionMap object to return when the generator is called.

    Returns
    -------
    function
        Generator function that always returns the given `rail_map' object.
    """
    def generator(width, height, num_resets=0):
        return rail_map

    return generator


def rail_from_list_of_saved_GridTransitionMap_generator(list_of_filenames):
    """
    Utility to sequentially and cyclically return GridTransitionMap-s from a list of files, on each environment reset.

    Parameters
    -------
    list_of_filenames : list
        List of filenames with the saved grids to load.

    Returns
    -------
    function
        Generator function that always returns the given `rail_map' object.
    """
    def generator(width, height, num_resets=0):
        t_utils = RailEnvTransitions()
        rail_map = GridTransitionMap(width=width, height=height, transitions=t_utils)
        rail_map.load_transition_map(list_of_filenames[num_resets % len(list_of_filenames)], override_gridsize=False)

        if rail_map.grid.dtype == np.uint64:
            rail_map.transitions = Grid8Transitions()

        return rail_map

    return generator


"""
def generate_rail_from_list_of_manual_specifications(list_of_specifications)
    def generator(width, height, num_resets=0):
        return generate_rail_from_manual_specifications(list_of_specifications)

    return generator
"""


def random_rail_generator(cell_type_relative_proportion=[1.0] * 8):
    """
    Dummy random level generator:
    - fill in cells at random in [width-2, height-2]
    - keep filling cells in among the unfilled ones, such that all transitions
      are legit;  if no cell can be filled in without violating some
      transitions, pick one among those that can satisfy most transitions
      (1,2,3 or 4), and delete (+mark to be re-filled) the cells that were
      incompatible.
    - keep trying for a total number of insertions
      (e.g., (W-2)*(H-2)*MAX_REPETITIONS ); if no solution is found, empty the
      board and try again from scratch.
    - finally pad the border of the map with dead-ends to avoid border issues.

    Dead-ends are not allowed inside the grid, only at the border; however, if
    no cell type can be inserted in a given cell (because of the neighboring
    transitions), deadends are allowed if they solve the problem. This was
    found to turn most un-genereatable levels into valid ones.

    Parameters
    -------
    width : int
        The width (number of cells) of the grid to generate.
    height : int
        The height (number of cells) of the grid to generate.

    Returns
    -------
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """

    def generator(width, height, num_resets=0):
        t_utils = RailEnvTransitions()

        transition_probability = cell_type_relative_proportion

        transitions_templates_ = []
        transition_probabilities = []
        for i in range(len(t_utils.transitions) - 1):  # don't include dead-ends
            all_transitions = 0
            for dir_ in range(4):
                trans = t_utils.get_transitions(t_utils.transitions[i], dir_)
                all_transitions |= (trans[0] << 3) | \
                                   (trans[1] << 2) | \
                                   (trans[2] << 1) | \
                                   (trans[3])

            template = [int(x) for x in bin(all_transitions)[2:]]
            template = [0] * (4 - len(template)) + template

            # add all rotations
            for rot in [0, 90, 180, 270]:
                transitions_templates_.append((template,
                                              t_utils.rotate_transition(
                                               t_utils.transitions[i],
                                               rot)))
                transition_probabilities.append(transition_probability[i])
                template = [template[-1]] + template[:-1]

        def get_matching_templates(template):
            ret = []
            for i in range(len(transitions_templates_)):
                is_match = True
                for j in range(4):
                    if template[j] >= 0 and \
                       template[j] != transitions_templates_[i][0][j]:
                        is_match = False
                        break
                if is_match:
                    ret.append((transitions_templates_[i][1], transition_probabilities[i]))
            return ret

        MAX_INSERTIONS = (width - 2) * (height - 2) * 10
        MAX_ATTEMPTS_FROM_SCRATCH = 10

        attempt_number = 0
        while attempt_number < MAX_ATTEMPTS_FROM_SCRATCH:
            cells_to_fill = []
            rail = []
            for r in range(height):
                rail.append([None] * width)
                if r > 0 and r < height - 1:
                    cells_to_fill = cells_to_fill + [(r, c) for c in range(1, width - 1)]

            num_insertions = 0
            while num_insertions < MAX_INSERTIONS and len(cells_to_fill) > 0:
                # cell = random.sample(cells_to_fill, 1)[0]
                cell = cells_to_fill[np.random.choice(len(cells_to_fill), 1)[0]]
                cells_to_fill.remove(cell)
                row = cell[0]
                col = cell[1]

                # look at its neighbors and see what are the possible transitions
                # that can be chosen from, if any.
                valid_template = [-1, -1, -1, -1]

                for el in [(0, 2, (-1, 0)),
                           (1, 3, (0, 1)),
                           (2, 0, (1, 0)),
                           (3, 1, (0, -1))]:  # N, E, S, W
                    neigh_trans = rail[row + el[2][0]][col + el[2][1]]
                    if neigh_trans is not None:
                        # select transition coming from facing direction el[1] and
                        # moving to direction el[1]
                        max_bit = 0
                        for k in range(4):
                            max_bit |= t_utils.get_transition(neigh_trans, k, el[1])

                        if max_bit:
                            valid_template[el[0]] = 1
                        else:
                            valid_template[el[0]] = 0

                possible_cell_transitions = get_matching_templates(valid_template)

                if len(possible_cell_transitions) == 0:  # NO VALID TRANSITIONS
                    # no cell can be filled in without violating some transitions
                    # can a dead-end solve the problem?
                    if valid_template.count(1) == 1:
                        for k in range(4):
                            if valid_template[k] == 1:
                                rot = 0
                                if k == 0:
                                    rot = 180
                                elif k == 1:
                                    rot = 270
                                elif k == 2:
                                    rot = 0
                                elif k == 3:
                                    rot = 90

                                rail[row][col] = t_utils.rotate_transition(int('0010000000000000', 2), rot)
                                num_insertions += 1

                                break

                    else:
                        # can I get valid transitions by removing a single
                        # neighboring cell?
                        bestk = -1
                        besttrans = []
                        for k in range(4):
                            tmp_template = valid_template[:]
                            tmp_template[k] = -1
                            possible_cell_transitions = get_matching_templates(tmp_template)
                            if len(possible_cell_transitions) > len(besttrans):
                                besttrans = possible_cell_transitions
                                bestk = k

                        if bestk >= 0:
                            # Replace the corresponding cell with None, append it
                            # to cells to fill, fill in a transition in the current
                            # cell.
                            replace_row = row - 1
                            replace_col = col
                            if bestk == 1:
                                replace_row = row
                                replace_col = col + 1
                            elif bestk == 2:
                                replace_row = row + 1
                                replace_col = col
                            elif bestk == 3:
                                replace_row = row
                                replace_col = col - 1

                            cells_to_fill.append((replace_row, replace_col))
                            rail[replace_row][replace_col] = None

                            possible_transitions, possible_probabilities = zip(*besttrans)
                            possible_probabilities = [p / sum(possible_probabilities) for p in possible_probabilities]

                            rail[row][col] = np.random.choice(possible_transitions,
                                                              p=possible_probabilities)
                            num_insertions += 1

                        else:
                            print('WARNING: still nothing!')
                            rail[row][col] = int('0000000000000000', 2)
                            num_insertions += 1
                            pass

                else:
                    possible_transitions, possible_probabilities = zip(*possible_cell_transitions)
                    possible_probabilities = [p / sum(possible_probabilities) for p in possible_probabilities]

                    rail[row][col] = np.random.choice(possible_transitions,
                                                      p=possible_probabilities)
                    num_insertions += 1

            if num_insertions == MAX_INSERTIONS:
                # Failed to generate a valid level; try again for a number of times
                attempt_number += 1
            else:
                break

        if attempt_number == MAX_ATTEMPTS_FROM_SCRATCH:
            print('ERROR: failed to generate level')

        # Finally pad the border of the map with dead-ends to avoid border issues;
        # at most 1 transition in the neigh cell
        for r in range(height):
            # Check for transitions coming from [r][1] to WEST
            max_bit = 0
            neigh_trans = rail[r][1]
            if neigh_trans is not None:
                for k in range(4):
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2**4 - 1)
                    max_bit = max_bit | (neigh_trans_from_direction & 1)
            if max_bit:
                rail[r][0] = t_utils.rotate_transition(int('0010000000000000', 2), 270)
            else:
                rail[r][0] = int('0000000000000000', 2)

            # Check for transitions coming from [r][-2] to EAST
            max_bit = 0
            neigh_trans = rail[r][-2]
            if neigh_trans is not None:
                for k in range(4):
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2**4 - 1)
                    max_bit = max_bit | (neigh_trans_from_direction & (1 << 2))
            if max_bit:
                rail[r][-1] = t_utils.rotate_transition(int('0010000000000000', 2),
                                                        90)
            else:
                rail[r][-1] = int('0000000000000000', 2)

        for c in range(width):
            # Check for transitions coming from [1][c] to NORTH
            max_bit = 0
            neigh_trans = rail[1][c]
            if neigh_trans is not None:
                for k in range(4):
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2**4 - 1)
                    max_bit = max_bit | (neigh_trans_from_direction & (1 << 3))
            if max_bit:
                rail[0][c] = int('0010000000000000', 2)
            else:
                rail[0][c] = int('0000000000000000', 2)

            # Check for transitions coming from [-2][c] to SOUTH
            max_bit = 0
            neigh_trans = rail[-2][c]
            if neigh_trans is not None:
                for k in range(4):
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2**4 - 1)
                    max_bit = max_bit | (neigh_trans_from_direction & (1 << 1))
            if max_bit:
                rail[-1][c] = t_utils.rotate_transition(int('0010000000000000', 2), 180)
            else:
                rail[-1][c] = int('0000000000000000', 2)

        # For display only, wrong levels
        for r in range(height):
            for c in range(width):
                if rail[r][c] is None:
                    rail[r][c] = int('0000000000000000', 2)

        tmp_rail = np.asarray(rail, dtype=np.uint16)

        return_rail = GridTransitionMap(width=width, height=height, transitions=t_utils)
        return_rail.grid = tmp_rail
        return return_rail

    return generator


class RailEnv(Environment):
    """
    RailEnv environment class.

    RailEnv is an environment inspired by a (simplified version of) a rail
    network, in which agents (trains) have to navigate to their target
    locations in the shortest time possible, while at the same time cooperating
    to avoid bottlenecks.

    The valid actions in the environment are:
        0: do nothing
        1: turn left and move to the next cell
        2: move to the next cell in front of the agent
        3: turn right and move to the next cell

    Moving forward in a dead-end cell makes the agent turn 180 degrees and step
    to the cell it came from.

    The actions of the agents are executed in order of their handle to prevent
    deadlocks and to allow them to learn relative priorities.

    TODO: WRITE ABOUT THE REWARD FUNCTION, and possibly allow for alpha and
    beta to be passed as parameters to __init__().
    """

    def __init__(self,
                 width,
                 height,
                 rail_generator=random_rail_generator(),
                 number_of_agents=1,
                 obs_builder_object=TreeObsForRailEnv(max_depth=2)):
        """
        Environment init.

        Parameters
        -------
        rail_generator : function
            The rail_generator function is a function that takes the width and
            height of a  rail map along with the number of times the env has
            been reset, and returns a GridTransitionMap object.
            Implemented functions are:
                random_rail_generator : generate a random rail of given size
                rail_from_GridTransitionMap_generator(rail_map) : generate a rail from
                                        a GridTransitionMap object
                rail_from_manual_specifications_generator(rail_spec) : generate a rail from
                                        a rail specifications array
                TODO: generate_rail_from_saved_list or from list of ndarray bitmaps ---
        width : int
            The width of the rail map. Potentially in the future,
            a range of widths to sample from.
        height : int
            The height of the rail map. Potentially in the future,
            a range of heights to sample from.
        number_of_agents : int
            Number of agents to spawn on the map. Potentially in the future,
            a range of number of agents to sample from.
        obs_builder_object: ObservationBuilder object
            ObservationBuilder-derived object that takes builds observation
            vectors for each agent.
        """

        self.rail_generator = rail_generator
        self.rail = None
        self.width = width
        self.height = height

        self.number_of_agents = number_of_agents

        self.obs_builder = obs_builder_object
        self.obs_builder._set_env(self)

        self.actions = [0] * self.number_of_agents
        self.rewards = [0] * self.number_of_agents
        self.done = False

        self.dones = {"__all__": False}
        self.obs_dict = {}
        self.rewards_dict = {}

        self.agents_handles = list(range(self.number_of_agents))

        # self.agents_position = []
        # self.agents_target = []
        # self.agents_direction = []
        self.num_resets = 0
        self.reset()
        self.num_resets = 0

        self.valid_positions = None

    def get_agent_handles(self):
        return self.agents_handles

    def fill_valid_positions(self):
        self.valid_positions = valid_positions = []
        for r in range(self.height):
            for c in range(self.width):
                if self.rail.get_transitions((r, c)) > 0:
                    valid_positions.append((r, c))

    def check_agent_lists(self):
        for lAgents, name in zip(
                [self.agents_handles, self.agents_position, self.agents_direction],
                ["handles", "positions", "directions"]):
            assert self.number_of_agents == len(lAgents), "Inconsistent agent list:"+name

    def check_agent_locdirpath(self, iAgent):
        valid_movements = []
        for direction in range(4):
            position = self.agents_position[iAgent]
            moves = self.rail.get_transitions((position[0], position[1], direction))
            for move_index in range(4):
                if moves[move_index]:
                    valid_movements.append((direction, move_index))

        valid_starting_directions = []
        for m in valid_movements:
            new_position = self._new_position(self.agents_position[iAgent], m[1])
            if m[0] not in valid_starting_directions and \
                    self._path_exists(new_position, m[0], self.agents_target[iAgent]):
                valid_starting_directions.append(m[0])

        if len(valid_starting_directions) == 0:
            return False

    def pick_agent_direction(self, rcPos, rcTarget):
        valid_movements = []
        for direction in range(4):
            moves = self.rail.get_transitions((*rcPos, direction))
            for move_index in range(4):
                if moves[move_index]:
                    valid_movements.append((direction, move_index))
        # print("pos", rcPos, "targ", rcTarget, "valid movements", valid_movements)

        valid_starting_directions = []
        for m in valid_movements:
            new_position = self._new_position(rcPos, m[1])
            if m[0] not in valid_starting_directions and \
                    self._path_exists(new_position, m[0], rcTarget):
                valid_starting_directions.append(m[0])

        if len(valid_starting_directions) == 0:
            return None
        else:
            return valid_starting_directions[np.random.choice(len(valid_starting_directions), 1)[0]]

    def add_agent(self, rcPos=None, rcTarget=None, iDir=None):
        self.check_agent_lists()

        if rcPos is None:
            rcPos = np.random.choice(len(self.valid_positions))

        iAgent = self.number_of_agents
        
        self.agents_position.append(tuple(rcPos))  # ensure it's a tuple not a list
        self.agents_handles.append(max(self.agents_handles + [-1]) + 1)  # max(handles) + 1, starting at 0
        self.agents_direction.append(0)
        self.agents_target.append(rcPos)  # set the target to the origin initially
        self.number_of_agents += 1
        self.check_agent_lists()
        return iAgent
    
    def reset(self, regen_rail=True, replace_agents=True):
        if regen_rail or self.rail is None:
            self.rail = self.rail_generator(self.width, self.height, self.num_resets)
            self.fill_valid_positions()

        self.num_resets += 1

        self.dones = {"__all__": False}
        for handle in self.agents_handles:
            self.dones[handle] = False

        # Use a TreeObsForRailEnv to compute distance maps to each agent's target, to sample initial
        # agent's orientations that allow a valid solution.

        self.fill_valid_positions()

        if replace_agents:
            re_generate = True
            while re_generate:

                # self.agents_position = random.sample(valid_positions,
                #                                     self.number_of_agents)
                self.agents_position = [
                    self.valid_positions[i] for i in
                    np.random.choice(len(self.valid_positions), self.number_of_agents)]
                self.agents_target = [
                    self.valid_positions[i] for i in
                    np.random.choice(len(self.valid_positions), self.number_of_agents)]

                # agents_direction must be a direction for which a solution is
                # guaranteed.
                self.agents_direction = [0] * self.number_of_agents
                re_generate = False

                for i in range(self.number_of_agents):
                    direction = self.pick_agent_direction(self.agents_position[i], self.agents_target[i])
                    if direction is None:
                        re_generate = True
                        break
                    else:
                        self.agents_direction[i] = direction

                # Jeremy extracted this into the method pick_agent_direction
                if False:
                    for i in range(self.number_of_agents):
                        valid_movements = []
                        for direction in range(4):
                            position = self.agents_position[i]
                            moves = self.rail.get_transitions((position[0], position[1], direction))
                            for move_index in range(4):
                                if moves[move_index]:
                                    valid_movements.append((direction, move_index))

                        valid_starting_directions = []
                        for m in valid_movements:
                            new_position = self._new_position(self.agents_position[i], m[1])
                            if m[0] not in valid_starting_directions and \
                                    self._path_exists(new_position, m[0], self.agents_target[i]):
                                valid_starting_directions.append(m[0])

                        if len(valid_starting_directions) == 0:
                            re_generate = True
                        else:
                            self.agents_direction[i] = valid_starting_directions[
                                np.random.choice(len(valid_starting_directions), 1)[0]]

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()

        # Return the new observation vectors for each agent
        return self._get_observations()

    def step(self, action_dict):
        alpha = 1.0
        beta = 1.0

        invalid_action_penalty = -2
        step_penalty = -1 * alpha
        global_reward = 1 * beta

        # Reset the step rewards
        self.rewards_dict = dict()
        for handle in self.agents_handles:
            self.rewards_dict[handle] = 0

        if self.dones["__all__"]:
            return self._get_observations(), self.rewards_dict, self.dones, {}

        for i in range(len(self.agents_handles)):
            handle = self.agents_handles[i]

            if handle not in action_dict:
                continue

            if self.dones[handle]:
                continue
            action = action_dict[handle]

            if action < 0 or action > 3:
                print('ERROR: illegal action=', action,
                      'for agent with handle=', handle)
                return

            if action > 0:
                pos = self.agents_position[i]
                direction = self.agents_direction[i]

                movement = direction
                if action == 1:
                    movement = direction - 1
                elif action == 3:
                    movement = direction + 1

                if movement < 0:
                    movement += 4
                if movement >= 4:
                    movement -= 4

                is_deadend = False
                if action == 2:
                    # compute number of possible transitions in the current
                    # cell
                    nbits = 0
                    tmp = self.rail.get_transitions((pos[0], pos[1]))
                    while tmp > 0:
                        nbits += (tmp & 1)
                        tmp = tmp >> 1
                    if nbits == 1:
                        # dead-end;  assuming the rail network is consistent,
                        # this should match the direction the agent has come
                        # from. But it's better to check in any case.
                        reverse_direction = 0
                        if direction == 0:
                            reverse_direction = 2
                        elif direction == 1:
                            reverse_direction = 3
                        elif direction == 2:
                            reverse_direction = 0
                        elif direction == 3:
                            reverse_direction = 1

                        valid_transition = self.rail.get_transition(
                            (pos[0], pos[1], direction),
                            reverse_direction)
                        if valid_transition:
                            direction = reverse_direction
                            movement = reverse_direction
                            is_deadend = True
                new_position = self._new_position(pos, movement)
                # Is it a legal move?  1) transition allows the movement in the
                # cell,  2) the new cell is not empty (case 0),  3) the cell is
                # free, i.e., no agent is currently in that cell
                if new_position[1] >= self.width or\
                   new_position[0] >= self.height or\
                   new_position[0] < 0 or new_position[1] < 0:
                    new_cell_isValid = False

                elif self.rail.get_transitions((new_position[0], new_position[1])) > 0:
                    new_cell_isValid = True
                else:
                    new_cell_isValid = False

                transition_isValid = self.rail.get_transition(
                    (pos[0], pos[1], direction),
                    movement) or is_deadend

                cell_isFree = True
                for j in range(self.number_of_agents):
                    if self.agents_position[j] == new_position:
                        cell_isFree = False
                        break

                if new_cell_isValid and transition_isValid and cell_isFree:
                    # move and change direction to face the movement that was
                    # performed
                    self.agents_position[i] = new_position
                    self.agents_direction[i] = movement
                else:
                    # the action was not valid, add penalty
                    self.rewards_dict[handle] += invalid_action_penalty

            # if agent is not in target position, add step penalty
            if self.agents_position[i][0] == self.agents_target[i][0] and \
               self.agents_position[i][1] == self.agents_target[i][1]:
                self.dones[handle] = True
            else:
                self.rewards_dict[handle] += step_penalty

        # Check for end of episode + add global reward to all rewards!
        num_agents_in_target_position = 0
        for i in range(self.number_of_agents):
            if self.agents_position[i][0] == self.agents_target[i][0] and \
               self.agents_position[i][1] == self.agents_target[i][1]:
                num_agents_in_target_position += 1

        if num_agents_in_target_position == self.number_of_agents:
            self.dones["__all__"] = True
            self.rewards_dict = [r + global_reward for r in self.rewards_dict]

        # Reset the step actions (in case some agent doesn't 'register_action'
        # on the next step)
        self.actions = [0] * self.number_of_agents
        return self._get_observations(), self.rewards_dict, self.dones, {}

    def _new_position(self, position, movement):
        if movement == 0:    # NORTH
            return (position[0] - 1, position[1])
        elif movement == 1:  # EAST
            return (position[0], position[1] + 1)
        elif movement == 2:  # SOUTH
            return (position[0] + 1, position[1])
        elif movement == 3:  # WEST
            return (position[0], position[1] - 1)

    def _path_exists(self, start, direction, end):
        # BFS - Check if a path exists between the 2 nodes

        visited = set()
        stack = [(start, direction)]
        while stack:
            node = stack.pop()
            if node[0][0] == end[0] and node[0][1] == end[1]:
                return 1
            if node not in visited:
                visited.add(node)
                moves = self.rail.get_transitions((node[0][0], node[0][1], node[1]))
                for move_index in range(4):
                    if moves[move_index]:
                        stack.append((self._new_position(node[0], move_index),
                                      move_index))

                # If cell is a dead-end, append previous node with reversed
                # orientation!
                nbits = 0
                tmp = self.rail.get_transitions((node[0][0], node[0][1]))
                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1
                if nbits == 1:
                    stack.append((node[0], (node[1] + 2) % 4))

        return 0

    def _get_observations(self):
        self.obs_dict = {}
        for handle in self.agents_handles:
            self.obs_dict[handle] = self.obs_builder.get(handle)
        return self.obs_dict

    def render(self):
        # TODO:
        pass
