import warnings
from enum import IntEnum

import msgpack
import numpy as np

from flatland.core.grid.grid4_utils import get_direction, mirror
from flatland.core.grid.grid_utils import distance_on_rail
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgentStatic
from flatland.envs.grid4_generators_utils import connect_rail, connect_from_nodes, connect_nodes, connect_to_nodes
from flatland.envs.grid4_generators_utils import get_rnd_agents_pos_tgt_dir_on_rail


def empty_rail_generator():
    """
    Returns a generator which returns an empty rail mail with no agents.
    Primarily used by the editor
    """

    def generator(width, height, num_agents=0, num_resets=0):
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)

        return grid_map, [], [], [], []

    return generator


def complex_rail_generator(nr_start_goal=1, nr_extra=100, min_dist=20, max_dist=99999, seed=0):
    """
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

    def generator(width, height, num_agents, num_resets=0):

        if num_agents > nr_start_goal:
            num_agents = nr_start_goal
            print("complex_rail_generator: num_agents > nr_start_goal, changing num_agents")
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)

        np.random.seed(seed + num_resets)

        # generate rail array
        # step 1:
        # - generate a start and goal position
        #   - validate min/max distance allowed
        #   - validate that start/goals are not placed too close to other start/goals
        #   - draw a rail from [start,goal]
        #     - if rail crosses existing rail then validate new connection
        #     - possibility that this fails to create a path to goal
        #     - on failure generate new start/goal
        #
        # step 2:
        # - add more rails to map randomly between cells that have rails
        #   - validate all new rails, on failure don't add new rails
        #
        # step 3:
        # - return transition map + list of [start_pos, start_dir, goal_pos] points
        #

        start_goal = []
        start_dir = []
        nr_created = 0
        created_sanity = 0
        sanity_max = 9000
        while nr_created < nr_start_goal and created_sanity < sanity_max:
            all_ok = False
            for _ in range(sanity_max):
                start = (np.random.randint(0, height), np.random.randint(0, width))
                goal = (np.random.randint(0, height), np.random.randint(0, width))

                # check to make sure start,goal pos is empty?
                if rail_array[goal] != 0 or rail_array[start] != 0:
                    continue
                # check min/max distance
                dist_sg = distance_on_rail(start, goal)
                if dist_sg < min_dist:
                    continue
                if dist_sg > max_dist:
                    continue
                # check distance to existing points
                sg_new = [start, goal]

                def check_all_dist(sg_new):
                    for sg in start_goal:
                        for i in range(2):
                            for j in range(2):
                                dist = distance_on_rail(sg_new[i], sg[j])
                                if dist < 2:
                                    return False
                    return True

                if check_all_dist(sg_new):
                    all_ok = True
                    break

            if not all_ok:
                # we might as well give up at this point
                break

            new_path = connect_rail(rail_trans, rail_array, start, goal)
            if len(new_path) >= 2:
                nr_created += 1
                start_goal.append([start, goal])
                start_dir.append(mirror(get_direction(new_path[0], new_path[1])))
            else:
                # after too many failures we will give up
                created_sanity += 1

        # add extra connections between existing rail
        created_sanity = 0
        nr_created = 0
        while nr_created < nr_extra and created_sanity < sanity_max:
            all_ok = False
            for _ in range(sanity_max):
                start = (np.random.randint(0, height), np.random.randint(0, width))
                goal = (np.random.randint(0, height), np.random.randint(0, width))
                # check to make sure start,goal pos are not empty
                if rail_array[goal] == 0 or rail_array[start] == 0:
                    continue
                else:
                    all_ok = True
                    break
            if not all_ok:
                break
            new_path = connect_rail(rail_trans, rail_array, start, goal)
            if len(new_path) >= 2:
                nr_created += 1

        agents_position = [sg[0] for sg in start_goal[:num_agents]]
        agents_target = [sg[1] for sg in start_goal[:num_agents]]
        agents_direction = start_dir[:num_agents]

        return grid_map, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator


def rail_from_manual_specifications_generator(rail_spec):
    """
    Utility to convert a rail given by manual specification as a map of tuples
    (cell_type, rotation), to a transition map with the correct 16-bit
    transitions specifications.

    Parameters
    -------
    rail_spec : list of list of tuples
        List (rows) of lists (columns) of tuples, each specifying a rail_spec_of_cell for
        the RailEnv environment as (cell_type, rotation), with rotation being
        clock-wise and in [0, 90, 180, 270].

    Returns
    -------
    function
        Generator function that always returns a GridTransitionMap object with
        the matrix of correct 16-bit bitmaps for each rail_spec_of_cell.
    """

    def generator(width, height, num_agents, num_resets=0):
        rail_env_transitions = RailEnvTransitions()

        height = len(rail_spec)
        width = len(rail_spec[0])
        rail = GridTransitionMap(width=width, height=height, transitions=rail_env_transitions)

        for r in range(height):
            for c in range(width):
                rail_spec_of_cell = rail_spec[r][c]
                index_basic_type_of_cell_ = rail_spec_of_cell[0]
                rotation_cell_ = rail_spec_of_cell[1]
                if index_basic_type_of_cell_ < 0 or index_basic_type_of_cell_ >= len(rail_env_transitions.transitions):
                    print("ERROR - invalid rail_spec_of_cell type=", index_basic_type_of_cell_)
                    return []
                basic_type_of_cell_ = rail_env_transitions.transitions[index_basic_type_of_cell_]
                effective_transition_cell = rail_env_transitions.rotate_transition(basic_type_of_cell_, rotation_cell_)
                rail.set_transitions((r, c), effective_transition_cell)

        agents_position, agents_direction, agents_target = get_rnd_agents_pos_tgt_dir_on_rail(
            rail,
            num_agents)

        return rail, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator


def rail_from_file(filename):
    """
    Utility to load pickle file

    Parameters
    -------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    function
        Generator function that always returns a GridTransitionMap object with
        the matrix of correct 16-bit bitmaps for each rail_spec_of_cell.
    """

    def generator(width, height, num_agents, num_resets):
        rail_env_transitions = RailEnvTransitions()
        with open(filename, "rb") as file_in:
            load_data = file_in.read()
        data = msgpack.unpackb(load_data, use_list=False)

        grid = np.array(data[b"grid"])
        rail = GridTransitionMap(width=np.shape(grid)[1], height=np.shape(grid)[0], transitions=rail_env_transitions)
        rail.grid = grid
        # agents are always reset as not moving
        agents_static = [EnvAgentStatic(d[0], d[1], d[2], moving=False) for d in data[b"agents_static"]]
        # setup with loaded data
        agents_position = [a.position for a in agents_static]
        agents_direction = [a.direction for a in agents_static]
        agents_target = [a.target for a in agents_static]
        if b"distance_maps" in data.keys():
            distance_maps = data[b"distance_maps"]
            if len(distance_maps) > 0:
                return rail, agents_position, agents_direction, agents_target, [1.0] * len(
                    agents_position), distance_maps
            else:
                return rail, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)
        else:
            return rail, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator


def rail_from_grid_transition_map(rail_map):
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

    def generator(width, height, num_agents, num_resets=0):
        agents_position, agents_direction, agents_target = get_rnd_agents_pos_tgt_dir_on_rail(
            rail_map,
            num_agents)

        return rail_map, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator


def random_rail_generator(cell_type_relative_proportion=[1.0] * 11):
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

    def generator(width, height, num_agents, num_resets=0):
        t_utils = RailEnvTransitions()

        transition_probability = cell_type_relative_proportion

        transitions_templates_ = []
        transition_probabilities = []
        for i in range(len(t_utils.transitions)):  # don't include dead-ends
            if t_utils.transitions[i] == int('0010000000000000', 2):
                continue

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
                    if template[j] >= 0 and template[j] != transitions_templates_[i][0][j]:
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
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2 ** 4 - 1)
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
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2 ** 4 - 1)
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
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2 ** 4 - 1)
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
                    neigh_trans_from_direction = (neigh_trans >> ((3 - k) * 4)) & (2 ** 4 - 1)
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

        agents_position, agents_direction, agents_target = get_rnd_agents_pos_tgt_dir_on_rail(
            return_rail,
            num_agents)

        return return_rail, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator


def realistic_rail_generator(nr_start_goal=1, nr_extra=100, min_dist=20, max_dist=99999, seed=0):
    """
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



    transition_list = [int('0000000000000000', 2),  # empty cell - Case 0
                       int('1000000000100000', 2),  # Case 1 - straight
                       int('1001001000100000', 2),  # Case 2 - simple switch
                       int('1000010000100001', 2),  # Case 3 - diamond drossing
                       int('1001011000100001', 2),  # Case 4 - single slip
                       int('1100110000110011', 2),  # Case 5 - double slip
                       int('0101001000000010', 2),  # Case 6 - symmetrical
                       int('0010000000000000', 2),  # Case 7 - dead end
                       int('0100000000000010', 2),  # Case 1b (8)  - simple turn right
                       int('0001001000000000', 2),  # Case 1c (9)  - simple turn left
                       int('1100000000100010', 2)]  # Case 2b (10) - simple switch mirrored

    """
    def min_max_cut(min_v,max_v,v):
        return max(min_v,min(max_v,v))

    def add_rail(width,height,grid_map, pt_from, pt_via, pt_to, bAddRemove=True):
        gRCTrans = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC

        lrcStroke = [[min_max_cut(0,height-1,pt_from[0]),
                      min_max_cut(0,width-1,pt_from[1])],
                     [min_max_cut(0,height-1,pt_via[0]),
                      min_max_cut(0,width-1,pt_via[1])],
                     [min_max_cut(0,height-1,pt_to[0]),
                      min_max_cut(0,width-1,pt_to[1])]]

        rc3Cells = np.array(lrcStroke[:3])  # the 3 cells
        rcMiddle = rc3Cells[1]  # the middle cell which we will update
        bDeadend = np.all(lrcStroke[0] == lrcStroke[2])  # deadend means cell 0 == cell 2

        # get the 2 row, col deltas between the 3 cells, eg [[-1,0],[0,1]] = North, East
        rc2Trans = np.diff(rc3Cells, axis=0)

        # get the direction index for the 2 transitions
        liTrans = []
        for rcTrans in rc2Trans:
            # gRCTrans - rcTrans gives an array of vector differences between our rcTrans
            # and the 4 directions stored in gRCTrans.
            # Where the vector difference is zero, we have a match...
            # np.all detects where the whole row,col vector is zero.
            # argwhere gives the index of the zero vector, ie the direction index
            iTrans = np.argwhere(np.all(gRCTrans - rcTrans == 0, axis=1))
            if len(iTrans) > 0:
                iTrans = iTrans[0][0]
                liTrans.append(iTrans)

        # check that we have two transitions
        if len(liTrans) == 2:
            # Set the transition
            # Set the transition
            # If this transition spans 3 cells, it is not a deadend, so remove any deadends.
            # The user will need to resolve any conflicts.
            grid_map.set_transition((*rcMiddle, liTrans[0]),
                                    liTrans[1],
                                    bAddRemove,
                                    remove_deadends=not bDeadend)

            # Also set the reverse transition
            # use the reversed outbound transition for inbound
            # and the reversed inbound transition for outbound
            grid_map.set_transition((*rcMiddle, mirror(liTrans[1])),
                                    mirror(liTrans[0]), bAddRemove, remove_deadends=not bDeadend)

    def make_switch_w_e(width,height,grid_map, center):
        # e -> w
        start = (center[0]+1, center[1]-1)
        via = (center[0], center[1] - 1)
        goal = (center[0], center[1])
        add_rail(width,height,grid_map, start, via, goal)
        start = (center[0], center[1]-1)
        via = (center[0]+1, center[1]-1)
        goal = (center[0]+1, center[1]-2)
        add_rail(width,height,grid_map, start, via, goal)

    def make_switch_e_w(width,height,grid_map, center):
        # e -> w
        start = (center[0] + 1, center[1])
        via = (center[0] + 1, center[1] - 1)
        goal = (center[0], center[1] - 1)
        add_rail(width,height,grid_map, start, via, goal)
        start = (center[0] + 1, center[1] - 1)
        via = (center[0], center[1] - 1)
        goal = (center[0], center[1] - 2)
        add_rail(width,height,grid_map, start, via, goal)

    class Grid4TransitionsEnum(IntEnum):
        NORTH = 0
        EAST = 1
        SOUTH = 2
        WEST = 3

        @staticmethod
        def to_char(int: int):
            return {0: 'N',
                    1: 'E',
                    2: 'S',
                    3: 'W'}[int]

    def generator(width, height, num_agents, num_resets=0):

        if num_agents > nr_start_goal:
            num_agents = nr_start_goal
            print("complex_rail_generator: num_agents > nr_start_goal, changing num_agents")
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)

        np.random.seed(seed + num_resets)



        max_n_track_seg=4
        x_offsets = np.arange(0,height,max_n_track_seg).astype(int)

        agents_position = []
        agents_target = []
        agents_direction = []

        for off_set_loop in range(len(x_offsets)):
            off_set = x_offsets[off_set_loop]
            # second track
            data = np.arange(int((width -4- max_n_track_seg) / max_n_track_seg)) * max_n_track_seg + 4
            n_track_seg = np.random.choice(max_n_track_seg) + 1

            start_track = (off_set, 0)
            goal_track = (off_set, width - 1)
            new_path = connect_rail(rail_trans, rail_array, start_track, goal_track)

            # track one (full track : left right)
            if off_set_loop > 0:
                if off_set_loop % 2 == 1:
                    start_track = (x_offsets[off_set_loop-1]+1, width - 1)
                    goal_track = (x_offsets[off_set_loop]-1, width - 1)
                    new_path = connect_rail(rail_trans, rail_array, start_track, goal_track)

                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop-1], width - 2),
                             (x_offsets[off_set_loop-1], width - 1),
                             (x_offsets[off_set_loop-1]+1, width - 1))
                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop], width - 2),
                             (x_offsets[off_set_loop], width - 1),
                             (x_offsets[off_set_loop]-1, width - 1))
                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop-1], width - 1),
                             (x_offsets[off_set_loop-1]+1, width - 1),
                             (x_offsets[off_set_loop-1]+2, width - 1))
                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop], width - 1),
                             (x_offsets[off_set_loop]-1, width - 1),
                             (x_offsets[off_set_loop]-2, width - 1))

                else:
                    start_track = (x_offsets[off_set_loop-1]+1,0)
                    goal_track = (x_offsets[off_set_loop]-1, 0)
                    new_path = connect_rail(rail_trans, rail_array, start_track, goal_track)

                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop-1], 1),
                             (x_offsets[off_set_loop-1], 0),
                             (x_offsets[off_set_loop-1]+1, 0))
                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop], 1),
                             (x_offsets[off_set_loop], 0),
                             (x_offsets[off_set_loop]-1, 0))
                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop-1], 0),
                             (x_offsets[off_set_loop-1]+1, 0),
                             (x_offsets[off_set_loop-1]+2, 0))
                    add_rail(width,height,grid_map,
                             (x_offsets[off_set_loop], 0),
                             (x_offsets[off_set_loop]-1, 0),
                             (x_offsets[off_set_loop]-2, 0))

            for nbr_track_loop in range(height-1):
                if len(data) < 2*n_track_seg+1:
                    break
                x = np.sort(np.random.choice(data, 2 * n_track_seg, False)).astype(int)
                data = []
                for x_loop in range(int(len(x) / 2)):
                    start = (max(0,min(off_set+nbr_track_loop+1,height-1)), max(0,min(x[2 * x_loop],width-1)))
                    goal = (max(0,min(off_set+nbr_track_loop+1,height-1)), max(0,min(x[2 * x_loop + 1],width-1)))
                    d = np.arange(x[2 * x_loop]+1,x[2 * x_loop+1]-1,2)
                    data.extend(d)

                    new_path = connect_rail(rail_trans, rail_array, start, goal)
                    if len(new_path) >0:
                        c = (off_set+nbr_track_loop, x[2 * x_loop] + 1)
                        make_switch_e_w(width,height,grid_map, c)
                        c = (off_set+nbr_track_loop, x[2 * x_loop + 1]+1)
                        make_switch_w_e(width,height,grid_map, c)


                    add_pos = (int((start[0]+goal[0])/2),int((start[1]+goal[1])/2))
                    agents_position.append(add_pos)
                    agents_target.append(add_pos)
                    agents_direction.append(np.random.choice([3,1]))

        print(agents_direction)
        print(agents_position)
        print(agents_target)

        x = np.arange(len(agents_direction))
        num_a = min(num_agents,np.floor(len(agents_direction)/2))
        if num_a > 1:
            filter_agent = np.random.choice(x,num_a,False)
            agents_position = agents_position[filter_agent]
            agents_direction = agents_direction[filter_agent]
            np.delete(x,filter_agent)
            filter_agent = np.random.choice(x,num_a,False)
            agents_target = agents_target[filter_agent]

        return grid_map, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator


def sparse_rail_generator(nr_nodes=100, max_neigbours=2, min_node_dist=20, node_radius=2,
                          seed=0):
    '''

    :param nr_train_stations:
    :param nr_nodes:
    :param mean_node_neighbours:
    :param min_node_dist:
    :param seed:
    :return:
    '''

    def generator(width, height, num_agents, num_resets=0):
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)
        np.random.seed(seed + num_resets)

        # Generate a set of nodes for the sparse network
        node_positions = []
        for node_idx in range(nr_nodes):
            to_close = True
            tries = 0
            while to_close:
                x_tmp = np.random.randint(height)
                y_tmp = np.random.randint(width)
                to_close = False
                for node_pos in node_positions:
                    if distance_on_rail((x_tmp, y_tmp), node_pos) < min_node_dist:
                        to_close = True
                if not to_close:
                    node_positions.append((x_tmp, y_tmp))
                tries += 1
                if tries > 100:
                    warnings.warn("Could not set nodes, please change initial parameters!!!!")
                    break

        # Chose node connection
        available_nodes = np.arange(nr_nodes)
        current_node = 0
        node_stack = [current_node]

        while len(node_stack) > 0:
            current_node = node_stack[0]
            delete_idx = np.where(available_nodes == current_node)
            available_nodes = np.delete(available_nodes, delete_idx, 0)

            # Get random number of neighbors
            num_neighb = 2  # np.random.randint(1, max_neigbours)
            if len(available_nodes) >= num_neighb:
                connected_neighb_idx = np.random.choice(available_nodes, num_neighb, replace=False)
            else:
                connected_neighb_idx = available_nodes

            for neighb in connected_neighb_idx:
                if neighb not in node_stack:
                    node_stack.append(neighb)
                new_path = connect_nodes(rail_trans, rail_array, node_positions[current_node], node_positions[neighb])
            node_stack.pop(0)



        # Generate start and target node directory for all agents
        agent_start_targets_nodes = []
        for agent_idx in range(num_agents):
            start_target_tuple = np.random.choice(nr_nodes, 2, replace=False)
            agent_start_targets_nodes.append(start_target_tuple)

        # Generate actual start and target locations from around nodes
        agents_position = []
        agents_target = []
        agents_direction = []
        agent_idx = 0
        for start_target in agent_start_targets_nodes:
            start_x = np.clip(node_positions[start_target[0]][0] + np.random.randint(-node_radius, node_radius), 0,
                              height - 1)
            start_y = np.clip(node_positions[start_target[0]][1] + np.random.randint(-node_radius, node_radius), 0,
                              width - 1)
            target_x = np.clip(node_positions[start_target[1]][0] + np.random.randint(-node_radius, node_radius), 0,
                               height - 1)
            target_y = np.clip(node_positions[start_target[1]][1] + np.random.randint(-node_radius, node_radius), 0,
                               width - 1)
            # Make sure we don't put to starts or targets on same cell
            while (start_x, start_y) in agents_position or (start_x, start_y) == node_positions[start_target[0]]:
                start_x = np.clip(node_positions[start_target[0]][0] + np.random.randint(-node_radius, node_radius),
                                  0,
                                  height - 1)
                start_y = np.clip(node_positions[start_target[0]][1] + np.random.randint(-node_radius, node_radius),
                                  0,
                                  width - 1)
            while (target_x, target_y) in agents_target or (target_x, target_y) == node_positions[start_target[1]] or \
                rail_array[(target_x, target_y)] != 0:
                target_x = np.clip(
                    node_positions[start_target[1]][0] + np.random.randint(-node_radius, node_radius), 0,
                    height - 1)
                target_y = np.clip(
                    node_positions[start_target[1]][1] + np.random.randint(-node_radius, node_radius), 0,
                    width - 1)
            agents_position.append((start_x, start_y))
            agents_target.append((target_x, target_y))
            new_path = connect_to_nodes(rail_trans, rail_array, agents_position[agent_idx],
                                        node_positions[start_target[0]])
            new_path = connect_from_nodes(rail_trans, rail_array, node_positions[start_target[1]],
                                          agents_target[agent_idx])
            agents_direction.append(0)
            agent_idx += 1
        return grid_map, agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator
