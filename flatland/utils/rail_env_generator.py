"""
The rail_env_generator module defines provides utilities to generate env
bitmaps for the RailEnv environment.
"""
import random
import numpy as np

from flatland.core.transitions import RailEnvTransitions
from flatland.core.transitionmap import GridTransitionMap


def generate_rail_from_manual_specifications(rail_spec):
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
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """
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
            rail.set_transitions((r, c), t_utils.rotate_transition(
                          t_utils.transitions[cell[0]], cell[1]))

    return rail


def generate_random_rail(width, height):
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

    t_utils = RailEnvTransitions()

    transitions_templates_ = []
    for i in range(len(t_utils.transitions)-1):  # don't include dead-ends
        all_transitions = 0
        for dir_ in range(4):
            trans = t_utils.get_transitions(t_utils.transitions[i], dir_)
            all_transitions |= (trans[0] << 3) | \
                               (trans[1] << 2) | \
                               (trans[2] << 1) | \
                               (trans[3])

        template = [int(x) for x in bin(all_transitions)[2:]]
        template = [0]*(4-len(template)) + template

        # add all rotations
        for rot in [0, 90, 180, 270]:
            transitions_templates_.append((template,
                                          t_utils.rotate_transition(
                                           t_utils.transitions[i],
                                           rot)))
            template = [template[-1]]+template[:-1]

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
                ret.append(transitions_templates_[i][1])
        return ret

    MAX_INSERTIONS = (width-2) * (height-2) * 10
    MAX_ATTEMPTS_FROM_SCRATCH = 10

    attempt_number = 0
    while attempt_number < MAX_ATTEMPTS_FROM_SCRATCH:
        cells_to_fill = []
        rail = []
        for r in range(height):
            rail.append([None]*width)
            if r > 0 and r < height-1:
                cells_to_fill = cells_to_fill \
                                + [(r, c) for c in range(1, width-1)]

        num_insertions = 0
        while num_insertions < MAX_INSERTIONS and len(cells_to_fill) > 0:
            cell = random.sample(cells_to_fill, 1)[0]
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
                neigh_trans = rail[row+el[2][0]][col+el[2][1]]
                if neigh_trans is not None:
                    # select transition coming from facing direction el[1] and
                    # moving to direction el[1]
                    max_bit = 0
                    for k in range(4):
                        max_bit |= \
                         t_utils.get_transition(neigh_trans, k, el[1])

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

                            rail[row][col] = t_utils.rotate_transition(
                                              int('0000000000100000', 2), rot)
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
                        possible_cell_transitions = get_matching_templates(
                                                     tmp_template)
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

                        rail[row][col] = random.sample(
                                                     besttrans, 1)[0]
                        num_insertions += 1

                    else:
                        print('WARNING: still nothing!')
                        rail[row][col] = int('0000000000000000', 2)
                        num_insertions += 1
                        pass

            else:
                rail[row][col] = random.sample(
                                             possible_cell_transitions, 1)[0]
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
                neigh_trans_from_direction = (neigh_trans >> ((3-k) * 4)) \
                                             & (2**4-1)
                max_bit = max_bit | (neigh_trans_from_direction & 1)
        if max_bit:
            rail[r][0] = t_utils.rotate_transition(
                           int('0000000000100000', 2), 270)
        else:
            rail[r][0] = int('0000000000000000', 2)

        # Check for transitions coming from [r][-2] to EAST
        max_bit = 0
        neigh_trans = rail[r][-2]
        if neigh_trans is not None:
            for k in range(4):
                neigh_trans_from_direction = (neigh_trans >> ((3-k) * 4)) \
                                             & (2**4-1)
                max_bit = max_bit | (neigh_trans_from_direction & (1 << 2))
        if max_bit:
            rail[r][-1] = t_utils.rotate_transition(int('0000000000100000', 2),
                                                    90)
        else:
            rail[r][-1] = int('0000000000000000', 2)

    for c in range(width):
        # Check for transitions coming from [1][c] to NORTH
        max_bit = 0
        neigh_trans = rail[1][c]
        if neigh_trans is not None:
            for k in range(4):
                neigh_trans_from_direction = (neigh_trans >> ((3-k) * 4)) \
                                              & (2**4-1)
                max_bit = max_bit | (neigh_trans_from_direction & (1 << 3))
        if max_bit:
            rail[0][c] = int('0000000000100000', 2)
        else:
            rail[0][c] = int('0000000000000000', 2)

        # Check for transitions coming from [-2][c] to SOUTH
        max_bit = 0
        neigh_trans = rail[-2][c]
        if neigh_trans is not None:
            for k in range(4):
                neigh_trans_from_direction = (neigh_trans >> ((3-k) * 4)) \
                                             & (2**4-1)
                max_bit = max_bit | (neigh_trans_from_direction & (1 << 1))
        if max_bit:
            rail[-1][c] = t_utils.rotate_transition(
                            int('0000000000100000', 2), 180)
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
