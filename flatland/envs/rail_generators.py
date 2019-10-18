"""Rail generators (infrastructure manager, "Infrastrukturbetreiber")."""
import warnings
from typing import Callable, Tuple, Optional, Dict, List

import msgpack
import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.grid_utils import distance_on_rail, IntVector2DArray, IntVector2D, \
    Vec2dOperations
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes

RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]


def empty_rail_generator() -> RailGenerator:
    """
    Returns a generator which returns an empty rail mail with no agents.
    Primarily used by the editor
    """

    def generator(width: int, height: int, num_agents: int = 0, num_resets: int = 0) -> RailGeneratorProduct:
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)

        return grid_map, None

    return generator


def complex_rail_generator(nr_start_goal=1,
                           nr_extra=100,
                           min_dist=20,
                           max_dist=99999,
                           seed=1) -> RailGenerator:
    """
    complex_rail_generator

    Parameters
    ----------
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
        grid_map = GridTransitionMap(width=width, height=height, transitions=RailEnvTransitions())
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

        rail_trans = grid_map.transitions
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

            new_path = connect_rail_in_grid_map(grid_map, start, goal, rail_trans, Vec2d.get_chebyshev_distance,
                                                flip_start_node_trans=True, flip_end_node_trans=True,
                                                respect_transition_validity=True, forbidden_cells=None)
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
            new_path = connect_rail_in_grid_map(grid_map, start, goal, rail_trans, Vec2d.get_chebyshev_distance,
                                                flip_start_node_trans=True, flip_end_node_trans=True,
                                                respect_transition_validity=True, forbidden_cells=None)

            if len(new_path) >= 2:
                nr_created += 1

        return grid_map, {'agents_hints': {
            'start_goal': start_goal,
            'start_dir': start_dir
        }}

    return generator


def rail_from_manual_specifications_generator(rail_spec):
    """
    Utility to convert a rail given by manual specification as a map of tuples
    (cell_type, rotation), to a transition map with the correct 16-bit
    transitions specifications.

    Parameters
    ----------
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

        return [rail, None]

    return generator


def rail_from_file(filename, load_from_package=None) -> RailGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    filename : Pickle file generated by env.save() or editor

    Returns
    -------
    function
        Generator function that always returns a GridTransitionMap object with
        the matrix of correct 16-bit bitmaps for each rail_spec_of_cell.
    """

    def generator(width, height, num_agents, num_resets):
        rail_env_transitions = RailEnvTransitions()
        if load_from_package is not None:
            from importlib_resources import read_binary
            load_data = read_binary(load_from_package, filename)
        else:
            with open(filename, "rb") as file_in:
                load_data = file_in.read()
        data = msgpack.unpackb(load_data, use_list=False)

        grid = np.array(data[b"grid"])
        rail = GridTransitionMap(width=np.shape(grid)[1], height=np.shape(grid)[0], transitions=rail_env_transitions)
        rail.grid = grid
        if b"distance_map" in data.keys():
            distance_map = data[b"distance_map"]
            if len(distance_map) > 0:
                return rail, {'distance_map': distance_map}
        return [rail, None]

    return generator


def rail_from_grid_transition_map(rail_map) -> RailGenerator:
    """
    Utility to convert a rail given by a GridTransitionMap map with the correct
    16-bit transitions specifications.

    Parameters
    ----------
    rail_map : GridTransitionMap object
        GridTransitionMap object to return when the generator is called.

    Returns
    -------
    function
        Generator function that always returns the given `rail_map` object.
    """

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0) -> RailGeneratorProduct:
        return rail_map, None

    return generator


def random_rail_generator(cell_type_relative_proportion=[1.0] * 11, seed=1) -> RailGenerator:
    """
    Dummy random level generator:
    - fill in cells at random in [width-2, height-2]
    - keep filling cells in among the unfilled ones, such that all transitions\
      are legit;  if no cell can be filled in without violating some\
      transitions, pick one among those that can satisfy most transitions\
      (1,2,3 or 4), and delete (+mark to be re-filled) the cells that were\
      incompatible.
    - keep trying for a total number of insertions\
      (e.g., (W-2)*(H-2)*MAX_REPETITIONS ); if no solution is found, empty the\
      board and try again from scratch.
    - finally pad the border of the map with dead-ends to avoid border issues.

    Dead-ends are not allowed inside the grid, only at the border; however, if
    no cell type can be inserted in a given cell (because of the neighboring
    transitions), deadends are allowed if they solve the problem. This was
    found to turn most un-genereatable levels into valid ones.

    Parameters
    ----------
    width : int
        The width (number of cells) of the grid to generate.
    height : int
        The height (number of cells) of the grid to generate.

    Returns
    -------
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0) -> RailGeneratorProduct:
        np.random.seed(seed + num_resets)
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

        return return_rail, None

    return generator


def sparse_rail_generator(max_num_cities: int = 5, grid_mode: bool = False, max_rails_between_cities: int = 4,
                          max_rails_in_city: int = 4, seed: int = 1) -> RailGenerator:
    """
    Generates railway networks with cities and inner city rails
    :param max_num_cities: Number of city centers in the map
    :param grid_mode: arrange cities in a grid or randomly
    :param max_rails_between_cities: Maximum number of connecting rails going out from a city
    :param max_rails_in_city: maximum number of internal rails
    :param seed: Random seed to initiate rail
    :return: generator
    """

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0) -> RailGeneratorProduct:
        np.random.seed(seed + num_resets)

        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        city_radius = int(np.ceil((max_rails_in_city + 2) / 2.0)) + 2
        vector_field = np.zeros(shape=(height, width)) - 1.

        min_nr_rails_in_city = 2
        # max_nr_rail_in_city = 6
        rails_in_city = min_nr_rails_in_city if max_rails_in_city < min_nr_rails_in_city else max_rails_in_city
        rails_between_cities = rails_in_city if max_rails_between_cities > rails_in_city else max_rails_between_cities

        # Evenly distribute cities
        if grid_mode:
            city_positions, city_cells = _generate_evenly_distr_city_positions(max_num_cities, city_radius, width,
                                                                               height, vector_field)
        else:
            city_positions, city_cells = _generate_random_city_positions(max_num_cities, city_radius, width, height,
                                                                         vector_field)

        # reduce num_cities if less were generated in random mode
        num_cities = len(city_positions)

        # Try with evenly distributed cities
        if num_cities < 2:
            city_positions, city_cells = _generate_evenly_distr_city_positions(max_num_cities, city_radius, width,
                                                                               height, vector_field)
        num_cities = len(city_positions)

        # Fail
        if num_cities < 2:
            warnings.warn("Initial parameters cannot generate valid railway")
            return
        # Set up connection points for all cities
        inner_connection_points, outer_connection_points, connection_info, city_orientations = \
            _generate_city_connection_points(
                city_positions, city_radius, rails_between_cities,
                rails_in_city)

        # Connect the cities through the connection points
        inter_city_lines = _connect_cities(city_positions, outer_connection_points, city_cells,
                                           rail_trans, grid_map)

        # Build inner cities
        free_rails = _build_inner_cities(city_positions, inner_connection_points,
                                         outer_connection_points,
                                         rail_trans,
                                         grid_map)

        # Populate cities
        train_stations = _set_trainstation_positions(city_positions, city_radius, free_rails)
        # Fix all transition elements
        _fix_transitions(city_cells, inter_city_lines, grid_map, vector_field, rail_trans)
        # Generate start target pairs
        agent_start_targets_cities = _generate_start_target_pairs(num_agents, num_cities, train_stations,
                                                                  city_orientations)
        return grid_map, {'agents_hints': {
            'num_agents': num_agents,
            'agent_start_targets_cities': agent_start_targets_cities,
            'train_stations': train_stations,
            'city_orientations': city_orientations
        }}

    def _generate_random_city_positions(num_cities: int, city_radius: int, width: int,
                                        height: int, vector_field) -> (IntVector2DArray, IntVector2DArray):
        city_positions: IntVector2DArray = []
        city_cells: IntVector2DArray = []
        for city_idx in range(num_cities):
            too_close = True
            tries = 0

            while too_close:
                row = city_radius + 1 + np.random.randint(height - 2 * (city_radius + 1))
                col = city_radius + 1 + np.random.randint(width - 2 * (city_radius + 1))
                too_close = False
                # Check distance to cities
                for city_pos in city_positions:
                    if _are_cities_overlapping((row, col), city_pos, 2 * (city_radius + 1) + 1):
                        too_close = True

                if not too_close:
                    city_positions.append((row, col))
                    city_cells.extend(_get_cells_in_city(city_positions[-1], city_radius, vector_field))

                tries += 1
                if tries > 200:
                    warnings.warn(
                        "Could not set all required cities!")
                    break
        return city_positions, city_cells

    def _generate_evenly_distr_city_positions(num_cities: int, city_radius: int, width: int, height: int,
                                              vector_field) -> (IntVector2DArray, IntVector2DArray):
        aspect_ratio = height / width

        cities_per_row = min(int(np.ceil(np.sqrt(num_cities * aspect_ratio))),
                             int((height - 2) / (2 * (city_radius + 1))))
        cities_per_col = min(int(np.ceil(num_cities / cities_per_row)),
                             int((width - 2) / (2 * (city_radius + 1))))
        num_build_cities = min(num_cities, cities_per_col * cities_per_row)
        row_positions = np.linspace(city_radius + 2, height - (city_radius + 2), cities_per_row, dtype=int)
        col_positions = np.linspace(city_radius + 2, width - (city_radius + 2), cities_per_col, dtype=int)
        city_positions = []
        city_cells = []
        for city_idx in range(num_build_cities):
            row = row_positions[city_idx % cities_per_row]
            col = col_positions[city_idx // cities_per_row]
            city_positions.append((row, col))
            city_cells.extend(_get_cells_in_city(city_positions[-1], city_radius, vector_field))
        return city_positions, city_cells

    def _generate_city_connection_points(city_positions: IntVector2DArray, city_radius: int, rails_between_cities: int,
                                         rails_in_city: int = 2) -> (List[List[List[IntVector2D]]],
                                                                     List[List[List[IntVector2D]]],
                                                                     List[np.ndarray],
                                                                     List[Grid4TransitionsEnum]):
        inner_connection_points: List[List[List[IntVector2D]]] = []
        outer_connection_points: List[List[List[IntVector2D]]] = []
        connection_info: List[np.ndarray] = []
        city_orientations: List[Grid4TransitionsEnum] = []
        for city_position in city_positions:

            # Chose the directions where close cities are situated
            neighb_dist = []
            for neighbour_city in city_positions:
                neighb_dist.append(Vec2dOperations.get_manhattan_distance(city_position, neighbour_city))
            closest_neighb_idx = argsort(neighb_dist)

            # Store the directions to these neighbours and orient city to face closest neighbour
            connection_sides_idx = []
            idx = 1
            if grid_mode:
                current_closest_direction = np.random.randint(4)
            else:
                current_closest_direction = direction_to_point(city_position, city_positions[closest_neighb_idx[idx]])
            connection_sides_idx.append(current_closest_direction)
            connection_sides_idx.append((current_closest_direction + 2) % 4)
            city_orientations.append(current_closest_direction)
            # set the number of tracks within a city, at least 2 tracks per city
            connections_per_direction = np.zeros(4, dtype=int)
            nr_of_connection_points = np.random.randint(2, rails_in_city + 1)
            for idx in connection_sides_idx:
                connections_per_direction[idx] = nr_of_connection_points
            connection_points_coordinates_inner: List[List[IntVector2D]] = [[] for i in range(4)]
            connection_points_coordinates_outer: List[List[IntVector2D]] = [[] for i in range(4)]
            number_of_out_rails = np.random.randint(1, min(rails_between_cities, nr_of_connection_points) + 1)
            start_idx = int((nr_of_connection_points - number_of_out_rails) / 2)
            for direction in range(4):
                connection_slots = np.arange(nr_of_connection_points) - start_idx
                offset_distances = np.arange(nr_of_connection_points) - int(nr_of_connection_points / 2)
                inner_point_offset = np.abs(offset_distances) + np.clip(offset_distances, 0, 1) + 1

                for connection_idx in range(connections_per_direction[direction]):
                    if direction == 0:
                        tmp_coordinates = (
                            city_position[0] - city_radius + inner_point_offset[connection_idx],
                            city_position[1] + connection_slots[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] - city_radius, city_position[1] + connection_slots[connection_idx])
                    if direction == 1:
                        tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx],
                            city_position[1] + city_radius - inner_point_offset[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx], city_position[1] + city_radius)
                    if direction == 2:
                        tmp_coordinates = (
                            city_position[0] + city_radius - inner_point_offset[connection_idx],
                            city_position[1] + connection_slots[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] + city_radius, city_position[1] + connection_slots[connection_idx])
                    if direction == 3:
                        tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx],
                            city_position[1] - city_radius + inner_point_offset[connection_idx])
                        out_tmp_coordinates = (
                            city_position[0] + connection_slots[connection_idx], city_position[1] - city_radius)
                    connection_points_coordinates_inner[direction].append(tmp_coordinates)
                    if connection_idx in range(start_idx, start_idx + number_of_out_rails):
                        connection_points_coordinates_outer[direction].append(out_tmp_coordinates)

            inner_connection_points.append(connection_points_coordinates_inner)
            outer_connection_points.append(connection_points_coordinates_outer)
            connection_info.append(connections_per_direction)
        return inner_connection_points, outer_connection_points, connection_info, city_orientations

    def _connect_cities(city_positions: IntVector2DArray, connection_points: List[List[List[IntVector2D]]],
                        city_cells: IntVector2DArray,
                        rail_trans: RailEnvTransitions, grid_map: GridTransitionMap) -> List[IntVector2DArray]:
        """
        Function to connect the different cities through their connection points
        :param city_positions: Positions of city centers
        :param connection_points: Boarder connection points of cities
        :param rail_trans: Transitions
        :param grid_map: Grid map
        :return:
        """
        all_paths: List[IntVector2DArray] = []

        grid4_directions = [Grid4TransitionsEnum.NORTH, Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.SOUTH,
                            Grid4TransitionsEnum.WEST]

        for current_city_idx in np.arange(len(city_positions)):
            closest_neighbours = _closest_neighbour_in_grid4_directions(current_city_idx, city_positions)
            for out_direction in grid4_directions:

                neighbour_idx = get_closest_neighbour_for_direction(closest_neighbours, out_direction)

                for city_out_connection_point in connection_points[current_city_idx][out_direction]:

                    min_connection_dist = np.inf
                    for direction in grid4_directions:
                        current_points = connection_points[neighbour_idx][direction]
                        for tmp_in_connection_point in current_points:
                            tmp_dist = Vec2dOperations.get_manhattan_distance(city_out_connection_point,
                                                                              tmp_in_connection_point)
                            if tmp_dist < min_connection_dist:
                                min_connection_dist = tmp_dist
                                neighbour_connection_point = tmp_in_connection_point

                    new_line = connect_rail_in_grid_map(grid_map, city_out_connection_point, neighbour_connection_point,
                                                        rail_trans, flip_start_node_trans=False,
                                                        flip_end_node_trans=False, respect_transition_validity=False,
                                                        avoid_rail=True,
                                                        forbidden_cells=city_cells)
                    all_paths.extend(new_line)

        return all_paths

    def get_closest_neighbour_for_direction(closest_neighbours, out_direction):
        neighbour_idx = closest_neighbours[out_direction]
        if neighbour_idx is not None:
            return neighbour_idx

        neighbour_idx = closest_neighbours[(out_direction - 1) % 4]  # counter-clockwise
        if neighbour_idx is not None:
            return neighbour_idx

        neighbour_idx = closest_neighbours[(out_direction + 1) % 4]  # clockwise
        if neighbour_idx is not None:
            return neighbour_idx

        return closest_neighbours[(out_direction + 2) % 4]  # clockwise

    def _build_inner_cities(city_positions: IntVector2DArray, inner_connection_points: List[List[List[IntVector2D]]],
                            outer_connection_points: List[List[List[IntVector2D]]], rail_trans: RailEnvTransitions,
                            grid_map: GridTransitionMap) -> (List[IntVector2DArray], List[List[List[IntVector2D]]]):
        """
        Builds inner city tracks. This current version connects all incoming connections to all outgoing connections
        :param city_positions: Positions of the cities
        :param inner_connection_points: Points on city boarder that are used to generate inner city track
        :param outer_connection_points: Points where the city is connected to neighboring cities
        :param rail_trans:
        :param grid_map:
        :return: Returns the cells of the through path which cannot be occupied by trainstations
        """
        free_rails: List[List[List[IntVector2D]]] = [[] for i in range(len(city_positions))]
        for current_city in range(len(city_positions)):

            # This part only works if we have keep same number of connection points for both directions
            # Also only works with two connection direction at each city
            for i in range(4):
                if len(inner_connection_points[current_city][i]) > 0:
                    boarder = i
                    break

            opposite_boarder = (boarder + 2) % 4
            nr_of_connection_points = len(inner_connection_points[current_city][boarder])
            number_of_out_rails = len(outer_connection_points[current_city][boarder])
            start_idx = int((nr_of_connection_points - number_of_out_rails) / 2)
            # Connect parallel tracks
            for track_id in range(nr_of_connection_points):
                source = inner_connection_points[current_city][boarder][track_id]
                target = inner_connection_points[current_city][opposite_boarder][track_id]
                current_track = connect_straight_line_in_grid_map(grid_map, source, target, rail_trans)

                free_rails[current_city].append(current_track)
            for track_id in range(nr_of_connection_points):
                source = inner_connection_points[current_city][boarder][track_id]
                target = inner_connection_points[current_city][opposite_boarder][track_id]

                # Connect parallel tracks with each other
                fix_inner_nodes(
                    grid_map, source, rail_trans)
                fix_inner_nodes(
                    grid_map, target, rail_trans)

                # Connect outer tracks to inner tracks
                if start_idx <= track_id < start_idx + number_of_out_rails:
                    source_outer = outer_connection_points[current_city][boarder][track_id - start_idx]
                    target_outer = outer_connection_points[current_city][opposite_boarder][track_id - start_idx]
                    connect_straight_line_in_grid_map(grid_map, source, source_outer, rail_trans)
                    connect_straight_line_in_grid_map(grid_map, target, target_outer, rail_trans)

        return free_rails

    def _set_trainstation_positions(city_positions: IntVector2DArray, city_radius: int,
                                    free_rails: List[List[List[IntVector2D]]]) -> List[List[Tuple[IntVector2D, int]]]:
        num_cities = len(city_positions)
        train_stations = [[] for i in range(num_cities)]
        for current_city in range(len(city_positions)):
            for track_nbr in range(len(free_rails[current_city])):
                possible_location = free_rails[current_city][track_nbr][
                    int(len(free_rails[current_city][track_nbr]) / 2)]
                train_stations[current_city].append((possible_location, track_nbr))
        return train_stations

    def _generate_start_target_pairs(num_agents: int, num_cities: int,
                                     train_stations: List[List[Tuple[IntVector2D, int]]],
                                     city_orientation: List[Grid4TransitionsEnum]) -> List[Tuple[int, int,
                                                                                                 Grid4TransitionsEnum]]:
        # Generate start and target city directory for all agents.
        # Assure that start and target are not in the same city
        agent_start_targets_cities = []

        # Slot availability in city
        city_available_start = []
        city_available_target = []
        for city_idx in range(num_cities):
            city_available_start.append(len(train_stations[city_idx]))
            city_available_target.append(len(train_stations[city_idx]))

        # Assign agents to slots
        for agent_idx in range(num_agents):
            avail_start_cities = [idx for idx, val in enumerate(city_available_start) if val > 0]
            # avail_target_cities = [idx for idx, val in enumerate(city_available_target) if val > 0]
            # Set probability to choose start and stop from trainstations
            sum_start = sum(np.array(city_available_start)[avail_start_cities])
            # sum_target = sum(np.array(city_available_target)[avail_target_cities])
            p_avail_start = [float(i) / sum_start for i in np.array(city_available_start)[avail_start_cities]]

            start_target_tuple = np.random.choice(avail_start_cities, p=p_avail_start, size=2, replace=False)
            start_city = start_target_tuple[0]
            target_city = start_target_tuple[1]
            agent_start_targets_cities.append((start_city, target_city, city_orientation[start_city]))
        return agent_start_targets_cities

    def _fix_transitions(city_cells: IntVector2DArray, inter_city_lines: List[IntVector2DArray],
                         grid_map: GridTransitionMap, vector_field, rail_trans: RailEnvTransitions, ):
        """
        Function to fix all transition elements in environment
        :param rail_trans:
        :param vector_field:
        """
        # Fix all cities with illegal transition maps
        rails_to_fix = np.zeros(3 * grid_map.height * grid_map.width * 2, dtype='int')
        rails_to_fix_cnt = 0
        cells_to_fix = city_cells + inter_city_lines
        for cell in cells_to_fix:
            cell_valid = grid_map.cell_neighbours_valid(cell, True)
            if not cell_valid:
                rails_to_fix[3 * rails_to_fix_cnt] = cell[0]
                rails_to_fix[3 * rails_to_fix_cnt + 1] = cell[1]
                rails_to_fix[3 * rails_to_fix_cnt + 2] = vector_field[cell]
                rails_to_fix_cnt += 1
        # Fix all other cells
        for cell in range(rails_to_fix_cnt):
            grid_map.fix_transitions((rails_to_fix[3 * cell], rails_to_fix[3 * cell + 1]), rails_to_fix[3 * cell + 2])

    def _closest_neighbour_in_grid4_directions(current_city_idx: int, city_positions: IntVector2DArray) -> List[int]:
        """
        Returns indices of closest neighbour in every direction NESW
        :param current_city_idx: Index of city in city_positions list
        :param city_positions: list of all points being considered
        :return: list of index of closest neighbour in all directions
        """
        city_distances = []
        closest_neighbour: List[int] = [None for i in range(4)]

        # compute distance to all other cities
        for city_idx in range(len(city_positions)):
            city_distances.append(
                Vec2dOperations.get_manhattan_distance(city_positions[current_city_idx], city_positions[city_idx]))
        sorted_neighbours = np.argsort(city_distances)

        for neighbour in sorted_neighbours[1:]:  # do not include city itself
            direction_to_neighbour = direction_to_point(city_positions[current_city_idx], city_positions[neighbour])
            if closest_neighbour[direction_to_neighbour] is None:
                closest_neighbour[direction_to_neighbour] = neighbour

            # early return once all 4 directions have a closest neighbour
            if None not in closest_neighbour:
                return closest_neighbour

        return closest_neighbour

    def argsort(seq):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__)

    def _get_cells_in_city(center: IntVector2D, radius: int, vector_field: Vec2d) -> IntVector2DArray:
        """

        Parameters
        ----------
        center center coordinates of city
        radius radius of city (it is a square)

        Returns
        -------
        flat list of all cell coordinates in the city

        """
        x_range = np.arange(center[0] - radius, center[0] + radius + 1)
        y_range = np.arange(center[1] - radius, center[1] + radius + 1)
        x_values = np.repeat(x_range, len(y_range))
        y_values = np.tile(y_range, len(x_range))
        city_cells = list(zip(x_values, y_values))
        for cell in city_cells:
            vector_field[cell] = direction_to_point(center, (cell[0], cell[1]))
        return city_cells

    def _are_cities_overlapping(center_1, center_2, radius):
        return np.abs(center_1[0] - center_2[0]) < radius and np.abs(center_1[1] - center_2[1]) < radius

    return generator
