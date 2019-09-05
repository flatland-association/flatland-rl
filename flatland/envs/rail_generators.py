"""Rail generators (infrastructure manager, "Infrastrukturbetreiber")."""
import warnings
from typing import Callable, Tuple, Optional, Dict, List, Any

import msgpack
import numpy as np

from flatland.core.grid.grid4_utils import get_direction, mirror
from flatland.core.grid.grid_utils import distance_on_rail
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail, connect_nodes, connect_from_nodes

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
                           seed=0) -> RailGenerator:
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

        return [rail, None]

    return generator


def rail_from_file(filename) -> RailGenerator:
    """
    Utility to load pickle file

    Parameters
    -------
    filename : Pickle file generated by env.save() or editor

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
        if b"distance_maps" in data.keys():
            distance_maps = data[b"distance_maps"]
            if len(distance_maps) > 0:
                return rail, {'distance_maps': distance_maps}
        return [rail, None]

    return generator


def rail_from_grid_transition_map(rail_map) -> RailGenerator:
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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0) -> RailGeneratorProduct:
        return rail_map, None

    return generator


def random_rail_generator(cell_type_relative_proportion=[1.0] * 11) -> RailGenerator:
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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0) -> RailGeneratorProduct:
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


def sparse_rail_generator(num_cities=5, num_intersections=4, num_trainstations=2, min_node_dist=20, node_radius=2,
                          num_neighb=3, grid_mode=False, enhance_intersection=False, seed=0):
    """
    This is a level generator which generates complex sparse rail configurations

    :param num_cities: Number of city node (can hold trainstations)
    :param num_intersections: Number of intersection that city nodes can connect to
    :param num_trainstations: Total number of trainstations in env
    :param min_node_dist: Minimal distance between nodes
    :param node_radius: Proximity of trainstations to center of city node
    :param num_neighb: Number of neighbouring nodes each node connects to
    :param grid_mode: True -> NOdes evenly distirbuted in env, False-> Random distribution of nodes
    :param enhance_intersection: True -> Extra rail elements added at intersections
    :param seed: Random Seed
    :return:
        -------
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """

    def generator(width, height, num_agents, num_resets=0):

        if num_agents > num_trainstations:
            num_agents = num_trainstations
            warnings.warn("sparse_rail_generator: num_agents > nr_start_goal, changing num_agents")

        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)
        np.random.seed(seed + num_resets)

        # Generate a set of nodes for the sparse network
        # Try to connect cities to nodes first
        city_positions = []
        intersection_positions = []

        # Evenly distribute cities and intersections
        node_positions: List[Any] = None
        nb_nodes = num_cities + num_intersections
        if grid_mode:
            nodes_ratio = height / width
            nodes_per_row = int(np.ceil(np.sqrt(nb_nodes * nodes_ratio)))
            nodes_per_col = int(np.ceil(nb_nodes / nodes_per_row))
            x_positions = np.linspace(node_radius, height - node_radius, nodes_per_row, dtype=int)
            y_positions = np.linspace(node_radius, width - node_radius, nodes_per_col, dtype=int)
            city_idx = np.random.choice(np.arange(nb_nodes), num_cities)

            node_positions = _generate_node_positions_grid_mode(city_idx, city_positions, intersection_positions,
                                                                nb_nodes,
                                                                nodes_per_row, x_positions,
                                                                y_positions)



        else:

            node_positions = _generate_node_positions_not_grid_mode(city_positions, height,
                                                                    intersection_positions,
                                                                    nb_nodes, width)

        # reduce nb_nodes, _num_cities, _num_intersections if less were generated in not_grid_mode
        nb_nodes = len(node_positions)
        _num_cities = len(city_positions)
        _num_intersections = len(intersection_positions)

        # Chose node connection
        # Set up list of available nodes to connect to
        available_nodes_full = np.arange(nb_nodes)
        available_cities = np.arange(_num_cities)
        available_intersections = np.arange(_num_cities, nb_nodes)

        # Start at some node
        current_node = np.random.randint(len(available_nodes_full))
        node_stack = [current_node]
        allowed_connections = num_neighb
        first_node = True
        while len(node_stack) > 0:
            current_node = node_stack[0]
            delete_idx = np.where(available_nodes_full == current_node)
            available_nodes_full = np.delete(available_nodes_full, delete_idx, 0)

            # Priority city to intersection connections
            if current_node < _num_cities and len(available_intersections) > 0:
                available_nodes = available_intersections
                delete_idx = np.where(available_cities == current_node)
                available_cities = np.delete(available_cities, delete_idx, 0)

            # Priority intersection to city connections
            elif current_node >= _num_cities and len(available_cities) > 0:
                available_nodes = available_cities
                delete_idx = np.where(available_intersections == current_node)
                available_intersections = np.delete(available_intersections, delete_idx, 0)

            # If no options possible connect to whatever node is still available
            else:
                available_nodes = available_nodes_full

            # Sort available neighbors according to their distance.
            node_dist = []
            for av_node in available_nodes:
                node_dist.append(distance_on_rail(node_positions[current_node], node_positions[av_node]))
            available_nodes = available_nodes[np.argsort(node_dist)]

            # Set number of neighboring nodes
            if len(available_nodes) >= allowed_connections:
                connected_neighb_idx = available_nodes[:allowed_connections]
            else:
                connected_neighb_idx = available_nodes

            # Less connections for subsequent nodes
            if first_node:
                allowed_connections -= 1
                first_node = False

            # Connect to the neighboring nodes
            for neighb in connected_neighb_idx:
                if neighb not in node_stack:
                    node_stack.append(neighb)
                connect_nodes(rail_trans, rail_array, node_positions[current_node], node_positions[neighb])
            node_stack.pop(0)

        # Place train stations close to the node
        # We currently place them uniformly distributed among all cities
        built_num_trainstation = 0
        train_stations = [[] for i in range(_num_cities)]

        if _num_cities > 1:

            for station in range(num_trainstations):
                spot_found = True
                trainstation_node = int(station / num_trainstations * _num_cities)

                station_x = np.clip(node_positions[trainstation_node][0] + np.random.randint(-node_radius, node_radius),
                                    0,
                                    height - 1)
                station_y = np.clip(node_positions[trainstation_node][1] + np.random.randint(-node_radius, node_radius),
                                    0,
                                    width - 1)
                tries = 0
                while (station_x, station_y) in train_stations \
                    or (station_x, station_y) == node_positions[trainstation_node] \
                    or rail_array[(station_x, station_y)] != 0:  # noqa: E125

                    station_x = np.clip(
                        node_positions[trainstation_node][0] + np.random.randint(-node_radius, node_radius),
                        0,
                        height - 1)
                    station_y = np.clip(
                        node_positions[trainstation_node][1] + np.random.randint(-node_radius, node_radius),
                        0,
                        width - 1)
                    tries += 1
                    if tries > 100:
                        warnings.warn("Could not set trainstations, please change initial parameters!!!!")
                        spot_found = False
                        break

                if spot_found:
                    train_stations[trainstation_node].append((station_x, station_y))

                # Connect train station to the correct node
                connection = connect_from_nodes(rail_trans, rail_array, node_positions[trainstation_node],
                                                (station_x, station_y))
                # Check if connection was made
                if len(connection) == 0:
                    if len(train_stations[trainstation_node]) > 0:
                        train_stations[trainstation_node].pop(-1)
                else:
                    built_num_trainstation += 1

        # Adjust the number of agents if you could not build enough trainstations
        if num_agents > built_num_trainstation:
            num_agents = built_num_trainstation
            warnings.warn("sparse_rail_generator: num_agents > nr_start_goal, changing num_agents")

        # Place passing lanes at intersections
        # We currently place them uniformly distirbuted among all cities
        if enhance_intersection:

            for intersection in range(_num_intersections):
                intersect_x_1 = np.clip(intersection_positions[intersection][0] + np.random.randint(1, 3),
                                        1,
                                        height - 2)
                intersect_y_1 = np.clip(intersection_positions[intersection][1] + np.random.randint(-3, 3),
                                        2,
                                        width - 2)
                intersect_x_2 = np.clip(
                    intersection_positions[intersection][0] + np.random.randint(-3, -1),
                    1,
                    height - 2)
                intersect_y_2 = np.clip(
                    intersection_positions[intersection][1] + np.random.randint(-3, 3),
                    1,
                    width - 2)

                # Connect train station to the correct node
                connect_nodes(rail_trans, rail_array, (intersect_x_1, intersect_y_1),
                              (intersect_x_2, intersect_y_2))
                connect_nodes(rail_trans, rail_array, intersection_positions[intersection],
                              (intersect_x_1, intersect_y_1))
                connect_nodes(rail_trans, rail_array, intersection_positions[intersection],
                              (intersect_x_2, intersect_y_2))
                grid_map.fix_transitions((intersect_x_1, intersect_y_1))
                grid_map.fix_transitions((intersect_x_2, intersect_y_2))

        # Fix all nodes with illegal transition maps
        for current_node in node_positions:
            grid_map.fix_transitions(current_node)

        # Generate start and target node directory for all agents.
        # Assure that start and target are not in the same node
        agent_start_targets_nodes = []

        # Slot availability in node
        node_available_start = []
        node_available_target = []
        for node_idx in range(_num_cities):
            node_available_start.append(len(train_stations[node_idx]))
            node_available_target.append(len(train_stations[node_idx]))

        # Assign agents to slots
        for agent_idx in range(num_agents):
            avail_start_nodes = [idx for idx, val in enumerate(node_available_start) if val > 0]
            avail_target_nodes = [idx for idx, val in enumerate(node_available_target) if val > 0]
            start_node = np.random.choice(avail_start_nodes)
            target_node = np.random.choice(avail_target_nodes)
            tries = 0
            found_agent_pair = True
            while target_node == start_node:
                target_node = np.random.choice(avail_target_nodes)
                tries += 1
                # Test again with new start node if no pair is found (This code needs to be improved)
                if (tries + 1) % 10 == 0:
                    start_node = np.random.choice(avail_start_nodes)
                if tries > 100:
                    warnings.warn("Could not set trainstations, removing agent!")
                    found_agent_pair = False
                    break
            if found_agent_pair:
                node_available_start[start_node] -= 1
                node_available_target[target_node] -= 1
                agent_start_targets_nodes.append((start_node, target_node))
            else:
                num_agents -= 1

        return grid_map, {'agents_hints': {
            'num_agents': num_agents,
            'agent_start_targets_nodes': agent_start_targets_nodes,
            'train_stations': train_stations
        }}

    def _generate_node_positions_not_grid_mode(city_positions, height, intersection_positions, nb_nodes,
                                               width):

        node_positions = []
        for node_idx in range(nb_nodes):
            to_close = True
            tries = 0

            while to_close:
                x_tmp = node_radius + np.random.randint(height - node_radius)
                y_tmp = node_radius + np.random.randint(width - node_radius)
                to_close = False

                # Check distance to cities
                for node_pos in city_positions:
                    if distance_on_rail((x_tmp, y_tmp), node_pos) < min_node_dist:
                        to_close = True

                # Check distance to intersections
                for node_pos in intersection_positions:
                    if distance_on_rail((x_tmp, y_tmp), node_pos) < min_node_dist:
                        to_close = True

                if not to_close:
                    node_positions.append((x_tmp, y_tmp))
                    if node_idx < num_cities:
                        city_positions.append((x_tmp, y_tmp))
                    else:
                        intersection_positions.append((x_tmp, y_tmp))
                tries += 1
                if tries > 100:
                    warnings.warn(
                        "Could not only set {} nodes after {} tries, although {} of nodes required to be generated!".format(
                            len(node_positions),
                            tries, nb_nodes))
                    break

        node_positions = city_positions + intersection_positions
        return node_positions

    def _generate_node_positions_grid_mode(city_idx, city_positions, intersection_positions, nb_nodes,
                                           nodes_per_row, x_positions, y_positions):
        for node_idx in range(nb_nodes):

            x_tmp = x_positions[node_idx % nodes_per_row]
            y_tmp = y_positions[node_idx // nodes_per_row]
            if node_idx in city_idx:
                city_positions.append((x_tmp, y_tmp))
            else:
                intersection_positions.append((x_tmp, y_tmp))
        node_positions = city_positions + intersection_positions
        return node_positions

    return generator
