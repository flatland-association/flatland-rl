"""Rail generators (infrastructure manager, "Infrastrukturbetreiber")."""
import sys
import warnings
from typing import Callable, Tuple, Optional, Dict, List

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.grid_utils import distance_on_rail, IntVector2DArray, IntVector2D, \
    Vec2dOperations
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes, align_cell_to_city
from flatland.envs import persistence


RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
""" A rail generator returns a RailGenerator Product, which is just
    a GridTransitionMap followed by an (optional) dict/
"""

RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]


class RailGen(object):
    """ Base class for RailGen(erator) replacement
    
        WIP to replace bare generators with classes / objects without unnamed local variables
        which prevent pickling.
    """ 
    def __init__(self, *args, **kwargs):
        """ constructor to record any state to be reused in each "generation"
        """
        pass

    def generate(self, width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGeneratorProduct:
        pass

    def __call__(self, *args, **kwargs) -> RailGeneratorProduct:
        return self.generate(*args, **kwargs)





def empty_rail_generator() -> RailGenerator:
    return EmptyRailGen()

class EmptyRailGen(RailGen):
    """
    Returns a generator which returns an empty rail mail with no agents.
    Primarily used by the editor
    """

    def generate(self, width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)

        return grid_map, None



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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:

        if num_agents > nr_start_goal:
            num_agents = nr_start_goal
            print("complex_rail_generator: num_agents > nr_start_goal, changing num_agents")
        grid_map = GridTransitionMap(width=width, height=height, transitions=RailEnvTransitions())
        rail_array = grid_map.grid
        rail_array.fill(0)

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
                start = (np_random.randint(0, height), np_random.randint(0, width))
                goal = (np_random.randint(0, height), np_random.randint(0, width))

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
                    """
                    Function to check the distance betweens start and goal
                    :param sg_new: start and goal tuple
                    :return: True if distance is larger than 2, False otherwise
                    """
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
                start = (np_random.randint(0, height), np_random.randint(0, width))
                goal = (np_random.randint(0, height), np_random.randint(0, width))
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
            else:
                # after too many failures we will give up
                created_sanity += 1

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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:
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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> List:
        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)
        rail_env_transitions = RailEnvTransitions()

        grid = np.array(env_dict["grid"])
        rail = GridTransitionMap(width=np.shape(grid)[1], height=np.shape(grid)[0], transitions=rail_env_transitions)
        rail.grid = grid
        if "distance_map" in env_dict:
            distance_map = env_dict["distance_map"]
            if len(distance_map) > 0:
                return rail, {'distance_map': distance_map}
        return [rail, None]

    return generator

class RailFromGridGen(RailGen):
    def __init__(self, rail_map):
        self.rail_map = rail_map

    def generate(self, width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:
        return self.rail_map, None


def rail_from_grid_transition_map(rail_map) -> RailGenerator:
    return RailFromGridGen(rail_map)

def rail_from_grid_transition_map_old(rail_map) -> RailGenerator:
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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:
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

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:
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
            """
            Returns a list of possible transition maps for a given template

            Parameters:
            ------
            template:List[int]

            Returns:
            ------
            List[int]
            """
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
                cell = cells_to_fill[np_random.choice(len(cells_to_fill), 1)[0]]
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

                            rail[row][col] = np_random.choice(possible_transitions,
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

                    rail[row][col] = np_random.choice(possible_transitions,
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




def sparse_rail_generator(*args, **kwargs):
    return SparseRailGen(*args, **kwargs)

class SparseRailGen(RailGen):

    def __init__(self, max_num_cities: int = 5, grid_mode: bool = False, max_rails_between_cities: int = 4,
                          max_rails_in_city: int = 4, seed=0) -> RailGenerator:
        """
        Generates railway networks with cities and inner city rails

        Parameters
        ----------
        max_num_cities : int
            Max number of cities to build. The generator tries to achieve this numbers given all the parameters
        grid_mode: Bool
            How to distribute the cities in the path, either equally in a grid or random
        max_rails_between_cities: int
            Max number of rails connecting to a city. This is only the number of connection points at city boarder.
            Number of tracks drawn inbetween cities can still vary
        max_rails_in_city: int
            Number of parallel tracks in the city. This represents the number of tracks in the trainstations
        seed: int
            Initiate the seed

        Returns
        -------
        Returns the rail generator object to the rail env constructor
        """
        self.max_num_cities = max_num_cities
        self.grid_mode = grid_mode
        self.max_rails_between_cities = max_rails_between_cities
        self.max_rails_in_city = max_rails_in_city
        self.seed = seed # TODO: seed in constructor or generate?


    def generate(self, width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:
        """

        Parameters
        ----------
        width: int
            Width of the environment
        height: int
            Height of the environment
        num_agents:
            Number of agents to be placed within the environment
        num_resets: int
            Count for how often the environment has been reset

        Returns
        -------
        Returns the grid_map --> The railway infrastructure
        Hints:
        agents_hints': {
            'num_agents': how many agents have starting and end spots
            'agent_start_targets_cities': touples of agent start and target cities
            'train_stations': locations of train stations for start and targets
            'city_orientations' : orientation of cities
        """
        if np_random is None:
            np_random = RandomState()
            
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        # We compute the city radius by the given max number of rails it can contain.
        # The radius is equal to the number of tracks divided by 2
        # We add 2 cells to avoid that track lenght is to short
        city_padding = 2
        # We use ceil if we get uneven numbers of city radius. This is to guarantee that all rails fit within the city.
        city_radius = int(np.ceil((self.max_rails_in_city) / 2)) + city_padding
        vector_field = np.zeros(shape=(height, width)) - 1.

        min_nr_rails_in_city = 2
        rails_in_city = min_nr_rails_in_city if self.max_rails_in_city < min_nr_rails_in_city else self.max_rails_in_city
        rails_between_cities = rails_in_city if self.max_rails_between_cities > rails_in_city else self.max_rails_between_cities

        # Calculate the max number of cities allowed
        # and reduce the number of cities to build to avoid problems
        max_feasible_cities = min(self.max_num_cities,
                                  ((height - 2) // (2 * (city_radius + 1))) * ((width - 2) // (2 * (city_radius + 1))))
        if max_feasible_cities < 2:
            # sys.exit("[ABORT] Cannot fit more than one city in this map, no feasible environment possible! Aborting.")
            raise ValueError("ERROR: Cannot fit more than one city in this map, no feasible environment possible!")

        # Evenly distribute cities
        if self.grid_mode:
            city_positions = self._generate_evenly_distr_city_positions(max_feasible_cities, city_radius, width,
                                                                   height)
        # Distribute cities randomlz
        else:
            city_positions = self._generate_random_city_positions(max_feasible_cities, city_radius, width, height,
                                                             np_random=np_random)

        # reduce num_cities if less were generated in random mode
        num_cities = len(city_positions)
        # If random generation failed just put the cities evenly
        if num_cities < 2:
            warnings.warn("[WARNING] Changing to Grid mode to place at least 2 cities.")
            city_positions = self._generate_evenly_distr_city_positions(max_feasible_cities, city_radius, width,
                                                                   height)
        num_cities = len(city_positions)

        # Set up connection points for all cities
        inner_connection_points, outer_connection_points, city_orientations, city_cells = \
            self._generate_city_connection_points(
                city_positions, city_radius, vector_field, rails_between_cities,
                rails_in_city, np_random=np_random)

        # Connect the cities through the connection points
        inter_city_lines = self._connect_cities(city_positions, outer_connection_points, city_cells,
                                           rail_trans, grid_map)

        # Build inner cities
        free_rails = self._build_inner_cities(city_positions, inner_connection_points,
                                         outer_connection_points,
                                         rail_trans,
                                         grid_map)

        # Populate cities
        train_stations = self._set_trainstation_positions(city_positions, city_radius, free_rails)

        # Fix all transition elements
        self._fix_transitions(city_cells, inter_city_lines, grid_map, vector_field)

        return grid_map, {'agents_hints': {
            'num_agents': num_agents,
            'city_positions': city_positions,
            'train_stations': train_stations,
            'city_orientations': city_orientations
        }}

    def _generate_random_city_positions(self, num_cities: int, city_radius: int, width: int,
                                        height: int, np_random: RandomState = None) -> (
        IntVector2DArray, IntVector2DArray):
        """
        Distribute the cities randomly in the environment while respecting city sizes and guaranteeing that they
        don't overlap.

        Parameters
        ----------
        num_cities: int
            Max number of cities that should be placed
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        width: int
            Width of the environment
        height: int
            Height of the environment

        Returns
        -------
        Returns a list of all city positions as coordinates (x,y)

        """

        city_positions: IntVector2DArray = []
        for city_idx in range(num_cities):
            too_close = True
            tries = 0

            while too_close:
                row = city_radius + 1 + np_random.randint(height - 2 * (city_radius + 1))
                col = city_radius + 1 + np_random.randint(width - 2 * (city_radius + 1))
                too_close = False
                # Check distance to cities
                for city_pos in city_positions:
                    if self.__class__._are_cities_overlapping((row, col), city_pos, 2 * (city_radius + 1) + 1):
                        too_close = True

                if not too_close:
                    city_positions.append((row, col))

                tries += 1
                if tries > 200:
                    warnings.warn(
                        "Could not set all required cities!")
                    break
        return city_positions

    def _generate_evenly_distr_city_positions(self, num_cities: int, city_radius: int, width: int, height: int
                                              ) -> (IntVector2DArray, IntVector2DArray):
        """
        Distribute the cities in an evenly spaced grid

        Parameters
        ----------
        num_cities: int
            Max number of cities that should be placed
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        width: int
            Width of the environment
        height: int
            Height of the environment

        Returns
        -------
        Returns a list of all city positions as coordinates (x,y)

        """
        aspect_ratio = height / width

        # Compute max numbe of possible cities per row and col.
        # Respect padding at edges of environment
        # Respect padding between cities
        padding = 2
        city_size = 2 * (city_radius + 1)
        max_cities_per_row = int((height - padding) // city_size)
        max_cities_per_col = int((width - padding) // city_size)

        # Choose number of cities per row.
        # Limit if it is more then max number of possible cities

        cities_per_row = min(int(np.ceil(np.sqrt(num_cities * aspect_ratio))), max_cities_per_row)
        cities_per_col = min(int(np.ceil(num_cities / cities_per_row)), max_cities_per_col)
        num_build_cities = min(num_cities, cities_per_col * cities_per_row)
        row_positions = np.linspace(city_radius + 2, height - (city_radius + 2), cities_per_row, dtype=int)
        col_positions = np.linspace(city_radius + 2, width - (city_radius + 2), cities_per_col, dtype=int)
        city_positions = []

        for city_idx in range(num_build_cities):
            row = row_positions[city_idx % cities_per_row]
            col = col_positions[city_idx // cities_per_row]
            city_positions.append((row, col))
        return city_positions

    def _generate_city_connection_points(self, city_positions: IntVector2DArray, city_radius: int,
                                         vector_field: IntVector2DArray, rails_between_cities: int,
                                         rails_in_city: int = 2, np_random: RandomState = None) -> (
        List[List[List[IntVector2D]]],
        List[List[List[IntVector2D]]],
        List[np.ndarray],
        List[Grid4TransitionsEnum]):
        """
        Generate the city connection points. Internal connection points are used to generate the parallel paths
        within the city.
        External connection points are used to connect different cities together

        Parameters
        ----------
        city_positions: IntVector2DArray
            Vector that contains all the positions of the cities
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        vector_field: IntVector2DArray
            Vectorfield of the size of the environment. It is used to generate preferred orienations for each cell.
            Each cell contains the prefered orientation of cells. If no prefered orientation is present it is set to -1
        rails_between_cities: int
            Number of rails that connect out from the city
        rails_in_city: int
            Number of rails within the city

        Returns
        -------
        inner_connection_points: List of List of length number of cities
            Contains all the inner connection points for each boarder of each city.
            [North_Points, East_Poinst, South_Points, West_Points]
        outer_connection_points: List of List of length number of cities
            Contains all the outer connection points for each boarder of the city.
            [North_Points, East_Poinst, South_Points, West_Points]
        city_orientations: List of length number of cities
            Contains all the orientations of cities. This is then used to orient agents according to the rails
        city_cells: List
            List containing the coordinates of all the cells that belong to a city. This is used by other algorithms
            to avoid drawing inter-city-rails through cities.
        """
        inner_connection_points: List[List[List[IntVector2D]]] = []
        outer_connection_points: List[List[List[IntVector2D]]] = []
        city_orientations: List[Grid4TransitionsEnum] = []
        city_cells: IntVector2DArray = []

        for city_position in city_positions:

            # Chose the directions where close cities are situated
            neighb_dist = []
            for neighbour_city in city_positions:
                neighb_dist.append(Vec2dOperations.get_manhattan_distance(city_position, neighbour_city))
            closest_neighb_idx = self.__class__.argsort(neighb_dist)

            # Store the directions to these neighbours and orient city to face closest neighbour
            connection_sides_idx = []
            idx = 1
            if self.grid_mode:
                current_closest_direction = np_random.randint(4)
            else:
                current_closest_direction = direction_to_point(city_position, city_positions[closest_neighb_idx[idx]])
            connection_sides_idx.append(current_closest_direction)
            connection_sides_idx.append((current_closest_direction + 2) % 4)
            city_orientations.append(current_closest_direction)
            city_cells.extend(self._get_cells_in_city(city_position, city_radius, city_orientations[-1], vector_field))
            # set the number of tracks within a city, at least 2 tracks per city
            connections_per_direction = np.zeros(4, dtype=int)
            nr_of_connection_points = np_random.randint(2, rails_in_city + 1)
            for idx in connection_sides_idx:
                connections_per_direction[idx] = nr_of_connection_points
            connection_points_coordinates_inner: List[List[IntVector2D]] = [[] for i in range(4)]
            connection_points_coordinates_outer: List[List[IntVector2D]] = [[] for i in range(4)]
            number_of_out_rails = np_random.randint(1, min(rails_between_cities, nr_of_connection_points) + 1)
            start_idx = int((nr_of_connection_points - number_of_out_rails) / 2)
            for direction in range(4):
                connection_slots = np.arange(nr_of_connection_points) - start_idx
                # Offset the rails away from the center of the city
                offset_distances = np.arange(nr_of_connection_points) - int(nr_of_connection_points / 2)
                # The clipping helps ofsetting one side more than the other to avoid switches at same locations
                # The magic number plus one is added such that all points have at least one offset
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
        return inner_connection_points, outer_connection_points, city_orientations, city_cells

    def _connect_cities(self, city_positions: IntVector2DArray, connection_points: List[List[List[IntVector2D]]],
                        city_cells: IntVector2DArray,
                        rail_trans: RailEnvTransitions, grid_map: RailEnvTransitions) -> List[IntVector2DArray]:
        """
        Connects cities together through rails. Each city connects from its outgoing connection points to the closest
        cities. This guarantees that all connection points are used.

        Parameters
        ----------
        city_positions: IntVector2DArray
            All coordinates of the cities
        connection_points: List[List[List[IntVector2D]]]
            List of coordinates of all outer connection points
        city_cells: IntVector2DArray
            Coordinates of all the cells contained in any city. This is used to avoid drawing rails through existing
            cities.
        rail_trans: RailEnvTransitions
            Railway transition objects
        grid_map: RailEnvTransitions
            The grid map containing the rails. Used to draw new rails

        Returns
        -------
        Returns a list of all the cells (Coordinates) that belong to a rail path. This can be used to access railway
        cells later.
        """
        all_paths: List[IntVector2DArray] = []

        grid4_directions = [Grid4TransitionsEnum.NORTH, Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.SOUTH,
                            Grid4TransitionsEnum.WEST]

        for current_city_idx in np.arange(len(city_positions)):
            closest_neighbours = self._closest_neighbour_in_grid4_directions(current_city_idx, city_positions)
            for out_direction in grid4_directions:

                neighbour_idx = self.get_closest_neighbour_for_direction(closest_neighbours, out_direction)

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

    def get_closest_neighbour_for_direction(self, closest_neighbours, out_direction):
        """
        Given a list of clostest neighbours in each direction this returns the city index of the neighbor in a given
        direction. Direction is a 90 degree cone facing the desired directiont.
        Exampe:
            North: The closes neighbour in the North direction is within the cone spanned by a line going
            North-West and North-East

        Parameters
        ----------
        closest_neighbours: List
            List of length 4 containing the index of closes neighbour in the corresponfing direction:
            [North-Neighbour, East-Neighbour, South-Neighbour, West-Neighbour]
        out_direction: int
            Direction we want to get city index from
            North: 0, East: 1, South: 2, West: 3

        Returns
        -------
        Returns the index of the closest neighbour in the desired direction. If none was present the neighbor clockwise
        or counter clockwise is returned
        """

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

    def _build_inner_cities(self, city_positions: IntVector2DArray, inner_connection_points: List[List[List[IntVector2D]]],
                            outer_connection_points: List[List[List[IntVector2D]]], rail_trans: RailEnvTransitions,
                            grid_map: GridTransitionMap) -> (List[IntVector2DArray], List[List[List[IntVector2D]]]):
        """
        Set the parallel tracks within the city. The center track of the city is of the length of the city, the lenght
        of the tracks decrease by 2 for every parallel track away from the center
        EG:

                ---     Left Track
               -----    Center Track
                ---     Right Track

        Parameters
        ----------
        city_positions: IntVector2DArray
                        All coordinates of the cities

        inner_connection_points: List[List[List[IntVector2D]]]
            Points on city boarder that are used to generate inner city track
        outer_connection_points: List[List[List[IntVector2D]]]
            Points where the city is connected to neighboring cities
        rail_trans: RailEnvTransitions
            Railway transition objects
        grid_map: RailEnvTransitions
            The grid map containing the rails. Used to draw new rails

        Returns
        -------
        Returns a list of all the cells (Coordinates) that belong to a rail paths within the city.
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

    def _set_trainstation_positions(self, city_positions: IntVector2DArray, city_radius: int,
                                    free_rails: List[List[List[IntVector2D]]]) -> List[List[Tuple[IntVector2D, int]]]:
        """
        Populate the cities with possible start and end positions. Trainstations are set on the center of each paralell
        track. Each trainstation gets a coordinate as well as number indicating what track it is on

        Parameters
        ----------
        city_positions: IntVector2DArray
                        All coordinates of the cities
        city_radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1
        free_rails: List[List[List[IntVector2D]]]
            Cells that allow for trainstations to be placed

        Returns
        -------
        Returns a List[List[Tuple[IntVector2D, int]]] containing the coordinates of trainstations as well as their
        track number within the city
        """
        num_cities = len(city_positions)
        train_stations = [[] for i in range(num_cities)]
        for current_city in range(len(city_positions)):
            for track_nbr in range(len(free_rails[current_city])):
                possible_location = free_rails[current_city][track_nbr][
                    int(len(free_rails[current_city][track_nbr]) / 2)]
                train_stations[current_city].append((possible_location, track_nbr))
        return train_stations

    def _fix_transitions(self, city_cells: IntVector2DArray, inter_city_lines: List[IntVector2DArray],
                         grid_map: GridTransitionMap, vector_field):
        """
        Check and fix transitions of all the cells that were modified. This is necessary because we ignore validity
        while drawing the rails.

        Parameters
        ----------
        city_cells: IntVector2DArray
            Cells within cities. All of these might have changed and are thus checked
        inter_city_lines: List[IntVector2DArray]
            All cells within rails drawn between cities
        vector_field: IntVector2DArray
            Vectorfield of the size of the environment. It is used to generate preferred orienations for each cell.
            Each cell contains the prefered orientation of cells. If no prefered orientation is present it is set to -1
        grid_map: RailEnvTransitions
            The grid map containing the rails. Used to draw new rails

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

    def _closest_neighbour_in_grid4_directions(self, current_city_idx: int, city_positions: IntVector2DArray) -> List[int]:
        """
        Finds the closest city in each direction of the current city
        Parameters
        ----------
        current_city_idx: int
            Index of current city
        city_positions: IntVector2DArray
            Vector containing the coordinates of all cities

        Returns
        -------
        Returns indices of closest neighbour in every direction NESW
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

    @staticmethod
    def argsort(seq):
        """
        Same as Numpy sort but for lists
        Parameters
        ----------
        seq: List
            list that we would like to sort from smallest to largest

        Returns
        -------
        Returns the sorted list

        """
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__)

    def _get_cells_in_city(self, center: IntVector2D, radius: int, city_orientation: int,
                           vector_field: IntVector2DArray) -> IntVector2DArray:
        """
        Function the collect cells of a city. It also populates the vector field accoring to the orientation of the
        city.

        Example: City oriented north with a radius of 5, the vectorfield in the city will be as follows:
            |S|S|S|S|S|
            |S|S|S|S|S|
            |S|S|S|S|S|  <-- City center
            |N|N|N|N|N|
            |N|N|N|N|N|

        This is used to later orient the switches to avoid infeasible maps.

        Parameters
        ----------
        center: IntVector2D
            center coordinates of city
        radius: int
            radius of city (it is a square)
        city_orientation: int
            Orientation of city
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
            vector_field[cell] = align_cell_to_city(center, city_orientation, cell)
        return city_cells

    @staticmethod
    def _are_cities_overlapping(center_1, center_2, radius):
        """
        Check if two cities overlap. That is we check if two squares with certain edge length and position overlap
        Parameters
        ----------
        center_1: (int, int)
            Center of first city
        center_2: (int, int)
            Center of second city

        radius: int
            Radius of each city. Cities are squares with edge length 2 * city_radius + 1

        Returns
        -------
        Returns True if the cities overlap and False otherwise
        """
        return np.abs(center_1[0] - center_2[0]) < radius and np.abs(center_1[1] - center_2[1]) < radius

