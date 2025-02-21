"""Line generators: Railway Undertaking (RU) / Eisenbahnverkehrsunternehmen (EVU)."""
from typing import Tuple, List, Callable, Mapping, Optional, Any

from numpy.random.mtrand import RandomState

from flatland.core.grid.grid_utils import IntVector2DArray
from flatland.core.transition_map import GridTransitionMap
from flatland.envs import persistence
from flatland.envs.timetable_utils import Line

AgentPosition = Tuple[int, int]
LineGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int], Optional[RandomState]], Line]


def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None,
                                seed: int = None, np_random: RandomState = None) -> List[float]:
    """
    Parameters
    ----------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
    List[float]
        A list of size nb_agents of speeds with the corresponding probabilistic ratios.
    """
    if speed_ratio_map is None:
        return [1.0] * nb_agents

    nb_classes = len(speed_ratio_map.keys())
    speed_ratio_map_as_list: List[Tuple[float, float]] = list(speed_ratio_map.items())
    speed_ratios = list(map(lambda t: t[1], speed_ratio_map_as_list))
    speeds = list(map(lambda t: t[0], speed_ratio_map_as_list))
    return list(map(lambda index: speeds[index], np_random.choice(nb_classes, nb_agents, p=speed_ratios)))


class BaseLineGen(object):
    def __init__(self, speed_ratio_map: Mapping[float, float] = None, seed: int = 1, line_length: int = 2):
        self.speed_ratio_map = speed_ratio_map
        self.seed = seed
        self.line_length = line_length

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                 np_random: RandomState = None) -> Line:
        pass

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def sparse_line_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1, line_length: int = 2) -> LineGenerator:
    return SparseLineGen(speed_ratio_map, seed, line_length)


class SparseLineGen(BaseLineGen):
    def __init__(self, speed_ratio_map: Mapping[float, float] = None, seed: int = 1, line_length: int = 2):
        """

        This is the line generator which is used for Round 2 of the Flatland challenge. It produces lines
        to railway networks provided by sparse_rail_generator.

        Parameters
        ----------
        speed_ratio_map : Mapping[float, float]
            Speed ratios of all agents. They are probabilities of all different speeds and have to add up to 1.
        seed : int
            Initiate random seed generator
        line_length : int
            The length of the lines.
        """
        super().__init__(speed_ratio_map, seed, line_length)

    def decide_orientation(self, rail, start, target, possible_orientations, np_random: RandomState) -> int:
        feasible_orientations = []

        for orientation in possible_orientations:
            if rail.check_path_exists(start[0], orientation, target[0]):
                feasible_orientations.append(orientation)

        if len(feasible_orientations) > 0:
            return np_random.choice(feasible_orientations)
        else:
            return 0

    def _assign_station_in_start_and_target_city(self, hints: dict, rail: GridTransitionMap, city_start: int, city_target: int,
                                                 np_random: RandomState):
        train_stations = hints['train_stations']
        city_orientation = hints['city_orientations']
        city_start_num_stations = len(train_stations[city_start])
        city_target_num_stations = len(train_stations[city_target])

        city_start_possible_orientations = [city_orientation[city_start],
                                            (city_orientation[city_start] + 2) % 4]

        agent_start_idx = ((2 * np_random.randint(0, 10))) % city_start_num_stations
        agent_target_idx = ((2 * np_random.randint(0, 10)) + 1) % city_target_num_stations

        agent_start = train_stations[city_start][agent_start_idx]
        agent_target = train_stations[city_target][agent_target_idx]

        agent_orientation = self.decide_orientation(
            rail, agent_start, agent_target, city_start_possible_orientations, np_random)

        return agent_start, agent_orientation, agent_target

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: dict, num_resets: int,
                 np_random: RandomState) -> Line:
        """
        Assigns tasks to all the agents.

        Parameters
        ----------
        rail : GridTransitionMap
            Rail infrastructure given by the rail_generator
        num_agents : int
            Number of agents to include in the line
        hints : dict
            Hints provided by the rail_generator These include positions of start/target positions
        num_resets: int
            How often the generator has been reset.

        Returns
        -------
        Line:
            the line
        """

        _runtime_seed = self.seed + num_resets
        city_positions: IntVector2DArray = hints['city_positions']

        # Place agents and targets within available train stations
        agents_position = []
        agents_target = []
        agents_direction = []

        city1, city2 = None, None

        for agent_idx in range(num_agents):
            if (agent_idx % 2 == 0):
                # Select 2 cities, find their num_stations and possible orientations
                city_idx: List[int] = np_random.choice(len(city_positions), self.line_length, replace=False)

                city1 = city_idx[0]
                city2 = city_idx[-1]

                # Run a train in the from city1..city2
                agent_start, agent_orientation, agent_target = self._assign_station_in_start_and_target_city(hints, rail, city1, city2, np_random)
            else:
                # Run a train in the opposite direction city2..city1
                agent_start, agent_orientation, agent_target = self._assign_station_in_start_and_target_city(hints, rail, city2, city1, np_random)

            agents_position.append((agent_start[0][0], agent_start[0][1]))
            agents_target.append((agent_target[0][0], agent_target[0][1]))
            agents_direction.append(agent_orientation)

        if self.speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, self.speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)

        agents_position = [[ap] for ap in agents_position]
        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                    agent_targets=agents_target, agent_speeds=speeds)


def line_from_file(filename, load_from_package=None) -> LineGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    filename : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:
        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)

        agents = env_dict["agents"]

        # setup with loaded data
        agents_position = [[a.initial_position] for a in agents]
        agents_direction = [a.initial_direction for a in agents]
        agents_target = [a.target for a in agents]
        agents_speed = [a.speed_counter.speed for a in agents]

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                    agent_targets=agents_target, agent_speeds=agents_speed)

    return generator
