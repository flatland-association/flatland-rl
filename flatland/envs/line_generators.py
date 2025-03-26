"""Line generators: Railway Undertaking (RU) / Eisenbahnverkehrsunternehmen (EVU)."""
import pickle
from pathlib import Path
from typing import Tuple, List, Callable, Mapping, Optional, Any

from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2DArray
from flatland.envs import persistence
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.timetable_utils import Line

AgentPosition = Tuple[int, int]
LineGenerator = Callable[[RailGridTransitionMap, int, Optional[Any], Optional[int], Optional[RandomState]], Line]


def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None, np_random: RandomState = None) -> List[float]:
    """
    Parameters
    ----------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mapping to their ratio of appearance. The ratios must sum up to 1.
    np_random : RandomState

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

    def generate(self, rail: RailGridTransitionMap, num_agents: int, hints: dict = None, num_resets: int = 0, np_random: RandomState = None) -> Line:
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

    @staticmethod
    def decide_orientation(rail, start, target, possible_orientations, np_random: RandomState) -> int:
        feasible_orientations = []

        for orientation in possible_orientations:
            if rail.check_path_exists(start[0], orientation, target[0]):
                feasible_orientations.append(orientation)

        if len(feasible_orientations) > 0:
            return np_random.choice(feasible_orientations)
        else:
            return 0

    def _assign_station_in_start_and_target_city(self, hints: dict, rail: RailGridTransitionMap, city_start: int, city_target: int,
                                                 np_random: RandomState):
        train_stations = hints['train_stations']
        city_orientation = hints['city_orientations']
        city_start_num_stations = len(train_stations[city_start])
        city_target_num_stations = len(train_stations[city_target])

        city_start_possible_orientations = [city_orientation[city_start],
                                            (city_orientation[city_start] + 2) % 4]

        agent_start_idx = (2 * np_random.randint(0, 10)) % city_start_num_stations
        agent_target_idx = ((2 * np_random.randint(0, 10)) + 1) % city_target_num_stations

        agent_start = train_stations[city_start][agent_start_idx]
        agent_target = train_stations[city_target][agent_target_idx]

        agent_orientation = self.decide_orientation(
            rail, agent_start, agent_target, city_start_possible_orientations, np_random)

        return agent_start, agent_orientation, agent_target

    def generate(self, rail: RailGridTransitionMap, num_agents: int, hints: dict = None, num_resets: int = 0, np_random: RandomState = None) -> Line:
        """
        Assigns tasks to all the agents.

        Parameters
        ----------
        rail : RailGridTransitionMap
            Rail infrastructure given by the rail_generator
        num_agents : int
            Number of agents to include in the line
        hints : dict
            Hints provided by the rail_generator These include positions of start/target positions
        num_resets: int
            How often the generator has been reset.
        np_random : RandomState

        Returns
        -------
        Line:
            the line
        """

        _runtime_seed = self.seed + num_resets
        city_positions: IntVector2DArray = hints['city_positions']

        # Place agents and targets within available train stations
        agent_positions = []
        agent_targets = []
        agents_directions = []

        for agent_idx in range(num_agents):
            if agent_idx % 2 == 0:
                # Select line_length cities, find their num_stations and possible orientations
                city_idx: List[int] = np_random.choice(len(city_positions), self.line_length, replace=False)
                # Run a train in from city 0 to city -1

            else:
                # Run a train in the opposite direction city -1 ...city 0
                city_idx = list(reversed(city_idx))

            cur_agent_orientations = []
            cur_agent_positions = []
            for city1, city2 in zip(city_idx, city_idx[1:]):
                cur_agent_start, cur_agent_orientation, cur_agent_target = self._assign_station_in_start_and_target_city(hints, rail, city1, city2, np_random)
                cur_agent_positions.append((cur_agent_start[0][0], cur_agent_start[0][1]))
                cur_agent_orientations.append(Grid4TransitionsEnum(cur_agent_orientation))
            agent_positions.append(cur_agent_positions)
            agent_targets.append((cur_agent_target[0][0], cur_agent_target[0][1]))
            agents_directions.append(cur_agent_orientations)

        if self.speed_ratio_map:
            agent_speeds = speed_initialization_helper(num_agents, self.speed_ratio_map, np_random=np_random)
        else:
            agent_speeds = [1.0] * len(agent_positions)

        return Line(agent_positions=agent_positions, agent_directions=agents_directions, agent_targets=agent_targets, agent_speeds=agent_speeds)


def line_from_file(filename, load_from_package: str = None) -> LineGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    filename : Pickle file generated by env.save() or editor
    load_from_package : str
                package

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: RailGridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:
        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)

        agents = env_dict["agents"]

        # setup with loaded data
        agents_position = [[a.initial_position] for a in agents]
        agents_direction = [[a.initial_direction] for a in agents]
        agents_target = [a.target for a in agents]
        agents_speed = [a.speed_counter.speed for a in agents]

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                    agent_targets=agents_target, agent_speeds=agents_speed)

    return generator


class FileLineGenerator(BaseLineGen):
    def __init__(self, filename: Path):
        self.filename = filename

    def generate(self, rail: RailGridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0, np_random: RandomState = None) -> Line:
        with open(self.filename, "rb") as file_in:
            return pickle.loads(file_in.read())

    @staticmethod
    def save(filename: Path, line: Line):
        with open(filename, "wb") as file_out:
            file_out.write(pickle.dumps(line))

    @staticmethod
    def wrap(line_generator: LineGenerator, line_pkl: Path) -> LineGenerator:
        def _wrap(*args, **kwargs):
            line = line_generator(*args, **kwargs)
            FileLineGenerator.save(line_pkl, line)
            return line

        return _wrap
