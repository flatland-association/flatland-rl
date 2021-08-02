"""Line generators (railway undertaking, "EVU")."""
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.schedule_utils import Line
from flatland.envs import persistence

AgentPosition = Tuple[int, int]
LineGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Line]


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
    def __init__(self, speed_ratio_map: Mapping[float, float] = None, seed: int = 1):
        self.speed_ratio_map = speed_ratio_map
        self.seed = seed

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any=None, num_resets: int = 0,
        np_random: RandomState = None) -> Line:
        pass

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)



def complex_line_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> LineGenerator:
    """

    Generator used to generate the levels of Round 1 in the Flatland Challenge. It can only be used together
    with complex_rail_generator. It places agents at end and start points provided by the rail generator.
    It assigns speeds to the different agents according to the speed_ratio_map
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    :return:
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the line
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """
        # Todo: Remove parameters and variables not used for next version, Issue: <https://gitlab.aicrowd.com/flatland/flatland/issues/305>
        _runtime_seed = seed + num_resets

        start_goal = hints['start_goal']
        start_dir = hints['start_dir']
        agents_position = [sg[0] for sg in start_goal[:num_agents]]
        agents_target = [sg[1] for sg in start_goal[:num_agents]]
        agents_direction = start_dir[:num_agents]

        if speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)

        # Compute max number of steps with given line
        extra_time_factor = 1.5  # Factor to allow for more then minimal time
        max_episode_steps = int(extra_time_factor * rail.height * rail.width)

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None)

    return generator


def sparse_line_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> LineGenerator:
    return SparseLineGen(speed_ratio_map, seed)


class SparseLineGen(BaseLineGen):
    """

    This is the line generator which is used for Round 2 of the Flatland challenge. It produces lines
    to railway networks provided by sparse_rail_generator.
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    """

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the line
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """

        _runtime_seed = self.seed + num_resets

        train_stations = hints['train_stations']
        city_positions = hints['city_positions']
        city_orientation = hints['city_orientations']
        max_num_agents = hints['num_agents']
        city_orientations = hints['city_orientations']
        if num_agents > max_num_agents:
            num_agents = max_num_agents
            warnings.warn("Too many agents! Changes number of agents.")
        # Place agents and targets within available train stations
        agents_position = []
        agents_target = []
        agents_direction = []

        for agent_pair_idx in range(0, num_agents, 2):
            infeasible_agent = True
            tries = 0
            while infeasible_agent:
                tries += 1
                infeasible_agent = False

                # Setlect 2 cities, find their num_stations and possible orientations
                city_idx = np_random.choice(len(city_positions), 2, replace=False)
                city1 = city_idx[0]
                city2 = city_idx[1]
                city1_num_stations = len(train_stations[city1])
                city2_num_stations = len(train_stations[city2])
                city1_possible_orientations = [city_orientation[city1],
                                                (city_orientation[city1] + 2) % 4]
                city2_possible_orientations = [city_orientation[city2],
                                                (city_orientation[city2] + 2) % 4]
                # Agent 1 : city1 > city2, Agent 2: city2 > city1
                agent1_start_idx = ((2 * np_random.randint(0, 10))) % city1_num_stations
                agent1_target_idx = ((2 * np_random.randint(0, 10)) + 1) % city2_num_stations
                agent2_start_idx = ((2 * np_random.randint(0, 10))) % city2_num_stations
                agent2_target_idx = ((2 * np_random.randint(0, 10)) + 1) % city1_num_stations
                
                agent1_start = train_stations[city1][agent1_start_idx]
                agent1_target = train_stations[city2][agent1_target_idx]
                agent2_start = train_stations[city2][agent2_start_idx]
                agent2_target = train_stations[city1][agent2_target_idx]
                            
                agent1_orientation = np_random.choice(city1_possible_orientations)
                agent2_orientation = np_random.choice(city2_possible_orientations)

                # check path exists then break if tries > 100
                if tries >= 100:
                    warnings.warn("Did not find any possible path, check your parameters!!!")
                    break
            
            # agent1 details
            agents_position.append((agent1_start[0][0], agent1_start[0][1]))
            agents_target.append((agent1_target[0][0], agent1_target[0][1]))
            agents_direction.append(agent1_orientation)
            # agent2 details
            agents_position.append((agent2_start[0][0], agent2_start[0][1]))
            agents_target.append((agent2_target[0][0], agent2_target[0][1]))
            agents_direction.append(agent2_orientation)

        if self.speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, self.speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)

        # We add multiply factors to the max number of time steps to simplify task in Flatland challenge.
        # These factors might change in the future.
        timedelay_factor = 4
        alpha = 2
        max_episode_steps = int(
            timedelay_factor * alpha * (rail.width + rail.height + num_agents / len(city_positions)))

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None)


def random_line_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> LineGenerator:
    return RandomLineGen(speed_ratio_map, seed)


class RandomLineGen(BaseLineGen):
    
    """
    Given a `rail` GridTransitionMap, return a random placement of agents (initial position, direction and target).

    Parameters
    ----------
        speed_ratio_map : Optional[Mapping[float, float]]
            A map of speeds mapping to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
        Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
            initial positions, directions, targets speeds
    """

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:
        _runtime_seed = self.seed + num_resets

        valid_positions = []
        for r in range(rail.height):
            for c in range(rail.width):
                if rail.get_full_transitions(r, c) > 0:
                    valid_positions.append((r, c))
        if len(valid_positions) == 0:
            return Line(agent_positions=[], agent_directions=[],
                            agent_targets=[], agent_speeds=[], agent_malfunction_rates=None)

        if len(valid_positions) < num_agents:
            warnings.warn("line_generators: len(valid_positions) < num_agents")
            return Line(agent_positions=[], agent_directions=[],
                            agent_targets=[], agent_speeds=[], agent_malfunction_rates=None)

        agents_position_idx = [i for i in np_random.choice(len(valid_positions), num_agents, replace=False)]
        agents_position = [valid_positions[agents_position_idx[i]] for i in range(num_agents)]
        agents_target_idx = [i for i in np_random.choice(len(valid_positions), num_agents, replace=False)]
        agents_target = [valid_positions[agents_target_idx[i]] for i in range(num_agents)]
        update_agents = np.zeros(num_agents)

        re_generate = True
        cnt = 0
        while re_generate:
            cnt += 1
            if cnt > 1:
                print("re_generate cnt={}".format(cnt))
            if cnt > 1000:
                raise Exception("After 1000 re_generates still not success, giving up.")
            # update position
            for i in range(num_agents):
                if update_agents[i] == 1:
                    x = np.setdiff1d(np.arange(len(valid_positions)), agents_position_idx)
                    agents_position_idx[i] = np_random.choice(x)
                    agents_position[i] = valid_positions[agents_position_idx[i]]
                    x = np.setdiff1d(np.arange(len(valid_positions)), agents_target_idx)
                    agents_target_idx[i] = np_random.choice(x)
                    agents_target[i] = valid_positions[agents_target_idx[i]]
            update_agents = np.zeros(num_agents)

            # agents_direction must be a direction for which a solution is
            # guaranteed.
            agents_direction = [0] * num_agents
            re_generate = False
            for i in range(num_agents):
                valid_movements = []
                for direction in range(4):
                    position = agents_position[i]
                    moves = rail.get_transitions(position[0], position[1], direction)
                    for move_index in range(4):
                        if moves[move_index]:
                            valid_movements.append((direction, move_index))

                valid_starting_directions = []
                for m in valid_movements:
                    new_position = get_new_position(agents_position[i], m[1])
                    if m[0] not in valid_starting_directions and rail.check_path_exists(new_position, m[1],
                                                                                        agents_target[i]):
                        valid_starting_directions.append(m[0])

                if len(valid_starting_directions) == 0:
                    update_agents[i] = 1
                    warnings.warn(
                        "reset position for agent[{}]: {} -> {}".format(i, agents_position[i], agents_target[i]))
                    re_generate = True
                    break
                else:
                    agents_direction[i] = valid_starting_directions[
                        np_random.choice(len(valid_starting_directions), 1)[0]]

        agents_speed = speed_initialization_helper(num_agents, self.speed_ratio_map, seed=_runtime_seed, 
            np_random=np_random)

        # Compute max number of steps with given line
        extra_time_factor = 1.5  # Factor to allow for more then minimal time
        max_episode_steps = int(extra_time_factor * rail.height * rail.width)

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, agent_malfunction_rates=None)



def line_from_file(filename, load_from_package=None) -> LineGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:

        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)

        max_episode_steps = env_dict.get("max_episode_steps", 0)
        if (max_episode_steps==0):
            print("This env file has no max_episode_steps (deprecated) - setting to 100")
            max_episode_steps = 100
            
        agents = env_dict["agents"]
        
        # setup with loaded data
        agents_position = [a.initial_position for a in agents]

        # this logic is wrong - we should really load the initial_direction as the direction.
        #agents_direction = [a.direction for a in agents]
        agents_direction = [a.initial_direction for a in agents]
        agents_target = [a.target for a in agents]
        agents_speed = [a.speed_data['speed'] for a in agents]

        # Malfunctions from here are not used.  They have their own generator.
        #agents_malfunction = [a.malfunction_data['malfunction_rate'] for a in agents]

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, 
                        agent_malfunction_rates=None)

    return generator
