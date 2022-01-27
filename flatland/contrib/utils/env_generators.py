import logging
import random
import numpy as np
from typing import NamedTuple

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters, ParamMalfunctionGen, no_malfunction_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.agent_utils import TrainState
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax

MalfunctionParameters = NamedTuple('MalfunctionParameters', [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])


def get_shortest_path_action(env,handle):
    distance_map = env.distance_map.get()

    agent = env.agents[handle]
    if agent.status in [TrainState.WAITING, TrainState.READY_TO_DEPART,
                        TrainState.MALFUNCTION_OFF_MAP]:
        agent_virtual_position = agent.initial_position
    elif agent.status in [TrainState.MALFUNCTION, TrainState.MOVING, TrainState.STOPPED]:
        agent_virtual_position = agent.position
    elif agent.status == TrainState.DONE:
        agent_virtual_position = agent.target
    else:
        return None

    if agent.position:
        possible_transitions = env.rail.get_transitions(
            *agent.position, agent.direction)
    else:
        possible_transitions = env.rail.get_transitions(
            *agent.initial_position, agent.direction)

    num_transitions = fast_count_nonzero(possible_transitions)

    min_distances = []
    for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
        if possible_transitions[direction]:
            new_position = get_new_position(
                agent_virtual_position, direction)
            min_distances.append(
                distance_map[handle, new_position[0],
                             new_position[1], direction])
        else:
            min_distances.append(np.inf)

    if num_transitions == 1:
        observation = [0, 1, 0]

    elif num_transitions == 2:
        idx = np.argpartition(np.array(min_distances), 2)
        observation = [0, 0, 0]
        observation[idx[0]] = 1
    return fast_argmax(observation) + 1


def small_v0(random_seed, observation_builder, max_width = 35, max_height = 35):
    random.seed(random_seed)
    width =  30
    height =  30
    nr_trains = 5
    max_num_cities = 4
    grid_mode = False
    max_rails_between_cities = 2
    max_rails_in_city = 3

    malfunction_rate = 0
    malfunction_min_duration = 0
    malfunction_max_duration = 0

    rail_generator = sparse_rail_generator(max_num_cities=max_num_cities, seed=random_seed, grid_mode=False,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rail_pairs_in_city=max_rails_in_city)

    stochastic_data = MalfunctionParameters(malfunction_rate=malfunction_rate,  # Rate of malfunction occurence
                                        min_duration=malfunction_min_duration,  # Minimal duration of malfunction
                                        max_duration=malfunction_max_duration  # Max duration of malfunction
                                        )
    speed_ratio_map = None
    line_generator = sparse_line_generator(speed_ratio_map)

    malfunction_generator = no_malfunction_generator()

    while width <= max_width and height <= max_height:
        try:
            env = RailEnv(width=width, height=height, rail_generator=rail_generator,
                          line_generator=line_generator, number_of_agents=nr_trains,
                        #   malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                          malfunction_generator_and_process_data=malfunction_generator,
                          obs_builder_object=observation_builder, remove_agents_at_target=False)

            print("[{}] {}x{} {} cities {} trains, max {} rails between cities, max {} rails in cities. Malfunction rate {}, {} to {} steps.".format(
                random_seed, width, height, max_num_cities, nr_trains, max_rails_between_cities,
                max_rails_in_city, malfunction_rate, malfunction_min_duration, malfunction_max_duration
            ))

            return env
        except ValueError as e:
            logging.error(f"Error: {e}")
            width += 5
            height += 5
            logging.info("Try again with larger env: (w,h):", width, height)
    logging.error(f"Unable to generate env with seed={random_seed}, max_width={max_height}, max_height={max_height}")
    return None


def random_sparse_env_small(random_seed, observation_builder, max_width = 45, max_height = 45):
    random.seed(random_seed)
    size = random.randint(0, 5)
    width = 20 + size * 5
    height = 20 + size * 5
    nr_cities = 2 + size // 2 + random.randint(0, 2)
    nr_trains = min(nr_cities * 5, 5 + random.randint(0, 5))  # , 10 + random.randint(0, 10))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, size)
    malfunction_rate = 30 + random.randint(0, 100)
    malfunction_min_duration = 3 + random.randint(0, 7)
    malfunction_max_duration = 20 + random.randint(0, 80)

    rail_generator = sparse_rail_generator(max_num_cities=nr_cities, seed=random_seed, grid_mode=False,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rail_pairs_in_city=max_rails_in_cities)

    stochastic_data = MalfunctionParameters(malfunction_rate=malfunction_rate,  # Rate of malfunction occurence
                                        min_duration=malfunction_min_duration,  # Minimal duration of malfunction
                                        max_duration=malfunction_max_duration  # Max duration of malfunction
                                        )

    line_generator = sparse_line_generator({1.: 0.25, 1. / 2.: 0.25, 1. / 3.: 0.25, 1. / 4.: 0.25})

    while width <= max_width and height <= max_height:
        try:
            env = RailEnv(width=width, height=height, rail_generator=rail_generator,
                          line_generator=line_generator, number_of_agents=nr_trains,
                        #   malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                          malfunction_generator=ParamMalfunctionGen(stochastic_data),
                          obs_builder_object=observation_builder, remove_agents_at_target=False)

            print("[{}] {}x{} {} cities {} trains, max {} rails between cities, max {} rails in cities. Malfunction rate {}, {} to {} steps.".format(
                random_seed, width, height, nr_cities, nr_trains, max_rails_between_cities,
                max_rails_in_cities, malfunction_rate, malfunction_min_duration, malfunction_max_duration
            ))

            return env
        except ValueError as e:
            logging.error(f"Error: {e}")
            width += 5
            height += 5
            logging.info("Try again with larger env: (w,h):", width, height)
    logging.error(f"Unable to generate env with seed={random_seed}, max_width={max_height}, max_height={max_height}")
    return None


def sparse_env_small(random_seed, observation_builder):
    width = 30  # With of map
    height = 30  # Height of map
    nr_trains = 2  # Number of trains that have an assigned task in the env
    cities_in_map = 3  # Number of cities where agents can start or end
    seed = 10  # Random seed
    grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
    max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
    max_rail_in_cities = 6  # Max number of parallel tracks within a city, representing a realistic trainstation

    rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                        seed=seed,
                                        grid_mode=grid_distribution_of_cities,
                                        max_rails_between_cities=max_rails_between_cities,
                                        max_rail_pairs_in_city=max_rail_in_cities,
                                        )

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # We can now initiate the schedule generator with the given speed profiles

    line_generator = sparse_rail_generator(speed_ration_map)

    # We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
    # during an episode.

    stochastic_data = MalfunctionParameters(malfunction_rate=1/10000,  # Rate of malfunction occurence
                                            min_duration=15,  # Minimal duration of malfunction
                                            max_duration=50  # Max duration of malfunction
                                            )

    rail_env = RailEnv(width=width,
                height=height,
                rail_generator=rail_generator,
                line_generator=line_generator,
                number_of_agents=nr_trains,
                obs_builder_object=observation_builder,
                # malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                malfunction_generator=ParamMalfunctionGen(stochastic_data),
                remove_agents_at_target=True)

    return rail_env

def _after_step(self, observation, reward, done, info):
    if not self.enabled: return done

    if type(done)== dict:
        _done_check = done['__all__']
    else:
        _done_check =  done
    if _done_check and self.env_semantics_autoreset:
        # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
        self.reset_video_recorder()
        self.episode_id += 1
        self._flush()

    # Record stats - Disabled as it causes error in multi-agent set up
    # self.stats_recorder.after_step(observation, reward, done, info)
    # Record video
    self.video_recorder.capture_frame()

    return done


def perc_completion(env):
    tasks_finished = 0
    if hasattr(env, "agents_data"):
        agent_data = env.agents_data
    else:
        agent_data = env.agents
    for current_agent in agent_data:
        if current_agent.status == TrainState.DONE:
            tasks_finished += 1

    return 100 * np.mean(tasks_finished / max(
                                1, len(agent_data)))
