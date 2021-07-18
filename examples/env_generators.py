import logging
import random
import numpy as np
from typing import NamedTuple

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.agent_utils import RailAgentStatus

MalfunctionParameters = NamedTuple('MalfunctionParameters', [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])


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
                                           max_rails_in_city=max_rails_in_cities)

    # new version:
    # stochastic_data = MalfunctionParameters(malfunction_rate, malfunction_min_duration, malfunction_max_duration)

    stochastic_data = {'malfunction_rate': malfunction_rate, 'min_duration': malfunction_min_duration,
                       'max_duration': malfunction_max_duration}

    schedule_generator = sparse_schedule_generator({1.: 0.25, 1. / 2.: 0.25, 1. / 3.: 0.25, 1. / 4.: 0.25})

    while width <= max_width and height <= max_height:
        try:
            env = RailEnv(width=width, height=height, rail_generator=rail_generator,
                          schedule_generator=schedule_generator, number_of_agents=nr_trains,
                          malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
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
                                        max_rails_in_city=max_rail_in_cities,
                                        )

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # We can now initiate the schedule generator with the given speed profiles

    schedule_generator = sparse_schedule_generator(speed_ration_map)

    # We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
    # during an episode.

    stochastic_data = MalfunctionParameters(malfunction_rate=1/10000,  # Rate of malfunction occurence
                                            min_duration=15,  # Minimal duration of malfunction
                                            max_duration=50  # Max duration of malfunction
                                            )

    rail_env = RailEnv(width=width,
                height=height,
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
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
    for current_agent in env.agents_data:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1

    return 100 * np.mean(tasks_finished / max(
                                1, env.num_agents)) 