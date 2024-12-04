import os
import pickle

import numpy as np
import pytest
from benchmarker import Benchmarker

from flatland.core.transition_map import GridTransitionMap
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.serialization_deserialization_snippet import SerializationDeserializion


class SerializationDeserializion:

    @staticmethod
    def _serialize_rail_env(rail_env: RailEnv, save_distance_maps: bool) -> list[any]:
        """Supports the serialization of the core_env.
        It decouples the serialization of the backend from the core_env.
        :return: A list of essential parameters needed to be cloned.
        """
        return [
            rail_env.agents,  # all agents state
            rail_env.np_random.get_state(),  # state of random generator
            rail_env._elapsed_steps,  # number of steps elapsed
            rail_env.num_resets,  # number of resets that have happened
            rail_env.rail,  # the rail matrix
            None,
            # rail_env.dev_pred_dict,  # we use it for caching the prediction of shortest path so we cache it.
            rail_env.dev_obs_dict,  # Not sure what we do with it
            rail_env.dones,  # Arrived trains
            rail_env._max_episode_steps,  # Maximum step within the episode
            rail_env.active_agents,  # Not sure what is this used for..
            rail_env.distance_map.distance_map if save_distance_maps else None,  # distance map for the agents
            rail_env.distance_map.agents_previous_computation,  # distance map other field...
            rail_env.malfunction_generator.MFP,  # malfunction generator parameters
            #            _rail_env_rnd_state_for_malfunctions,  # rnd generator for malfunctions
            rail_env.malfunction_generator._rand_idx,  # pointer to the index for the generator for malfunctions
            rail_env.malfunction_generator._cached_rand is not None,  # flag
        ]

    @staticmethod
    def _deserialize_rail_env(serialised_rail_env: list[any], _rail_env: RailEnv):
        """restore the state of the current rail_env to the state of the given one.
        :param serialised_rail_env: the serial
        """

        # _rail_env.random_seed = _random_seed
        # if _rail_env.seed_history[-1] != _random_seed:
        #     _rail_env.seed_history.append(_random_seed)
        _rail_env.agents = serialised_rail_env[0]
        _rail_env._elapsed_steps = serialised_rail_env[2]
        _rail_env.num_resets = serialised_rail_env[3]
        _rail_env.rail = serialised_rail_env[4]
        # _rail_env.dev_pred_dict = serialised_rail_env[5]
        _rail_env.dev_obs_dict = serialised_rail_env[6]
        _rail_env.dones = serialised_rail_env[7]
        _rail_env._max_episode_steps = serialised_rail_env[8]
        _rail_env.active_agents = serialised_rail_env[9]
        # deserialize distance map
        _rail_env.distance_map.agents = _rail_env.agents
        _rail_env.distance_map.rail = _rail_env.rail
        _rail_env.distance_map.distance_map = serialised_rail_env[10]
        _rail_env.distance_map.agents_previous_computation = serialised_rail_env[11]
        # deserialize MFP
        _rail_env.malfunction_generator.MFP = serialised_rail_env[12]
        # borrow rnd generator and seed it with the state used to generate malfunctions data.
        _rail_env_rnd_state_for_malfunctions = serialised_rail_env[13]
        _rail_env.malfunction_generator._cached_rand = None
        if serialised_rail_env[-1]:
            _rail_env.np_random.set_state(_rail_env_rnd_state_for_malfunctions)
            _rail_env.malfunction_generator.generate_rand_numbers(_rail_env.np_random)
        _rail_env.malfunction_generator._rand_idx = serialised_rail_env[14]

        # restore the true rnd generator
        _rail_env.np_random.set_state(serialised_rail_env[1])
        # restore agent position
        _rail_env.agent_positions = np.zeros((_rail_env.height, _rail_env.width), dtype=int) - 1
        for agent in _rail_env.agents:
            if agent.position is not None:
                _rail_env.agent_positions[agent.position] = agent.handle


def create_env(nAgents=25,
               n_cities=2,
               max_rails_between_cities=2,
               max_rails_in_city=4,
               seed=0,
               width=30,
               height=30,
               ):
    env = RailEnv(
        width=width,
        height=height,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=nAgents,
        obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
    )
    env.reset()
    return env


def readable_size(size2, decimal_point=3):
    for i in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size2 < 1024.0:
            break
        size2 /= 1024.0
    return f"{size2:.{decimal_point}f}{i}"


@pytest.mark.parametrize("width,height,nAgents,save_distance_maps", [(*t, s) for s in [True, False] for t in [
    (30, 30, 25),
    (100, 100, 50),
    (200, 200, 100),
]])
def test_bench_persistence(width, height, nAgents, save_distance_maps):
    env = create_env(width=width, height=height, nAgents=nAgents)
    e = create_env(width=width, height=height, nAgents=nAgents)
    cycle = 20
    sizes = {}
    with Benchmarker(cycle=cycle, extra=1) as bench:
        @bench("RailEnvPersister")
        def _(_):
            RailEnvPersister.save(env, "1234.pkl", save_distance_maps=save_distance_maps)
            RailEnvPersister.load(e, "1234.pkl")

            print(readable_size(os.path.getsize("1234.pkl")))
            sizes["RailEnvPersister"] = readable_size(os.path.getsize("1234.pkl"))

        @bench("get_state+pickle")
        def _(_):
            RailEnvPersister.new_save(env, "1234.pkl", save_distance_maps=save_distance_maps)
            RailEnvPersister.new_load(env, "1234.pkl")
            print(readable_size(os.path.getsize("1234.pkl")))
            sizes["get_state+pickle"] = readable_size(os.path.getsize("1234.pkl"))

        @bench("get_state+msgpack")
        def _(_):
            RailEnvPersister.new_save(env, "1234.mpk", save_distance_maps=save_distance_maps)
            RailEnvPersister.new_load(env, "1234.mpk")
            print(readable_size(os.path.getsize("1234.mpk")))
            sizes["get_state+msgpack"] = readable_size(os.path.getsize("1234.mpk"))

        # TODO dev_obs_dict has int keys values
        # @bench("get_state+json")
        # def _(_):
        #     RailEnvPersister.new_save(env,"1234.json", save_distance_maps)
        #     RailEnvPersister.new_load(env,"1234.json")
        #     print(readable_size(os.path.getsize("1234.json")))
        #     sizes["get_state+json"] = readable_size(os.path.getsize("1234.json"))

        @bench("SerializationDeserializion")
        def _(_):
            with open("1234.pkl", "wb") as file_out:
                data = pickle.dumps(SerializationDeserializion._serialize_rail_env(env, save_distance_maps=save_distance_maps))
                # data = msgpack.packb(SerializationDeserializion(env)._serialize_rail_env())
                file_out.write(data)
            with open("1234.pkl", "rb") as file_in:
                data = pickle.load(file_in)
                rail: GridTransitionMap = data[4]
                SerializationDeserializion._deserialize_rail_env(data, RailEnv(width=rail.width, height=rail.height))
                print(readable_size(os.path.getsize("./1234.pkl")))
                sizes["SerializationDeserializion"] = readable_size(os.path.getsize("1234.pkl"))
    print("=========================================")
    print(f" width={width}, height={height}, nAgents={nAgents}, save_distance_maps={save_distance_maps}")
    print("=========================================")
    print(sizes)
