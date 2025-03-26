import pickle
from typing import Tuple, Dict

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs import rail_env
from flatland.utils.seeding import random_state_to_hashablestate
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import EnvAgent, load_env_agent

# cannot import objects / classes directly because of circular import
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import line_generators as line_gen
from flatland.envs import timetable_generators as tt_gen


class RailEnvPersister(object):

    @classmethod
    def save(cls, env, filename, save_distance_maps=False):
        """
        Saves environment and distance map information in a file

        Parameters:
        ---------
        filename: string
        save_distance_maps: bool
        """

        env_dict = cls.get_full_state(env)

        # We have an unresolved problem with msgpack loading the list of agents
        # see also 20 lines below.
        # print(f"env save - agents: {env_dict['agents'][0]}")
        # a0 = env_dict["agents"][0]
        # print("agent type:", type(a0))

        if save_distance_maps is True:
            oDistMap = env.distance_map.get()
            if oDistMap is not None:
                if len(oDistMap) > 0:
                    env_dict["distance_map"] = oDistMap
                else:
                    print("[WARNING] Unable to save the distance map for this environment, as none was found !")
            else:
                print("[WARNING] Unable to save the distance map for this environment, as none was found !")

        with open(filename, "wb") as file_out:

            if filename.endswith("mpk"):
                data = msgpack.packb(env_dict)


            elif filename.endswith("pkl"):
                data = pickle.dumps(env_dict)
                # pickle.dump(env_dict, file_out)

            file_out.write(data)

        # We have an unresovled problem with msgpack loading the list of Agents
        # with open(filename, "rb") as file_in:
        # if filename.endswith("mpk"):
        # bytes_in = file_in.read()
        # dIn = msgpack.unpackb(data, encoding="utf-8")
        # print(f"msgpack check - {dIn.keys()}")
        # print(f"msgpack check - {dIn['agents'][0]}")

    @classmethod
    def save_episode(cls, env, filename):
        dict_env = cls.get_full_state(env)

        # Add additional info to dict_env before saving
        dict_env["episode"] = env.cur_episode
        dict_env["actions"] = env.list_actions
        dict_env["shape"] = (env.width, env.height)
        dict_env["max_episode_steps"] = env._max_episode_steps

        with open(filename, "wb") as file_out:
            if filename.endswith(".mpk"):
                file_out.write(msgpack.packb(dict_env))
            elif filename.endswith(".pkl"):
                pickle.dump(dict_env, file_out)

    @classmethod
    def load(cls, env, filename, load_from_package=None):
        """
        Load environment with distance map from a file

        Parameters:
        -------
        filename: string
        """
        env_dict = cls.load_env_dict(filename, load_from_package=load_from_package)
        cls.set_full_state(env, env_dict)

    @classmethod
    def load_new(cls, filename, load_from_package=None) -> Tuple["RailEnv", Dict]:

        env_dict = cls.load_env_dict(filename, load_from_package=load_from_package)

        llGrid = env_dict["grid"]
        height = len(llGrid)
        width = len(llGrid[0])

        # TODO: inefficient - each one of these generators loads the complete env file.
        env = rail_env.RailEnv(
            width=width, height=height,
            rail_generator=rail_gen.rail_from_file(filename,
                                                   load_from_package=load_from_package),
            line_generator=line_gen.line_from_file(filename,
                                                   load_from_package=load_from_package),
            timetable_generator=tt_gen.timetable_from_file(filename, load_from_package=load_from_package),
            # malfunction_generator_and_process_data=mal_gen.malfunction_from_file(filename,
            #    load_from_package=load_from_package),
            malfunction_generator=mal_gen.FileMalfunctionGen(env_dict),
            obs_builder_object=DummyObservationBuilder(),
            record_steps=True)

        env.rail = RailGridTransitionMap(1, 1)  # dummy

        # TODO bad code smell - agent_position initialized in reset() only.
        env.agent_positions = np.zeros((env.height, env.width), dtype=int) - 1

        cls.set_full_state(env, env_dict)
        return env, env_dict

    @classmethod
    def load_env_dict(cls, filename, load_from_package=None):

        if load_from_package is not None:
            from importlib_resources import read_binary
            load_data = read_binary(load_from_package, filename)
        else:
            with open(filename, "rb") as file_in:
                load_data = file_in.read()

        if filename.endswith("mpk"):
            env_dict = msgpack.unpackb(load_data, use_list=False, raw=False)
        elif filename.endswith("pkl"):
            try:
                env_dict = pickle.loads(load_data)
            except ValueError:
                print("pickle failed to load file:", filename, " trying msgpack (deprecated)...")
                env_dict = msgpack.unpackb(load_data, use_list=False, raw=False)
        else:
            print(f"filename {filename} must end with either pkl or mpk")
            env_dict = {}

        # Replace the agents tuple with EnvAgent objects
        if "agents_static" in env_dict:
            env_dict["agents"] = EnvAgent.load_legacy_static_agent(env_dict["agents_static"])
            # remove the legacy key
            del env_dict["agents_static"]
        elif "agents" in env_dict:
            # env_dict["agents"] = [EnvAgent(*d[0:len(d)]) for d in env_dict["agents"]]
            env_dict["agents"] = [load_env_agent(d) for d in env_dict["agents"]]

        return env_dict

    @classmethod
    def load_resource(cls, package, resource):
        """
        Load environment (with distance map?) from a binary
        """
        # from importlib_resources import read_binary
        # load_data = read_binary(package, resource)

        # if resource.endswith("pkl"):
        #    env_dict = pickle.loads(load_data)
        # elif resource.endswith("mpk"):
        #    env_dict = msgpack.unpackb(load_data, encoding="utf-8")

        # cls.set_full_state(env, env_dict)

        return cls.load_new(resource, load_from_package=package)

    @classmethod
    def set_full_state(cls, env, env_dict):
        """
        Sets environment state from env_dict

        Parameters
        -------
        env_dict: dict
        """
        grid = np.array(env_dict["grid"])

        # Initialise the env with the frozen agents in the file
        env.agents = env_dict.get("agents", [])

        # For consistency, set number_of_agents, which is the number which will be generated on reset
        env.number_of_agents = env.get_num_agents()

        env.height, env.width = grid.shape

        # use new rail object instance for lru cache scoping and garbage collection to work properly
        env.rail = RailGridTransitionMap(height=env.height, width=env.width)
        env.rail.grid = grid
        env.dones = dict.fromkeys(list(range(env.get_num_agents())) + ["__all__"], False)

        max_episode_steps = env_dict.get('max_episode_steps', None)
        if max_episode_steps is not None:
            env._max_episode_steps = max_episode_steps
        _elapsed_steps = env_dict.get("elapsed_steps", None)
        if _elapsed_steps is not None:
            env._elapsed_steps = _elapsed_steps

        env.distance_map.distance_map = env_dict.get('distance_map', None)
        env.distance_map.reset(env.agents, env.rail)
        env.distance_map._compute(env.agents, env.rail)

        random_seed = env.random_seed = env_dict.get("random_seed", None)
        if random_seed is not None:
            env.random_seed = random_seed

        seed_history = env_dict.get("seed_history", None)
        if seed_history is not None:
            env.seed_history = seed_history

        # it's not sufficient to store random_seed, as seeding from random_seed is done
        # at start of reset (before rail/line/timetable (re-)generation,
        # hence np_random depends on rail/line/timetable generation
        np_random_state = env_dict.get("np_random_state", None)
        if np_random_state is not None:
            env.np_random.set_state(np_random_state)
        dev_pred_dict_ = env_dict.get("dev_pred_dict", None)
        if dev_pred_dict_ is not None:
            env.dev_pred_dict = dev_pred_dict_
        dev_obs_dict_ = env_dict.get("dev_obs_dict", None)
        if dev_pred_dict_ is not None:
            env.dev_obs_dict = dev_obs_dict_

        malfunction_cached_rand = env_dict.get("malfunction_cached_rand", None)
        malfunction_rand_idx = env_dict.get("malfunction_rand_idx", None)
        if malfunction_cached_rand is not None:
            env.malfunction_generator._cached_rand = malfunction_cached_rand
        if malfunction_rand_idx is not None:
            env.malfunction_generator._rand_idx = malfunction_rand_idx

    @classmethod
    def get_full_state(cls, env):
        """
        Returns state of environment in dict object, ready for serialization

        """
        grid_data = env.rail.grid.tolist()

        # msgpack cannot persist EnvAgent so use the Agent namedtuple.
        agent_data = [agent.to_agent() for agent in env.agents]
        malfunction_data: mal_gen.MalfunctionProcessData = env.malfunction_process_data

        msg_data_dict = {
            "grid": grid_data,
            "agents": agent_data,
            "malfunction": malfunction_data,
            "malfunction_cached_rand": env.malfunction_generator._cached_rand if hasattr(env.malfunction_generator, '_cached_rand') else None,
            "malfunction_rand_idx": env.malfunction_generator._rand_idx if hasattr(env.malfunction_generator, '_rand_idx') else None,
            "max_episode_steps": env._max_episode_steps,
            "elapsed_steps": env._elapsed_steps,
            "random_seed": env.random_seed,
            "seed_history": env.seed_history,
            "np_random_state": random_state_to_hashablestate(env.np_random),
            "dev_pred_dict": env.dev_pred_dict,
            "dev_obs_dict": env.dev_obs_dict,
        }
        return msg_data_dict

    ################################################################################################
    # deprecated methods moved from RailEnv.  Most likely broken.

    def deprecated_get_full_state_msg(self) -> msgpack.Packer:
        """
        Returns state of environment in msgpack object
        """
        msg_data_dict = self.get_full_state_dict()
        return msgpack.packb(msg_data_dict, use_bin_type=True)

    def deprecated_get_agent_state_msg(self) -> msgpack.Packer:
        """
        Returns agents information in msgpack object
        """
        agent_data = [agent.to_agent() for agent in self.agents]
        msg_data = {
            "agents": agent_data}
        return msgpack.packb(msg_data, use_bin_type=True)

    def deprecated_get_full_state_dist_msg(self) -> msgpack.Packer:
        """
        Returns environment information with distance map information as msgpack object
        """
        grid_data = self.rail.grid.tolist()
        agent_data = [agent.to_agent() for agent in self.agents]

        # I think these calls do nothing - they create packed data and it is discarded
        # msgpack.packb(grid_data, use_bin_type=True)
        # msgpack.packb(agent_data, use_bin_type=True)

        distance_map_data = self.distance_map.get()
        malfunction_data: mal_gen.MalfunctionProcessData = self.malfunction_process_data
        # msgpack.packb(distance_map_data, use_bin_type=True)  # does nothing
        msg_data = {
            "grid": grid_data,
            "agents": agent_data,
            "distance_map": distance_map_data,
            "malfunction": malfunction_data}
        return msgpack.packb(msg_data, use_bin_type=True)

    def deprecated_set_full_state_msg(self, msg_data):
        """
        Sets environment state with msgdata object passed as argument

        Parameters
        -------
        msg_data: msgpack object
        """
        data = msgpack.unpackb(msg_data, use_list=False, encoding='utf-8')
        self.rail.grid = np.array(data["grid"])
        # agents are always reset as not moving
        if "agents_static" in data:
            self.agents = EnvAgent.load_legacy_static_agent(data["agents_static"])
        else:
            self.agents = [EnvAgent(*d[0:12]) for d in data["agents"]]
        # setup with loaded data
        self.height, self.width = self.rail.grid.shape
        self.rail.height = self.height
        self.rail.width = self.width
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

    def deprecated_set_full_state_dist_msg(self, msg_data):
        """
        Sets environment grid state and distance map with msgdata object passed as argument

        Parameters
        -------
        msg_data: msgpack object
        """
        data = msgpack.unpackb(msg_data, use_list=False, encoding='utf-8')
        self.rail.grid = np.array(data["grid"])
        # agents are always reset as not moving
        if "agents_static" in data:
            self.agents = EnvAgent.load_legacy_static_agent(data["agents_static"])
        else:
            self.agents = [EnvAgent(*d[0:12]) for d in data["agents"]]
        if "distance_map" in data.keys():
            self.distance_map.set(data["distance_map"])
        # setup with loaded data
        self.height, self.width = self.rail.grid.shape
        self.rail.height = self.height
        self.rail.width = self.width
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)
