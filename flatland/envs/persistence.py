

import pickle
import msgpack
import numpy as np

from flatland.envs import rail_env 

#from flatland.core.env import Environment
from flatland.core.env_observation_builder import DummyObservationBuilder
#from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
#from flatland.core.grid.grid4_utils import get_new_position
#from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import Agent, EnvAgent, RailAgentStatus
from flatland.envs.distance_map import DistanceMap

#from flatland.envs.observations import GlobalObsForRailEnv

# cannot import objects / classes directly because of circular import
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import schedule_generators as sched_gen


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
                #pickle.dump(env_dict, file_out)

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
    def load_new(cls, filename, load_from_package=None):

        env_dict = cls.load_env_dict(filename, load_from_package=load_from_package)


        # TODO: inefficient - each one of these generators loads the complete env file.
        env = rail_env.RailEnv(width=1, height=1,
                rail_generator=rail_gen.rail_from_file(filename, 
                    load_from_package=load_from_package),
                schedule_generator=sched_gen.schedule_from_file(filename,
                    load_from_package=load_from_package),
                malfunction_generator_and_process_data=mal_gen.malfunction_from_file(filename,
                    load_from_package=load_from_package),
                obs_builder_object=DummyObservationBuilder(),
                record_steps=True)

        env.rail = GridTransitionMap(1,1) # dummy        

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
            env_dict = msgpack.unpackb(load_data, use_list=False, encoding="utf-8")
        elif filename.endswith("pkl"):
            env_dict = pickle.loads(load_data)
        else:
            print(f"filename {filename} must end with either pkl or mpk")
            env_dict = {}
        
        # Replace the agents tuple with EnvAgent objects
        if "agents_static" in env_dict:
            env_dict["agents"] = EnvAgent.load_legacy_static_agent(env_dict["agents_static"])
            # remove the legacy key
            del env_dict["agents_static"]
        elif "agents" in env_dict:
            env_dict["agents"] = [EnvAgent(*d[0:12]) for d in env_dict["agents"]]

        return env_dict

    @classmethod
    def load_resource(cls, package, resource):
        """
        Load environment (with distance map?) from a binary
        """
        #from importlib_resources import read_binary
        #load_data = read_binary(package, resource)

        #if resource.endswith("pkl"):
        #    env_dict = pickle.loads(load_data)
        #elif resource.endswith("mpk"):
        #    env_dict = msgpack.unpackb(load_data, encoding="utf-8")
        
        #cls.set_full_state(env, env_dict)

        return cls.load_new(resource, load_from_package=package)

    @classmethod
    def set_full_state(cls, env, env_dict):
        """
        Sets environment state from env_dict 

        Parameters
        -------
        env_dict: dict
        """
        env.rail.grid = np.array(env_dict["grid"])

        # Initialise the env with the frozen agents in the file
        env.agents = env_dict.get("agents", [])

        # For consistency, set number_of_agents, which is the number which will be generated on reset
        env.number_of_agents = env.get_num_agents()

        env.height, env.width = env.rail.grid.shape
        env.rail.height = env.height
        env.rail.width = env.width
        env.dones = dict.fromkeys(list(range(env.get_num_agents())) + ["__all__"], False)

    @classmethod
    def get_full_state(cls, env):
        """
        Returns state of environment in dict object, ready for serialization

        """
        grid_data = env.rail.grid.tolist()

        # msgpack cannot persist EnvAgent so use the Agent namedtuple.
        agent_data = [agent.to_agent() for agent in env.agents]
        #print("get_full_state - agent_data:", agent_data)
        malfunction_data: MalfunctionProcessData = env.malfunction_process_data

        msg_data_dict = {
            "grid": grid_data,
            "agents": agent_data,
            "malfunction": malfunction_data,
            "max_episode_steps": env._max_episode_steps,
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
        #msgpack.packb(grid_data, use_bin_type=True)
        #msgpack.packb(agent_data, use_bin_type=True)

        distance_map_data = self.distance_map.get()
        malfunction_data: MalfunctionProcessData = self.malfunction_process_data
        #msgpack.packb(distance_map_data, use_bin_type=True)  # does nothing
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