#!/usr/bin/env python
from __future__ import print_function
import redis

from flatland.envs.generators import rail_from_file
from flatland.envs.rail_env import RailEnv

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


from flatland.evaluators import messages

import json
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()
import flatland
import os
import timeout_decorator
import time

########################################################
# CONSTANTS
########################################################
PER_STEP_TIMEOUT = 5*60 # 5 minutes


class FlatlandRemoteEvaluationService:
    def __init__(   self,
                    test_env_folder="/tmp",
                    flatland_rl_service_id = 'FLATLAND_RL_SERVICE_ID',
                    remote_host = '127.0.0.1',
                    remote_port = 6379,
                    remote_db = 0,
                    remote_password = None,
                    visualize = False,
                    report = None,
                    verbose = False):


        # Test Env folder Paths
        self.test_env_folder = test_env_folder
        self.env_file_paths = self.get_env_filepaths()

        # Logging and Reporting related vars
        self.verbose = verbose
        self.report = report
        
        # Communication Protocol Related vars
        self.namespace = "flatland-rl"
        self.service_id = flatland_rl_service_id
        self.command_channel = "{}::{}::commands".format(
                                    self.namespace, 
                                    self.service_id
                                )
        
        # Message Broker related vars
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_db = remote_db
        self.remote_password = remote_password
        self.instantiate_redis_connection_pool()
        
        # RailEnv specific variables
        self.env = False
        self.env_available = False
        self.reward = 0
        self.simulation_count = 0
        self.simualation_rewards = []
        self.simulation_times = []
        self.begin_simulation = False
        self.current_step = 0
        self.visualize = visualize



    def get_env_filepaths(self):
        env_paths = []
        folder_path = self.test_env_folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pkl"):
                    env_paths.append(
                        os.path.join(root, file)
                        )
        return sorted(env_paths)        

    def instantiate_redis_connection_pool(self):
        if self.verbose or self.report:
            print("Attempting to connect to redis server at {}:{}/{}".format(
                    self.remote_host, 
                    self.remote_port, 
                    self.remote_db)
                )

        self.redis_pool = redis.ConnectionPool(
                            host=self.remote_host, 
                            port=self.remote_port, 
                            db=self.remote_db, 
                            password=self.remote_password
                        )

    def get_redis_connection(self):
        redis_conn = redis.Redis(connection_pool=self.redis_pool)
        try:
            redis_conn.ping()
        except:
            raise Exception(
                    "Unable to connect to redis server at {}:{} ."
                    "Are you sure there is a redis-server running at the "
                    "specified location ?".format(
                        self.remote_host,
                        self.remote_port
                        )
                    )
        return redis_conn

    def _error_template(self, payload):
        _response = {}
        _response['type'] = messages.FLATLAND_RL.ERROR
        _response['payload'] = payload
        return _response

    @timeout_decorator.timeout(PER_STEP_TIMEOUT)# timeout for each command
    def _get_next_command(self, _redis):
        command = _redis.brpop(self.command_channel)[1]
        return command
    
    def get_next_command(self):
        try:
            _redis = self.get_redis_connection()
            command = self._get_next_command(_redis)
            if self.verbose or self.report:
                print("Command Service: ", command)
        except timeout_decorator.timeout_decorator.TimeoutError:
            raise Exception(
                    "Timeout in step {} of simulation {}".format(
                            self.current_step,
                            self.simulation_count
                            ))
        command_response_channel = "default_response_channel"
        command = msgpack.unpackb(
                    command, 
                    object_hook=m.decode, 
                    encoding="utf8"
                )
        if self.verbose:
            print("Received Request : ", command)
        
        return command

    def handle_ping(self, command):
        _redis = self.get_redis_connection()
        command_response_channel = command['response_channel']

        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.PONG
        _command_response['payload'] = {}
        if self.verbose: print("Responding with : ", _command_response)
        _redis.rpush(
            command_response_channel, 
            msgpack.packb(
                _command_response, 
                default=m.encode, 
                use_bin_type=True)
        )

    def handle_env_create(self, command):
        _redis = self.get_redis_connection()
        command_response_channel = command['response_channel']
        _payload = command['payload']

        if self.simulation_count < len(self.env_file_paths):
            """
            There are still test envs left that are yet to be evaluated 
            """

            test_env_file_path = self.env_file_paths[self.simulation_count]
            del self.env
            self.env = RailEnv(
                width=1,
                height=1,
                rail_generator=rail_from_file(test_env_file_path),
                obs_builder_object=TreeObsForRailEnv(
                                    max_depth=3, 
                                    predictor=ShortestPathPredictorForRailEnv()
                                    )
            )
            
            # Set max episode steps allowed
            self.env._max_episode_steps = \
                int(1.5 * (self.env.width + self.env.height))

            self.env_available = True


            self.simulation_count += 1

            if self.begin_simulation:
                # If begin simulation has already been initialized 
                # atleast once
                self.simulation_times.append(time.time()-self.begin_simulation)
            self.begin_simulation = time.time()

            self.simualation_rewards.append(0)
            self.current_step = 0

            _observation = self.env.reset()

            _command_response = {}
            _command_response['type'] = messages.FLATLAND_RL.ENV_CREATE_RESPONSE
            _command_response['payload'] = {}
            _command_response['payload']['observation'] = _observation
            _command_response['payload']['env_file_path'] = test_env_file_path
            if self.verbose: print("Responding with : ", _command_response)
            _redis.rpush(
                command_response_channel, 
                msgpack.packb(
                    _command_response, 
                    default=m.encode, 
                    use_bin_type=True)
                )
        else:
            """
            All test env evaluations are complete
            """
            _command_response = {}
            _command_response['type'] = messages.FLATLAND_RL.ENV_RESET_RESPONSE
            _command_response['payload'] = {}
            _command_response['payload']['observation'] = False
            _command_response['payload']['env_file_path'] = False            
            if self.verbose: print("Responding with : ", _command_response)
            _redis.rpush(
                command_response_channel, 
                msgpack.packb(
                    _command_response, 
                    default=m.encode, 
                    use_bin_type=True)
                )


    def run(self):
        print("Listening for commands at : ", self.command_channel)
        while True:
            command = self.get_next_command()

            if self.verbose:
                print("Self.Reward : ", self.reward)
                print("Current Simulation : ", self.simulation_count)
                if self.env_file_paths \
                    and self.simulation_count < len(self.env_file_paths):
                    print("Current Env Path : ", \
                        self.env_file_paths[self.simulation_count]
                    )

            try:                
                if command['type'] == messages.FLATLAND_RL.PING:
                    """
                        INITIAL HANDSHAKE : Respond with PONG
                    """
                    self.handle_ping(command)
                
                elif command['type'] == messages.FLATLAND_RL.ENV_CREATE:
                    """
                        ENV_CREATE

                        Respond with an internal _env object
                    """
                    self.handle_env_create(command)
                elif command['type'] == messages.FLATLAND_RL.ENV_RESET:
                    """
                        ENV_RESET

                        Respond with observation from next simulation or
                        False if no simulations are left
                    """
                    self.simulation_count += 1
                    if self.begin_simulation:
                        self.simulation_times.append(time.time()-self.begin_simulation)
                        self.begin_simulation = time.time()
                    if self.seed_map and self.simulation_count < len(self.seed_map):
                        _observation = self.env.reset(seed=self.seed_map[self.simulation_count], project=False)
                        self.simualation_rewards.append(0)
                        self.env_available = True
                        self.current_step = 0
                        #_observation = list(_observation)

                        _command_response = {}
                        _command_response['type'] = messages.FLATLAND_RL.ENV_RESET_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = _observation
                        if self.verbose: print("Responding with : ", _command_response)
                        _redis.rpush(command_response_channel, msgpack.packb(_command_response, default=m.encode, use_bin_type=True))
                    else:
                        _command_response = {}
                        _command_response['type'] = messages.FLATLAND_RL.ENV_RESET_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = False
                        if self.verbose: print("Responding with : ", _command_response)
                        _redis.rpush(command_response_channel, msgpack.packb(_command_response, default=m.encode, use_bin_type=True))
                elif command['type'] == messages.FLATLAND_RL.ENV_STEP:
                    """
                        ENV_STEP

                        Request : Action array
                        Respond with updated [observation,reward,done,info] after step
                    """
                    args = command['payload']
                    action = args['action']
                    if self.env and self.env_available:
                        [_observation, reward, done, info] = self.env.step(action)
                    else:
                        if self.env:
                            raise Exception("Attempt to call `step` function after max_steps={} in a single simulation. Please reset your environment before calling the `step` function after max_step s".format(self.max_steps))
                        else:
                                raise Exception("Attempt to call `step` function on a non existent `env`")
                    self.reward += reward
                    self.simualation_rewards[-1] += reward
                    self.current_step += 1
                    #_observation = np.array(_observation).tolist()

                    if self.current_step >= self.max_steps:
                        _command_response = {}
                        _command_response['type'] = messages.FLATLAND_RL.ENV_STEP_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = _observation
                        _command_response['payload']['reward'] = reward
                        _command_response['payload']['done'] = True
                        _command_response['payload']['info'] = info

                        """
                        Mark env as unavailable until next reset
                        """
                        self.env_available = False
                    else:
                        _command_response = {}
                        _command_response['type'] = messages.FLATLAND_RL.ENV_STEP_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = _observation
                        _command_response['payload']['reward'] = reward
                        _command_response['payload']['done'] = done
                        _command_response['payload']['info'] = info

                        if done:
                            """
                                Mark env as unavailable until next reset
                            """
                            self.env_available = False
                    if self.verbose: print("Responding with : ", _command_response)
                    if self.verbose: print("Current Step : ", self.current_step)
                    _redis.rpush(command_response_channel, msgpack.packb(_command_response, default=m.encode, use_bin_type=True))
                elif command['type'] == messages.FLATLAND_RL.ENV_SUBMIT:
                    """
                        ENV_SUBMIT

                        Submit the final cumulative reward
                    """
                    _response = {}
                    _response['type'] = messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE
                    _payload = {}
                    _payload['mean_reward'] = np.float(self.reward)/len(self.seed_map) #Mean reward
                    _payload['simulation_rewards'] = self.simualation_rewards
                    _payload['simulation_times'] = self.simulation_times
                    _response['payload'] = _payload
                    _redis.rpush(command_response_channel, msgpack.packb(_response, default=m.encode, use_bin_type=True))
                elif command['type'] == messages.FLATLAND_RL.ENV_SUBMIT:
                    if self.verbose: print("Responding with : ", _response)
                    return _response
                else:
                    _error = self._error_template(
                                    "UNKNOWN_REQUEST:{}".format(
                                        str(command)))
                    if self.verbose:print("Responding with : ", _error)
                    _redis.rpush(command_response_channel, msgpack.packb(_error, default=m.encode, use_bin_type=True))
                    return _error

            except Exception as e:
                print("Error : ", str(e))
                _redis.rpush(   command_response_channel,
                                msgpack.packb(self._error_template(str(e)), default=m.encode, use_bin_type=True))
                return self._error_template(str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Submit the result to AIcrowd')
    parser.add_argument('--service_id', dest='service_id', default='FLATLAND_RL_SERVICE_ID', required=False)
    parser.add_argument('--test_folder',
                        dest='test_folder',
                        default="/Users/spmohanty/work/SBB/submission-scoring/Envs-Small",
                        help="Folder containing the pickle files for the test envs",
                        required=False)
    args = parser.parse_args()
    
    test_folder = args.test_folder

    grader = FlatlandRemoteEvaluationService(
                test_env_folder=test_folder,
                flatland_rl_service_id=args.service_id,
                verbose=True
                )
    result = grader.run()
    if result['type'] == messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE:
        cumulative_results = result['payload']
        print("Results : ", cumulative_results)
    elif result['type'] == messages.FLATLAND_RL.ERROR:
        error = result['payload']
        raise Exception("Evaluation Failed : {}".format(str(error)))
    else:
        #Evaluation failed
        print("Evaluation Failed : ", result['payload'])
