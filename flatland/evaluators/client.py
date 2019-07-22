import redis
import json
import os
import glob
import pkg_resources
import sys
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()
import hashlib
import random
from flatland.evaluators import messages

from flatland.envs.rail_env import RailEnv
from flatland.envs.generators import rail_from_file
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FlatlandRemoteClient(object):
    """
        Redis client to interface with flatland-rl remote-evaluation-service
        The Docker container hosts a redis-server inside the container.
        This client connects to the same redis-server, 
        and communicates with the service.
        The service eventually will reside outside the docker container, 
        and will communicate
        with the client only via the redis-server of the docker container.
        On the instantiation of the docker container, one service will be 
        instantiated parallely.
        The service will accepts commands at "`service_id`::commands"
        where `service_id` is either provided as an `env` variable or is
        instantiated to "flatland_rl_redis_service_id"
    """
    def __init__(self,  
                remote_host='127.0.0.1',
                remote_port=6379,
                remote_db=0,
                remote_password=None,
                verbose=False
                ):

        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_db = remote_db
        self.remote_password = remote_password
        self.redis_pool = redis.ConnectionPool(
                                host=remote_host,
                                port=remote_port,
                                db=remote_db,
                                password=remote_password)
        self.namespace = "flatland-rl"
        try:
            self.service_id =  os.environ['FLATLAND_RL_SERVICE_ID']
        except KeyError:
            self.service_id = "FLATLAND_RL_SERVICE_ID"
        self.command_channel = "{}::{}::commands".format(
                                    self.namespace,
                                    self.service_id
                                )
        self.verbose = verbose

        self.env = None
        self.ping_pong()

    def get_redis_connection(self):
        return redis.Redis(connection_pool=self.redis_pool)

    def _generate_response_channel(self):
        random_hash = hashlib.md5(
                        "{}".format(
                                random.randint(0, 10**10)
                            ).encode('utf-8')).hexdigest()
        response_channel = "{}::{}::response::{}".format(   self.namespace,
                                                            self.service_id,
                                                            random_hash)
        return response_channel

    def _blocking_request(self, _request):
        """
            request:
                -command_type
                -payload
                -response_channel
            response: (on response_channel)
                - RESULT
            * Send the payload on command_channel (self.namespace+"::command")
                ** redis-left-push (LPUSH)
            * Keep listening on response_channel (BLPOP)
        """
        assert type(_request) ==type({})
        _request['response_channel'] = self._generate_response_channel()

        _redis = self.get_redis_connection()
        """
            The client always pushes in the left
            and the service always pushes in the right
        """
        if self.verbose: print("Request : ", _response)
        # Push request in command_channels
        # Note: The patched msgpack supports numpy arrays
        payload = msgpack.packb(_request, default=m.encode, use_bin_type=True)
        _redis.lpush(self.command_channel, payload)
        # Wait with a blocking pop for the response
        _response = _redis.blpop(_request['response_channel'])[1]
        if self.verbose: print("Response : ", _response)
        _response = msgpack.unpackb(
                        _response, 
                        object_hook=m.decode, 
                        encoding="utf8")
        if _response['type'] == messages.FLATLAND_RL.ERROR:
            raise Exception(str(_response["payload"]))
        else:
            return _response

    def ping_pong(self):
        """
            Official Handshake with the evaluation service
            Send a PING
            and wait for PONG
            If not PONG, raise error
        """
        _request = {}
        _request['type'] = messages.FLATLAND_RL.PING
        _request['payload'] = {}
        _response = self._blocking_request(_request)
        if _response['type'] != messages.FLATLAND_RL.PONG:
            raise Exception(
                "Unable to perform handshake with the redis service. \
                Expected PONG; received {}".format(json.dumps(_response)))
        else:
            return True

    def env_create(self):
        _request = {}
        _request['type'] = messages.FLATLAND_RL.ENV_CREATE
        _request['payload'] = {}
        _response = self._blocking_request(_request)
        observation = _response['payload']['observation']

        if not observation:
            # If the observation is False,
            # then the evaluations are complete
            # hence return false
            return observation

        test_env_file_path = _response['payload']['env_file_path']
        self.env = RailEnv(
            width=1,
            height=1,
            rail_generator=rail_from_file(test_env_file_path),
            obs_builder_object=TreeObsForRailEnv(
                                max_depth=3, 
                                predictor=ShortestPathPredictorForRailEnv()
                                )
        )
        self.env._max_episode_steps = \
            int(1.5 * (self.env.width + self.env.height))

        _ = self.env.reset()
        # Use the observation from the remote service instead
        return observation

    def env_step(self, action, render=False):
        """
            Respond with [observation, reward, done, info]
        """
        _request = {}
        _request['type'] = messages.FLATLAND_RL.ENV_STEP
        _request['payload'] = {}
        _request['payload']['action'] = action
        _response = self._blocking_request(_request)
        _payload = _response['payload']
        observation = _payload['observation']
        reward = _payload['reward']
        done = _payload['done']
        info = _payload['info']
        return [observation, reward, done, info]

    def submit(self):
        _request = {}
        _request['type'] = messages.FLATLAND_RL.ENV_SUBMIT
        _request['payload'] = {}
        _response = self._blocking_request(_request)
        if os.getenv("AICROWD_BLOCKING_SUBMIT"):
            """
            If the submission is supposed to happen as a blocking submit,
            then wait indefinitely for the evaluator to decide what to 
            do with the container.
            """
            while True:
                time.sleep(10)
        return _response['payload']

if __name__ == "__main__":
    env_client = FlatlandRemoteClient()
    def my_controller(obs, _env):
        _action = {}
        for _idx, _ in enumerate(_env.agents):
            _action[_idx] = np.random.randint(0, 5)
        return _action

    episode = 0
    obs = True
    while obs:
        obs = env_client.env_create()
        if not obs:
            break
        print("Episode : {}".format(episode))
        episode += 1

        print(env_client.env.dones['__all__'])

        while True:
            action = my_controller(obs, env_client.env)
            observation, all_rewards, done, info = env_client.env_step(action)
            if done['__all__']:
                print("Current Episode : ", episode)
                print("Episode Done")
                print("Reward : ", sum(list(all_rewards.values())))
                break

    print("Evaluation Complete...")       
    print(env_client.submit())


