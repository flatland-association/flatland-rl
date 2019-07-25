import redis
import json
import os
import numpy as np
import msgpack
import msgpack_numpy as m
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
m.patch()


def are_dicts_equal(d1, d2):
    """ return True if all keys and values are the same """
    return all(k in d2 and d1[k] == d2[k]
               for k in d1) \
        and all(k in d1 and d1[k] == d2[k]
               for k in d2)


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
                test_envs_root=None,
                verbose=False):

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
        self.service_id = os.getenv(
                            'FLATLAND_RL_SERVICE_ID',
                            'FLATLAND_RL_SERVICE_ID'
                            )
        self.command_channel = "{}::{}::commands".format(
                                    self.namespace,
                                    self.service_id
                                )
        if test_envs_root:
            self.test_envs_root = test_envs_root
        else:
            self.test_envs_root = os.getenv(
                                'AICROWD_TESTS_FOLDER',
                                '/tmp/flatland_envs'
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
        response_channel = "{}::{}::response::{}".format(self.namespace,
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
        assert isinstance(_request, dict)
        _request['response_channel'] = self._generate_response_channel()

        _redis = self.get_redis_connection()
        """
            The client always pushes in the left
            and the service always pushes in the right
        """
        if self.verbose:
            print("Request : ", _request)
        # Push request in command_channels
        # Note: The patched msgpack supports numpy arrays
        payload = msgpack.packb(_request, default=m.encode, use_bin_type=True)
        _redis.lpush(self.command_channel, payload)
        # Wait with a blocking pop for the response
        _response = _redis.blpop(_request['response_channel'])[1]
        if self.verbose:
            print("Response : ", _response)
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

    def env_create(self, obs_builder_object):
        """
            Create a local env and remote env on which the 
            local agent can operate.
            The observation builder is only used in the local env
            and the remote env uses a DummyObservationBuilder
        """
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
        print("Received Env : ", test_env_file_path)
        test_env_file_path = os.path.join(
            self.test_envs_root,
            test_env_file_path
        )
        if not os.path.exists(test_env_file_path):
            raise Exception(
                "\nWe cannot seem to find the env file paths at the required location.\n"
                "Did you remember to set the AICROWD_TESTS_FOLDER environment variable "
                "to point to the location of the Tests folder ? \n"
                "We are currently looking at `{}` for the tests".format(self.test_envs_root)
                )
        print("Current env path : ", test_env_file_path)
        self.env = RailEnv(
            width=1,
            height=1,
            rail_generator=rail_from_file(test_env_file_path),
            obs_builder_object=obs_builder_object
        )
        self.env._max_episode_steps = \
            int(1.5 * (self.env.width + self.env.height))

        local_observation = self.env.reset()
        # Use the local observation 
        # as the remote server uses a dummy observation builder
        return local_observation

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
        
        # remote_observation = _payload['observation']
        remote_reward = _payload['reward']
        remote_done = _payload['done']
        remote_info = _payload['info']

        # Replicate the action in the local env
        local_observation, local_reward, local_done, local_info = \
            self.env.step(action)
        
        print(local_reward)
        if not are_dicts_equal(remote_reward, local_reward):
            raise Exception("local and remote `reward` are diverging")
            print(remote_reward, local_reward)
        if not are_dicts_equal(remote_done, local_done):
            raise Exception("local and remote `done` are diverging")
        
        # Return local_observation instead of remote_observation
        # as the remote_observation is build using a dummy observation
        # builder
        # We return the remote rewards and done as they are the 
        # once used by the evaluator
        return [local_observation, remote_reward, remote_done, remote_info]

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
    remote_client = FlatlandRemoteClient()

    def my_controller(obs, _env):
        _action = {}
        for _idx, _ in enumerate(_env.agents):
            _action[_idx] = np.random.randint(0, 5)
        return _action
    
    my_observation_builder = TreeObsForRailEnv(max_depth=3,
                                predictor=ShortestPathPredictorForRailEnv())

    episode = 0
    obs = True
    while obs:        
        obs = remote_client.env_create(
                    obs_builder_object=my_observation_builder
                    )
        if not obs:
            """
            The remote env returns False as the first obs
            when it is done evaluating all the individual episodes
            """
            break
        print("Episode : {}".format(episode))
        episode += 1

        print(remote_client.env.dones['__all__'])

        while True:
            action = my_controller(obs, remote_client.env)
            observation, all_rewards, done, info = remote_client.env_step(action)
            if done['__all__']:
                print("Current Episode : ", episode)
                print("Episode Done")
                print("Reward : ", sum(list(all_rewards.values())))
                break

    print("Evaluation Complete...")       
    print(remote_client.submit())


