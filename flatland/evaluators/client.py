import hashlib
import json
import logging
import os
import pickle
import random
import time

import msgpack
import msgpack_numpy as m
import numpy as np
import redis

import flatland
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.persistence import RailEnvPersister
from flatland.evaluators import messages

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
m.patch()

# CONSTANTS
FLATLAND_RL_SERVICE_ID = os.getenv(
    'AICROWD_SUBMISSION_ID',
    'T12345')


class TimeoutException(StopAsyncIteration):
    """ Custom exception for evaluation timeouts. """
    pass


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
                 test_env_folder=None,
                 flatland_rl_service_id=FLATLAND_RL_SERVICE_ID,
                 remote_host=os.getenv("redis_ip", '127.0.0.1'),
                 remote_port=6379,
                 remote_db=0,
                 remote_password=None,
                 verbose=False,
                 use_pickle=False):
        self.use_pickle = use_pickle
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_db = remote_db
        self.remote_password = remote_password
        self.redis_pool = redis.ConnectionPool(
            host=remote_host,
            port=remote_port,
            db=remote_db,
            password=remote_password)
        self.redis_conn = redis.Redis(connection_pool=self.redis_pool)

        self.namespace = "flatland-rl"
        self.service_id = flatland_rl_service_id
        self.command_channel = "{}::{}::commands".format(
            self.namespace,
            self.service_id
        )
        # for timeout messages sent out-of-band
        self.error_channel = "{}::{}::errors".format(
            self.namespace,
            self.service_id
        )

        if test_env_folder:
            self.test_envs_root = test_env_folder
        else:
            self.test_envs_root = os.getenv(
                'AICROWD_TESTS_FOLDER',
                '/tmp/flatland_envs'
            )
        self.current_env_path = None

        self.verbose = verbose

        self.env = None
        self.ping_pong()

        self.env_step_times = []
        self.stats = {}

    def update_running_stats(self, key, scalar):
        """
        Computes the running mean for certain params
        """
        mean_key = "{}_mean".format(key)
        counter_key = "{}_counter".format(key)
        min_key = "{}_min".format(key)
        max_key = "{}_max".format(key)

        try:
            # Update Mean
            self.stats[mean_key] = \
                ((self.stats[mean_key] * self.stats[counter_key]) + scalar) / (self.stats[counter_key] + 1)
            # Update min
            if scalar < self.stats[min_key]:
                self.stats[min_key] = scalar
            # Update max
            if scalar > self.stats[max_key]:
                self.stats[max_key] = scalar

            self.stats[counter_key] += 1
        except KeyError:
            self.stats[mean_key] = scalar
            self.stats[min_key] = scalar
            self.stats[max_key] = scalar
            self.stats[counter_key] = 1

    def get_redis_connection(self):
        return self.redis_conn

    def _generate_response_channel(self):
        random_hash = hashlib.md5(
            "{}".format(
                random.randint(0, 10 ** 10)
            ).encode('utf-8')).hexdigest()
        response_channel = "{}::{}::response::{}".format(self.namespace,
                                                         self.service_id,
                                                         random_hash)
        return response_channel

    def _remote_request(self, _request, blocking=True):
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
        _request['timestamp'] = time.time()

        _redis = self.get_redis_connection()
        """
            The client always pushes in the left
            and the service always pushes in the right
        """
        if self.verbose:
            print("Request : ", _request)

        # check for errors (essentially just timeouts, for now.)
        error_bytes = _redis.rpop(self.error_channel)
        if error_bytes is not None:
            if self.use_pickle:
                error_dict = pickle.loads(error_bytes)
            else:
                error_dict = msgpack.unpackb(
                    error_bytes,
                    object_hook=m.decode,
                    strict_map_key=False,  # new for msgpack 1.0?
                )
            print("Error received: ", error_dict)
            raise TimeoutException(error_dict["type"])

        # Push request in command_channels
        # Note: The patched msgpack supports numpy arrays
        if self.use_pickle:
            payload = pickle.dumps(_request)
        else:
            payload = msgpack.packb(_request, default=m.encode, use_bin_type=True)
        _redis.lpush(self.command_channel, payload)

        if blocking:
            # Wait with a blocking pop for the response
            _response = _redis.blpop(_request['response_channel'])[1]
            if self.verbose:
                print("Response : ", _response)
            if self.use_pickle:
                _response = pickle.loads(_response)
            else:
                _response = msgpack.unpackb(
                    _response,
                    object_hook=m.decode,
                    strict_map_key=False,  # new for msgpack 1.0?
                )
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
        _request['payload'] = {
            "version": flatland.__version__
        }
        _response = self._remote_request(_request)
        if _response['type'] != messages.FLATLAND_RL.PONG:
            raise Exception(
                "Unable to perform handshake with the evaluation service. \
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
        time_start = time.time()
        _request = {}
        _request['type'] = messages.FLATLAND_RL.ENV_CREATE
        _request['payload'] = {}
        _response = self._remote_request(_request)
        observation = _response['payload']['observation']
        info = _response['payload']['info']
        random_seed = _response['payload']['random_seed']
        test_env_file_path = _response['payload']['env_file_path']
        time_diff = time.time() - time_start
        self.update_running_stats("env_creation_wait_time", time_diff)

        if not observation:
            # If the observation is False,
            # then the evaluations are complete
            # hence return false
            return observation, info

        if self.verbose:
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

        if self.verbose:
            print("Current env path : ", test_env_file_path)
        self.current_env_path = test_env_file_path
        self.env, _ = RailEnvPersister.load_new(test_env_file_path)

        time_start = time.time()
        # Use the local observation
        # as the remote server uses a dummy observation builder
        local_observation, info = self.env.reset(
            regenerate_rail=True,
            regenerate_schedule=True,
            random_seed=random_seed
        )
        time_diff = time.time() - time_start
        self.update_running_stats("internal_env_reset_time", time_diff)

        # We use the last_env_step_time as an approximate measure of the inference time
        self.last_env_step_time = time.time()
        return local_observation, info

    def env_step(self, action, render=False):
        """
            Respond with [observation, reward, done, info]
        """
        # We use the last_env_step_time as an approximate measure of the inference time
        approximate_inference_time = time.time() - self.last_env_step_time
        self.update_running_stats("inference_time(approx)", approximate_inference_time)

        _request = {}
        _request['type'] = messages.FLATLAND_RL.ENV_STEP
        _request['payload'] = {}
        _request['payload']['action'] = action
        _request['payload']['inference_time'] = approximate_inference_time

        # Relay the action in a non-blocking way to the server
        # so that it can start doing an env.step on it in ~ parallel
        # Note - this can throw a Timeout
        self._remote_request(_request, blocking=False)

        # Apply the action in the local env
        time_start = time.time()
        local_observation, local_reward, local_done, local_info = \
            self.env.step(action)
        time_diff = time.time() - time_start
        # Compute a running mean of env step times
        self.update_running_stats("internal_env_step_time", time_diff)

        # We use the last_env_step_time as an approximate measure of the inference time
        self.last_env_step_time = time.time()

        return [local_observation, local_reward, local_done, local_info]

    def submit(self):
        _request = {}
        _request['type'] = messages.FLATLAND_RL.ENV_SUBMIT
        _request['payload'] = {}
        _response = self._remote_request(_request)

        ######################################################################
        # Print Local Stats
        ######################################################################
        print("=" * 100)
        print("=" * 100)
        print("## Client Performance Stats")
        print("=" * 100)
        for _key in self.stats:
            if _key.endswith("_mean"):
                metric_name = _key.replace("_mean", "")
                mean_key = "{}_mean".format(metric_name)
                min_key = "{}_min".format(metric_name)
                max_key = "{}_max".format(metric_name)
                print("\t - {}\t => min: {} || mean: {} || max: {}".format(
                    metric_name,
                    self.stats[min_key],
                    self.stats[mean_key],
                    self.stats[max_key]))
        print("=" * 100)
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


    my_observation_builder = DummyObservationBuilder()

    episode = 0
    obs = True
    while obs:
        obs, info = remote_client.env_create(
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
            time_start = time.time()

            try:
                observation, all_rewards, done, info = remote_client.env_step(action)
                time_diff = time.time() - time_start
                print("Step Time : ", time_diff)
                if done['__all__']:
                    print("Current Episode : ", episode)
                    print("Episode Done")
                    print("Reward : ", sum(list(all_rewards.values())))
                    break
            except TimeoutException as err:
                print("Timeout: ", err)
                break

    print("Evaluation Complete...")
    print(remote_client.submit())
