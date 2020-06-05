#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import random
import shutil
import time
import traceback

import crowdai_api
import msgpack
import msgpack_numpy as m
import numpy as np
import pandas as pd
import redis
import timeout_decorator

import flatland
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.evaluators import aicrowd_helpers
from flatland.evaluators import messages
from flatland.utils.rendertools import RenderTool

use_signals_in_timeout = True
if os.name == 'nt':
    """
    Windows doesnt support signals, hence
    timeout_decorators usually fall apart.
    Hence forcing them to not using signals 
    whenever using the timeout decorator.
    """
    use_signals_in_timeout = False

m.patch()

########################################################
# CONSTANTS
########################################################
INTIAL_PLANNING_TIMEOUT = int(os.getenv(
                                "FLATLAND_INITIAL_PLANNING_TIMEOUT",
                                5 * 60))  # 5 mins
PER_STEP_TIMEOUT = int(os.getenv(
                                "FLATLAND_PER_STEP_TIMEOUT",
                                5))  # 5 seconds
DEFAULT_COMMAND_TIMEOUT = int(os.getenv(
                                "FLATLAND_DEFAULT_COMMAND_TIMEOUT",
                                1 * 60))  # 1 min
# This applies to the rest of the commands


RANDOM_SEED = int(os.getenv("FLATLAND_EVALUATION_RANDOM_SEED", 1001))
SUPPORTED_CLIENT_VERSIONS = \
    [
        flatland.__version__
    ]


class FlatlandRemoteEvaluationService:
    """
    A remote evaluation service which exposes the following interfaces
    of a RailEnv :
    - env_create
    - env_step
    and an additional `env_submit` to cater to score computation and on-episode-complete post processings.

    This service is designed to be used in conjunction with
    `FlatlandRemoteClient` and both the srevice and client maintain a
    local instance of the RailEnv instance, and in case of any unexpected
    divergences in the state of both the instances, the local RailEnv
    instance of the `FlatlandRemoteEvaluationService` is supposed to act
    as the single source of truth.

    Both the client and remote service communicate with each other
    via Redis as a message broker. The individual messages are packed and
    unpacked with `msgpack` (a patched version of msgpack which also supports
    numpy arrays).
    """

    def __init__(self,
                 test_env_folder="/tmp",
                 flatland_rl_service_id='FLATLAND_RL_SERVICE_ID',
                 remote_host='127.0.0.1',
                 remote_port=6379,
                 remote_db=0,
                 remote_password=None,
                 visualize=False,
                 video_generation_envs=[],
                 report=None,
                 result_output_path=None,
                 verbose=False):

        # Test Env folder Paths
        self.test_env_folder = test_env_folder
        self.video_generation_envs = video_generation_envs
        self.env_file_paths = self.get_env_filepaths()
        random.shuffle(self.env_file_paths)
        print(self.env_file_paths)
        # Shuffle all the env_file_paths for more exciting videos
        # and for more uniform time progression
        self.instantiate_evaluation_metadata()

        # Logging and Reporting related vars
        self.verbose = verbose
        self.report = report

        self.result_output_path = result_output_path

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

        # AIcrowd evaluation specific vars
        self.oracle_events = crowdai_api.events.CrowdAIEvents(with_oracle=True)
        self.evaluation_state = {
            "state": "PENDING",
            "progress": 0.0,
            "simulation_count": 0,
            "total_simulation_count": len(self.env_file_paths),
            "score": {
                "score": 0.0,
                "score_secondary": 0.0
            },
            "meta": {
                "normalized_reward": 0.0
            }
        }
        self.stats = {}
        self.previous_command = {
            "type": None
        }

        # RailEnv specific variables
        self.env = False
        self.env_renderer = False
        self.reward = 0
        self.simulation_count = -1
        self.simulation_env_file_paths = []
        self.simulation_rewards = []
        self.simulation_rewards_normalized = []
        self.simulation_percentage_complete = []
        self.simulation_steps = []
        self.simulation_times = []
        self.env_step_times = []
        self.begin_simulation = False
        self.current_step = 0
        self.visualize = visualize
        self.vizualization_folder_name = "./.visualizations"
        self.record_frame_step = 0

        if self.visualize:
            if os.path.exists(self.vizualization_folder_name):
                print("[WARNING] Deleting already existing visualizations folder at : {}".format(
                    self.vizualization_folder_name
                ))
                shutil.rmtree(self.vizualization_folder_name)
            os.mkdir(self.vizualization_folder_name)

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

    def delete_key_in_running_stats(self, key):
        """
        This deletes a particular key in the running stats
        dictionary, if it exists
        """
        mean_key = "{}_mean".format(key)
        counter_key = "{}_counter".format(key)
        min_key = "{}_min".format(key)
        max_key = "{}_max".format(key)

        try:
            del mean_key
            del counter_key
            del min_key
            del max_key
        except KeyError:
            pass

    def get_env_filepaths(self):
        """
        Gathers a list of all available rail env files to be used
        for evaluation. The folder structure expected at the `test_env_folder`
        is similar to :

            .
            ├── Test_0
            │   ├── Level_1.pkl
            │   ├── .......
            │   ├── .......
            │   └── Level_99.pkl
            └── Test_1
                ├── Level_1.pkl
                ├── .......
                ├── .......
                └── Level_99.pkl
        """
        env_paths = sorted(glob.glob(
            os.path.join(
                self.test_env_folder,
                "*/*.pkl"
            )
        ))
        # Remove the root folder name from the individual
        # lists, so that we only have the path relative
        # to the test root folder
        env_paths = sorted([os.path.relpath(
            x, self.test_env_folder
        ) for x in env_paths])

        return env_paths
    
    def instantiate_evaluation_metadata(self):
        """
            This instantiates a pandas dataframe to record
            information specific to each of the individual env evaluations.

            This loads the template CSV with pre-filled information from the
            provided metadata.csv file, and fills it up with 
            evaluation runtime information.
        """
        self.evaluation_metadata_df = None
        metadata_file_path = os.path.join(
                self.test_env_folder,
                "metadata.csv"
            )
        if os.path.exists(metadata_file_path):
            self.evaluation_metadata_df = pd.read_csv(metadata_file_path)
            self.evaluation_metadata_df["filename"] = \
                self.evaluation_metadata_df["test_id"] + \
                "/" + self.evaluation_metadata_df["env_id"] + ".pkl"
            self.evaluation_metadata_df = self.evaluation_metadata_df.set_index("filename")

            # Add custom columns for evaluation specific metrics
            self.evaluation_metadata_df["reward"] = np.nan
            self.evaluation_metadata_df["normalized_reward"] = np.nan
            self.evaluation_metadata_df["percentage_complete"] = np.nan
            self.evaluation_metadata_df["steps"] = np.nan
            self.evaluation_metadata_df["simulation_time"] = np.nan

            # Add client specific columns
            # TODO: This needs refactoring
            self.evaluation_metadata_df["controller_inference_time_min"] = np.nan
            self.evaluation_metadata_df["controller_inference_time_mean"] = np.nan
            self.evaluation_metadata_df["controller_inference_time_max"] = np.nan
        else:
            raise Exception("metadata.csv not found in tests folder. Please use an updated version of the test set.")

    def update_evaluation_metadata(self):
        """
        This function is called when we move from one simulation to another
        and it simply tries to update the simulation specific information 
        for the **previous** episode in the metadata_df if it exists.
        """
        if self.evaluation_metadata_df is not None and len(self.simulation_env_file_paths) > 0:
            
            last_simulation_env_file_path = self.simulation_env_file_paths[-1]

            _row = self.evaluation_metadata_df.loc[
                last_simulation_env_file_path
            ]

            _row.reward = self.simulation_rewards[-1]
            _row.normalized_reward = self.simulation_rewards_normalized[-1]
            _row.percentage_complete = self.simulation_percentage_complete[-1]
            _row.steps = self.simulation_steps[-1]
            _row.simulation_time = self.simulation_times[-1]

            # TODO: This needs refactoring
            # Add controller_inference_time_metrics
            _row.controller_inference_time_min = self.stats[
                "current_episode_controller_inference_time_min"
            ]
            _row.controller_inference_time_mean = self.stats[
                "current_episode_controller_inference_time_mean"
            ]
            _row.controller_inference_time_max = self.stats[
                "current_episode_controller_inference_time_max"
            ]

            self.evaluation_metadata_df.loc[
                last_simulation_env_file_path
            ] = _row

            # Delete this key from the stats to ensure that it 
            # gets computed again from scratch in the next episode
            self.delete_key_in_running_stats(
                "current_episode_controller_inference_time")

            if self.verbose:
                print(self.evaluation_metadata_df)

    def instantiate_redis_connection_pool(self):
        """
        Instantiates a Redis connection pool which can be used to
        communicate with the message broker
        """
        if self.verbose or self.report:
            print("Attempting to connect to redis server at {}:{}/{}".format(
                self.remote_host,
                self.remote_port,
                self.remote_db))

        self.redis_pool = redis.ConnectionPool(
            host=self.remote_host,
            port=self.remote_port,
            db=self.remote_db,
            password=self.remote_password
        )
        self.redis_conn = redis.Redis(connection_pool=self.redis_pool)

    def get_redis_connection(self):
        """
        Obtains a new redis connection from a previously instantiated
        redis connection pool
        """
        return self.redis_conn

    def _error_template(self, payload):
        """
        Simple helper function to pass a payload as a part of a
        flatland comms error template.
        """
        _response = {}
        _response['type'] = messages.FLATLAND_RL.ERROR
        _response['payload'] = payload
        return _response

    def get_next_command(self):
        """
        A helper function to obtain the next command, which transparently
        also deals with things like unpacking of the command from the
        packed message, and consider the timeouts, etc when trying to
        fetch a new command.
        """
        
        COMMAND_TIMEOUT = DEFAULT_COMMAND_TIMEOUT
        """
        Handle case specific timeouts :
            - INTIAL_PLANNING_TIMEOUT
                The timeout between an env_create call and the first env_step call
            - PER_STEP_TIMEOUT
                The timeout between two consecutive env_step calls
        """
        if self.previous_command['type'] == messages.FLATLAND_RL.ENV_CREATE:
            """
            In case the previous command is an env_create, then leave 
            a but more time for the intial planning
            """
            COMMAND_TIMEOUT = INTIAL_PLANNING_TIMEOUT
        elif self.previous_command['type'] == messages.FLATLAND_RL.ENV_STEP:
            """
            Use the per_step_time for all timesteps between two env_step calls
            # Corner Case : 
                - Are there any reasons why a call between the last env_step call 
                and the subsequent env_create call will take an excessively large 
                amount of time (>5s in this case)
            """
            COMMAND_TIMEOUT = PER_STEP_TIMEOUT
        elif self.previous_command['type'] == messages.FLATLAND_RL.ENV_SUBMIT:
            """
            If the user has already done an env_submit call, then the timeout 
            can be an arbitrarily large number.
            """
            COMMAND_TIMEOUT = 10**6

        @timeout_decorator.timeout(
            COMMAND_TIMEOUT,
            use_signals=use_signals_in_timeout)  # timeout for each command
        def _get_next_command(command_channel, _redis):
            """
            A low level wrapper for obtaining the next command from a
            pre-agreed command channel.
            At the momment, the communication protocol uses lpush for pushing
            in commands, and brpop for reading out commands.
            """
            command = _redis.brpop(command_channel)[1]
            return command

        try:
            _redis = self.get_redis_connection()
            command = _get_next_command(self.command_channel, _redis)
            if self.verbose or self.report:
                print("Command Service: ", command)
        except timeout_decorator.timeout_decorator.TimeoutError:
            raise Exception(
                "Timeout of {}s in step {} of simulation {}".format(
                    COMMAND_TIMEOUT,
                    self.current_step,
                    self.simulation_count
                ))
        command = msgpack.unpackb(
            command,
            object_hook=m.decode,
            encoding="utf8"
        )
        if self.verbose:
            print("Received Request : ", command)

        message_queue_latency = time.time() - command["timestamp"]
        self.update_running_stats("message_queue_latency", message_queue_latency)
        return command

    def send_response(self, _command_response, command, suppress_logs=False):
        _redis = self.get_redis_connection()
        command_response_channel = command['response_channel']

        if self.verbose and not suppress_logs:
            print("Responding with : ", _command_response)

        _redis.rpush(
            command_response_channel,
            msgpack.packb(
                _command_response,
                default=m.encode,
                use_bin_type=True)
        )

    def handle_ping(self, command):
        """
        Handles PING command from the client.
        """
        service_version = flatland.__version__
        if "version" in command["payload"].keys():
            client_version = command["payload"]["version"]
        else:
            # 2.1.4 -> when the version mismatch check was added
            client_version = "2.1.4"

        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.PONG
        _command_response['payload'] = {}
        if client_version not in SUPPORTED_CLIENT_VERSIONS:
            _command_response['type'] = messages.FLATLAND_RL.ERROR
            _command_response['payload']['message'] = \
                "Client-Server Version Mismatch => " + \
                "[ Client Version : {} ] ".format(client_version) + \
                "[ Server Version : {} ] ".format(service_version)
            self.send_response(_command_response, command)
            raise Exception(_command_response['payload']['message'])

        self.send_response(_command_response, command)

    def handle_env_create(self, command):
        """
        Handles a ENV_CREATE command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        self.simulation_count += 1
        if self.simulation_count < len(self.env_file_paths):
            """
            There are still test envs left that are yet to be evaluated 
            """
            test_env_file_path = self.env_file_paths[self.simulation_count]
            print("Evaluating : {}".format(test_env_file_path))
            test_env_file_path = os.path.join(
                self.test_env_folder,
                test_env_file_path
            )
            del self.env
            self.env = RailEnv(width=1, height=1, rail_generator=rail_from_file(test_env_file_path),
                               schedule_generator=schedule_from_file(test_env_file_path),
                               malfunction_generator_and_process_data=malfunction_from_file(test_env_file_path),
                               obs_builder_object=DummyObservationBuilder())

            if self.begin_simulation:
                # If begin simulation has already been initialized
                # atleast once
                # This adds the simulation time for the previous episode
                self.simulation_times.append(time.time() - self.begin_simulation)
            self.begin_simulation = time.time()

            # Update evaluation metadata for the previous episode
            self.update_evaluation_metadata()

            # Start adding placeholders for the new episode
            self.simulation_env_file_paths.append(
                os.path.relpath(
                    test_env_file_path,
                    self.test_env_folder
                ))  # relative path

            self.simulation_rewards.append(0)
            self.simulation_rewards_normalized.append(0)
            self.simulation_percentage_complete.append(0)
            self.simulation_steps.append(0)

            self.current_step = 0

            _observation, _info = self.env.reset(
                regenerate_rail=True,
                regenerate_schedule=True,
                activate_agents=False,
                random_seed=RANDOM_SEED
            )

            if self.visualize:
                if self.env_renderer:
                    del self.env_renderer
                self.env_renderer = RenderTool(self.env, gl="PILSVG", )

            _command_response = {}
            _command_response['type'] = messages.FLATLAND_RL.ENV_CREATE_RESPONSE
            _command_response['payload'] = {}
            _command_response['payload']['observation'] = _observation
            _command_response['payload']['env_file_path'] = self.env_file_paths[self.simulation_count]
            _command_response['payload']['info'] = _info
            _command_response['payload']['random_seed'] = RANDOM_SEED
        else:
            """
            All test env evaluations are complete
            """
            _command_response = {}
            _command_response['type'] = messages.FLATLAND_RL.ENV_CREATE_RESPONSE
            _command_response['payload'] = {}
            _command_response['payload']['observation'] = False
            _command_response['payload']['env_file_path'] = False
            _command_response['payload']['info'] = False
            _command_response['payload']['random_seed'] = False

        self.send_response(_command_response, command)
        #####################################################################
        # Update evaluation state
        #####################################################################
        progress = np.clip(
            self.simulation_count * 1.0 / len(self.env_file_paths),
            0, 1)

        mean_reward, mean_normalized_reward, mean_percentage_complete = self.compute_mean_scores()

        self.evaluation_state["state"] = "IN_PROGRESS"
        self.evaluation_state["progress"] = progress
        self.evaluation_state["simulation_count"] = self.simulation_count
        self.evaluation_state["score"]["score"] = mean_percentage_complete
        self.evaluation_state["score"]["score_secondary"] = mean_reward
        self.evaluation_state["meta"]["normalized_reward"] = mean_normalized_reward
        self.handle_aicrowd_info_event(self.evaluation_state)

    def handle_env_step(self, command):
        """
        Handles a ENV_STEP command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        _payload = command['payload']

        if not self.env:
            raise Exception(
                "env_client.step called before env_client.env_create() call")
        if self.env.dones['__all__']:
            raise Exception(
                "Client attempted to perform an action on an Env which \
                has done['__all__']==True")

        action = _payload['action']
        inference_time = _payload['inference_time']
        # We record this metric in two keys:
        #   - One for the current episode
        #   - One global
        self.update_running_stats("current_episode_controller_inference_time", inference_time)
        self.update_running_stats("controller_inference_time", inference_time)

        time_start = time.time()
        _observation, all_rewards, done, info = self.env.step(action)
        time_diff = time.time() - time_start
        self.update_running_stats("internal_env_step_time", time_diff)

        cumulative_reward = sum(all_rewards.values())
        self.simulation_rewards[-1] += cumulative_reward
        self.simulation_steps[-1] += 1
        """
        The normalized rewards normalize the reward for an 
        episode by dividing the whole reward by max-time-steps 
        allowed in that episode, and the number of agents present in 
        that episode
        """
        self.simulation_rewards_normalized[-1] += \
            cumulative_reward / (
                self.env._max_episode_steps *
                self.env.get_num_agents()
            )

        if done["__all__"]:
            # Compute percentage complete
            complete = 0
            for i_agent in range(self.env.get_num_agents()):
                agent = self.env.agents[i_agent]
                if agent.status in [RailAgentStatus.DONE_REMOVED]:
                    complete += 1
            percentage_complete = complete * 1.0 / self.env.get_num_agents()
            self.simulation_percentage_complete[-1] = percentage_complete

        # Record Frame
        if self.visualize:
            self.env_renderer.render_env(
                show=False,
                show_observations=False,
                show_predictions=False
            )
            """
            Only save the frames for environments which are separately provided 
            in video_generation_indices param
            """
            current_env_path = self.env_file_paths[self.simulation_count]
            if current_env_path in self.video_generation_envs:
                self.env_renderer.gl.save_image(
                    os.path.join(
                        self.vizualization_folder_name,
                        "flatland_frame_{:04d}.png".format(self.record_frame_step)
                    ))
                self.record_frame_step += 1

    def handle_env_submit(self, command):
        """
        Handles a ENV_SUBMIT command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        _payload = command['payload']

        ######################################################################
        # Print Local Stats
        ######################################################################
        print("=" * 100)
        print("=" * 100)
        print("## Server Performance Stats")
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

        # Register simulation time of the last episode
        self.simulation_times.append(time.time() - self.begin_simulation)
        # Compute the evaluation metadata for the last episode
        self.update_evaluation_metadata()

        if len(self.simulation_rewards) != len(self.env_file_paths):
            raise Exception(
                """env.submit called before the agent had the chance 
                to operate on all the test environments.
                """
            )

        mean_reward, mean_normalized_reward, mean_percentage_complete = self.compute_mean_scores()

        if self.visualize and len(os.listdir(self.vizualization_folder_name)) > 0:
            # Generate the video
            #
            # Note, if you had depdency issues due to ffmpeg, you can
            # install it by :
            #
            # conda install -c conda-forge x264 ffmpeg

            print("Generating Video from thumbnails...")
            video_output_path, video_thumb_output_path = \
                aicrowd_helpers.generate_movie_from_frames(
                    self.vizualization_folder_name
                )
            print("Videos : ", video_output_path, video_thumb_output_path)
            # Upload to S3 if configuration is available
            if aicrowd_helpers.is_grading() and aicrowd_helpers.is_aws_configured() and self.visualize:
                video_s3_key = aicrowd_helpers.upload_to_s3(
                    video_output_path
                )
                video_thumb_s3_key = aicrowd_helpers.upload_to_s3(
                    video_thumb_output_path
                )
                static_thumbnail_s3_key = aicrowd_helpers.upload_random_frame_to_s3(
                    self.vizualization_folder_name
                )
                self.evaluation_state["score"]["media_content_type"] = "video/mp4"
                self.evaluation_state["score"]["media_large"] = video_s3_key
                self.evaluation_state["score"]["media_thumbnail"] = video_thumb_s3_key

                self.evaluation_state["meta"]["static_media_frame"] = static_thumbnail_s3_key
            else:
                print("[WARNING] Ignoring uploading of video to S3")

        #####################################################################
        # Write Results to a file (if applicable)
        #####################################################################
        if self.result_output_path:
            self.evaluation_metadata_df.to_csv(self.result_output_path)
            print("Wrote output results to : {}".format(self.result_output_path))
            
            # Upload the metadata file to S3 
            if aicrowd_helpers.is_grading() and aicrowd_helpers.is_aws_configured():
                metadata_s3_key = aicrowd_helpers.upload_to_s3(
                    self.result_output_path
                )
                self.evaluation_state["meta"]["private_metadata_s3_key"] = metadata_s3_key

        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE
        _payload = {}
        _payload['mean_reward'] = mean_reward
        _payload['mean_normalized_reward'] = mean_normalized_reward
        _payload['mean_percentage_complete'] = mean_percentage_complete
        _command_response['payload'] = _payload
        self.send_response(_command_response, command)

        #####################################################################
        # Update evaluation state
        #####################################################################
        self.evaluation_state["state"] = "FINISHED"
        self.evaluation_state["progress"] = 1.0
        self.evaluation_state["simulation_count"] = self.simulation_count
        self.evaluation_state["score"]["score"] = mean_normalized_reward
        self.evaluation_state["score"]["score_secondary"] = mean_percentage_complete
        self.evaluation_state["meta"]["normalized_reward"] = mean_normalized_reward
        self.evaluation_state["meta"]["reward"] = mean_reward
        self.evaluation_state["meta"]["percentage_complete"] = mean_percentage_complete
        self.handle_aicrowd_success_event(self.evaluation_state)
        print("#" * 100)
        print("EVALUATION COMPLETE !!")
        print("#" * 100)
        print("# Mean Reward : {}".format(mean_reward))
        print("# Mean Normalized Reward : {}".format(mean_normalized_reward))
        print("# Mean Percentage Complete : {}".format(mean_percentage_complete))
        print("#" * 100)
        print("#" * 100)

    def compute_mean_scores(self):
        #################################################################################
        #################################################################################
        # Compute the mean rewards, mean normalized_reward and mean_percentage_complete
        # we group all the results by the test_ids
        # so we first compute the mean in each of the test_id groups, 
        # and then we compute the mean across each of the test_id groups
        #
        #
        #################################################################################
        #################################################################################
        source_df = self.evaluation_metadata_df.dropna()
        grouped_df = source_df.groupby(['test_id']).mean()

        mean_reward = grouped_df["reward"].mean()
        mean_normalized_reward = grouped_df["normalized_reward"].mean()
        mean_percentage_complete = grouped_df["percentage_complete"].mean()
        # Round off the reward values
        mean_reward = round(mean_reward, 2)
        mean_normalized_reward = round(mean_normalized_reward, 5)
        mean_percentage_complete = round(mean_percentage_complete, 3)

        return mean_reward, mean_normalized_reward, mean_percentage_complete

    def report_error(self, error_message, command_response_channel):
        """
        A helper function used to report error back to the client
        """
        _redis = self.get_redis_connection()
        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.ERROR
        _command_response['payload'] = error_message
        _redis.rpush(
            command_response_channel,
            msgpack.packb(
                _command_response,
                default=m.encode,
                use_bin_type=True)
        )
        self.evaluation_state["state"] = "ERROR"
        self.evaluation_state["error"] = error_message
        self.handle_aicrowd_error_event(self.evaluation_state)

    def handle_aicrowd_info_event(self, payload):
        self.oracle_events.register_event(
            event_type=self.oracle_events.CROWDAI_EVENT_INFO,
            payload=payload
        )

    def handle_aicrowd_success_event(self, payload):
        self.oracle_events.register_event(
            event_type=self.oracle_events.CROWDAI_EVENT_SUCCESS,
            payload=payload
        )

    def handle_aicrowd_error_event(self, payload):
        self.oracle_events.register_event(
            event_type=self.oracle_events.CROWDAI_EVENT_ERROR,
            payload=payload
        )

    def run(self):
        """
        Main runner function which waits for commands from the client
        and acts accordingly.
        """
        print("Listening at : ", self.command_channel)
        MESSAGE_QUEUE_LATENCY = []
        while True:
            command = self.get_next_command()
            if "timestamp" in command.keys():
                latency = time.time() - command["timestamp"]
                MESSAGE_QUEUE_LATENCY.append(latency)

            if self.verbose:
                print("Self.Reward : ", self.reward)
                print("Current Simulation : ", self.simulation_count)
                if self.env_file_paths and \
                        self.simulation_count < len(self.env_file_paths):
                    print("Current Env Path : ",
                        self.env_file_paths[self.simulation_count])

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
                elif command['type'] == messages.FLATLAND_RL.ENV_STEP:
                    """
                        ENV_STEP

                        Request : Action dict
                        Respond with updated [observation,reward,done,info] after step
                    """
                    self.handle_env_step(command)
                elif command['type'] == messages.FLATLAND_RL.ENV_SUBMIT:
                    """
                        ENV_SUBMIT

                        Submit the final cumulative reward
                    """

                    print("Overall Message Queue Latency : ", np.array(MESSAGE_QUEUE_LATENCY).mean())
                    self.handle_env_submit(command)
                else:
                    _error = self._error_template(
                        "UNKNOWN_REQUEST:{}".format(
                            str(command)))
                    if self.verbose:
                        print("Responding with : ", _error)
                    self.report_error(
                        _error,
                        command['response_channel'])
                    return _error
                ###########################################
                # We keep a record of the previous command
                # to be able to have different behaviors 
                # between different "command transitions"
                # 
                # An example use case, is when we want to 
                # have a different timeout for the 
                # first step in every environment 
                # to account for some initial planning time
                self.previous_command = command
            except Exception as e:
                print("Error : ", str(e))
                print(traceback.format_exc())
                self.report_error(
                    self._error_template(str(e)),
                    command['response_channel'])
                return self._error_template(str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Submit the result to AIcrowd')
    parser.add_argument('--service_id',
                        dest='service_id',
                        default='FLATLAND_RL_SERVICE_ID',
                        required=False)
    parser.add_argument('--test_folder',
                        dest='test_folder',
                        default="../../../submission-scoring/Envs-Small",
                        help="Folder containing the files for the test envs",
                        required=False)
    args = parser.parse_args()

    test_folder = args.test_folder

    grader = FlatlandRemoteEvaluationService(
        test_env_folder=test_folder,
        flatland_rl_service_id=args.service_id,
        verbose=False,
        visualize=True,
        video_generation_envs=["Test_0/Level_100.pkl"],
        result_output_path="/tmp/output.csv"
    )
    result = grader.run()
    if result['type'] == messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE:
        cumulative_results = result['payload']
    elif result['type'] == messages.FLATLAND_RL.ERROR:
        error = result['payload']
        raise Exception("Evaluation Failed : {}".format(str(error)))
    else:
        # Evaluation failed
        print("Evaluation Failed : ", result['payload'])
