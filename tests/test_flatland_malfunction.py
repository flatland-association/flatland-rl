import random
from typing import Dict, List

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail2
from test_utils import Replay, ReplayConfig, run_replay_config, set_penalties_for_replay


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent_virtual_position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def test_malfunction_process():
    # Set fixed malfunction duration for this test
    stochastic_data = MalfunctionParameters(malfunction_rate=1,  # Rate of malfunction occurence
                                            min_duration=3,  # Minimal duration of malfunction
                                            max_duration=3  # Max duration of malfunction
                                            )

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    obs, info = env.reset(False, False, True, random_seed=10)

    agent_halts = 0
    total_down_time = 0
    agent_old_position = env.agents[0].position

    # Move target to unreachable position in order to not interfere with test
    env.agents[0].target = (0, 0)
    for step in range(100):
        actions = {}

        for i in range(len(obs)):
            actions[i] = np.argmax(obs[i]) + 1

        obs, all_rewards, done, _ = env.step(actions)

        if env.agents[0].malfunction_data['malfunction'] > 0:
            agent_malfunctioning = True
        else:
            agent_malfunctioning = False

        if agent_malfunctioning:
            # Check that agent is not moving while malfunctioning
            assert agent_old_position == env.agents[0].position

        agent_old_position = env.agents[0].position
        total_down_time += env.agents[0].malfunction_data['malfunction']

    # Check that the appropriate number of malfunctions is achieved
    assert env.agents[0].malfunction_data['nr_malfunctions'] == 23, "Actual {}".format(
        env.agents[0].malfunction_data['nr_malfunctions'])

    # Check that malfunctioning data was standing around
    assert total_down_time > 0


def test_malfunction_process_statistically():
    """Tests hat malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    stochastic_data = MalfunctionParameters(malfunction_rate=5,  # Rate of malfunction occurence
                                            min_duration=5,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )

    env.reset(True, True, False, random_seed=10)

    env.agents[0].target = (0, 0)
    # Next line only for test generation
    # agent_malfunction_list = [[] for i in range(10)]
    agent_malfunction_list = [[0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 0, 0, 5, 4],
                              [0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 0, 5, 4, 3, 2],
                              [0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1],
                              [0, 0, 5, 4, 3, 2, 1, 0, 0, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                              [5, 4, 3, 2, 1, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 5],
                              [5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2],
                              [5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 5, 4]]

    for step in range(20):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent_idx in range(env.get_num_agents()):
            # We randomly select an action
            action_dict[agent_idx] = RailEnvActions(np.random.randint(4))
            # For generating tests only:
            # agent_malfunction_list[agent_idx].append(env.agents[agent_idx].malfunction_data['malfunction'])
            assert env.agents[agent_idx].malfunction_data['malfunction'] == agent_malfunction_list[agent_idx][step]
        env.step(action_dict)
    # print(agent_malfunction_list)


def test_malfunction_before_entry():
    """Tests that malfunctions are working properly for agents before entering the environment!"""
    # Set fixed malfunction duration for this test
    stochastic_data = MalfunctionParameters(malfunction_rate=2,  # Rate of malfunction occurence
                                            min_duration=10,  # Minimal duration of malfunction
                                            max_duration=10  # Max duration of malfunction
                                            )

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    env.reset(False, False, False, random_seed=10)
    env.agents[0].target = (0, 0)

    # Test initial malfunction values for all agents
    # we want some agents to be malfuncitoning already and some to be working
    # we want different next_malfunction values for the agents
    assert env.agents[0].malfunction_data['malfunction'] == 0
    assert env.agents[1].malfunction_data['malfunction'] == 10
    assert env.agents[2].malfunction_data['malfunction'] == 0
    assert env.agents[3].malfunction_data['malfunction'] == 10
    assert env.agents[4].malfunction_data['malfunction'] == 10
    assert env.agents[5].malfunction_data['malfunction'] == 10
    assert env.agents[6].malfunction_data['malfunction'] == 10
    assert env.agents[7].malfunction_data['malfunction'] == 10
    assert env.agents[8].malfunction_data['malfunction'] == 10
    assert env.agents[9].malfunction_data['malfunction'] == 10

    # for a in range(10):
    # print("assert env.agents[{}].malfunction_data['malfunction'] == {}".format(a,env.agents[a].malfunction_data['malfunction']))


def test_malfunction_values_and_behavior():
    """
    Test the malfunction counts down as desired
    Returns
    -------

    """
    # Set fixed malfunction duration for this test

    rail, rail_map = make_simple_rail2()
    action_dict: Dict[int, RailEnvActions] = {}
    stochastic_data = MalfunctionParameters(malfunction_rate=0.001,  # Rate of malfunction occurence
                                            min_duration=10,  # Minimal duration of malfunction
                                            max_duration=10  # Max duration of malfunction
                                            )
    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )

    env.reset(False, False, activate_agents=True, random_seed=10)

    # Assertions
    assert_list = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 9, 8, 7, 6, 5]
    print("[")
    for time_step in range(15):
        # Move in the env
        env.step(action_dict)
        # Check that next_step decreases as expected
        assert env.agents[0].malfunction_data['malfunction'] == assert_list[time_step]


def test_initial_malfunction():
    stochastic_data = MalfunctionParameters(malfunction_rate=1000,  # Rate of malfunction occurence
                                            min_duration=2,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=10),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static
    env.reset(False, False, True, random_seed=10)
    print(env.agents[0].malfunction_data)
    env.agents[0].target = (0, 5)
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=2,
                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action MOVE_FORWARD, agent should restart and move to the next cell
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=1,
                reward=env.step_penalty

            ),  # malfunctioning ends: starting and running at speed 1.0
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0  # running at speed 1.0
            ),
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty  # running at speed 1.0
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target,
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [replay_config])


def test_initial_malfunction_stop_moving():
    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=1,
                  obs_builder_object=SingleAgentNavigationObs())
    env.reset()

    print(env.agents[0].initial_position, env.agents[0].direction, env.agents[0].position, env.agents[0].status)

    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=None,
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty,  # full step penalty when stopped
                status=RailAgentStatus.READY_TO_DEPART
            ),
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=2,
                reward=env.step_penalty,  # full step penalty when stopped
                status=RailAgentStatus.ACTIVE
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action STOP_MOVING, agent should restart without moving
            #
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.STOP_MOVING,
                malfunction=1,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            # we have stopped and do nothing --> should stand still
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=0,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            # we start to move forward --> should go to next cell now
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target,
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )

    run_replay_config(env, [replay_config], activate_agents=False)


def test_initial_malfunction_do_nothing():
    stochastic_data = MalfunctionParameters(malfunction_rate=70,  # Rate of malfunction occurence
                                            min_duration=2,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  # Malfunction data generator
                  )
    env.reset()
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=None,
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty,  # full step penalty while malfunctioning
                status=RailAgentStatus.READY_TO_DEPART
            ),
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=2,
                reward=env.step_penalty,  # full step penalty while malfunctioning
                status=RailAgentStatus.ACTIVE
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action DO_NOTHING, agent should restart without moving
            #
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=1,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            # we haven't started moving yet --> stay here
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=0,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),

            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0,  # start penalty + step penalty for speed 1.0
                status=RailAgentStatus.ACTIVE
            ),  # we start to move forward --> should go to next cell now
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0,  # step penalty for speed 1.0
                status=RailAgentStatus.ACTIVE
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target,
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [replay_config], activate_agents=False)


def tests_random_interference_from_outside():
    """Tests that malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    rail, rail_map = make_simple_rail2()
    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=2), number_of_agents=1, random_seed=1)
    env.reset()
    env.agents[0].speed_data['speed'] = 0.33
    env.reset(False, False, False, random_seed=10)
    env_data = []

    for step in range(200):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # We randomly select an action
            action_dict[agent.handle] = RailEnvActions(2)

        _, reward, _, _ = env.step(action_dict)
        # Append the rewards of the first trial
        env_data.append((reward[0], env.agents[0].position))
        assert reward[0] == env_data[step][0]
        assert env.agents[0].position == env_data[step][1]
    # Run the same test as above but with an external random generator running
    # Check that the reward stays the same

    rail, rail_map = make_simple_rail2()
    random.seed(47)
    np.random.seed(1234)
    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=2), number_of_agents=1, random_seed=1)
    env.reset()
    env.agents[0].speed_data['speed'] = 0.33
    env.reset(False, False, False, random_seed=10)

    dummy_list = [1, 2, 6, 7, 8, 9, 4, 5, 4]
    for step in range(200):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # We randomly select an action
            action_dict[agent.handle] = RailEnvActions(2)

            # Do dummy random number generations
            random.shuffle(dummy_list)
            np.random.rand()

        _, reward, _, _ = env.step(action_dict)
        assert reward[0] == env_data[step][0]
        assert env.agents[0].position == env_data[step][1]


def test_last_malfunction_step():
    """
    Test to check that agent moves when it is not malfunctioning

    """

    # Set fixed malfunction duration for this test

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=2), number_of_agents=1, random_seed=1)
    env.reset()
    env.agents[0].speed_data['speed'] = 1. / 3.
    env.agents[0].target = (0, 0)

    env.reset(False, False, True)
    # Force malfunction to be off at beginning and next malfunction to happen in 2 steps
    env.agents[0].malfunction_data['next_malfunction'] = 2
    env.agents[0].malfunction_data['malfunction'] = 0
    env_data = []
    for step in range(20):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # Go forward all the time
            action_dict[agent.handle] = RailEnvActions(2)

        if env.agents[0].malfunction_data['malfunction'] < 1:
            agent_can_move = True
        # Store the position before and after the step
        pre_position = env.agents[0].speed_data['position_fraction']
        _, reward, _, _ = env.step(action_dict)
        # Check if the agent is still allowed to move in this step

        if env.agents[0].malfunction_data['malfunction'] > 0:
            agent_can_move = False
        post_position = env.agents[0].speed_data['position_fraction']
        # Assert that the agent moved while it was still allowed
        if agent_can_move:
            assert pre_position != post_position
        else:
            assert post_position == pre_position
