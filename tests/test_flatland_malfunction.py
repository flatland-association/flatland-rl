import random
from fractions import Fraction
from typing import Dict, List

import numpy as np
import pytest

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import _sanitize_entry_point, virtual_entry_point
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.states import TrainState
from flatland.utils.simple_rail import make_simple_rail2
from tests.test_utils import Replay, ReplayConfig, run_replay_config, set_penalties_for_replay

pytestmark = pytest.mark.cython_ext


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        entry_point = virtual_entry_point(agent)
        if entry_point is None:
            return None
        agent_virtual_position, agent_virtual_direction = entry_point

        possible_transitions = self.env.rail.get_transitions((agent_virtual_position, agent_virtual_direction))
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent_virtual_direction + i) % 4 for i in range(-1, 2)]:
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
    stochastic_data = MalfunctionParameters(malfunction_rate=1,  # Rate of malfunction occurrence
                                            min_duration=3,  # Minimal duration of malfunction
                                            max_duration=3  # Max duration of malfunction
                                            )

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    obs, info = env.reset(False, False, random_seed=10)
    for a_idx in range(len(env.agents)):
        env_agent = env.agents[a_idx]
        env_agent.current_entry_point = env_agent.initial_entry_point
        env_agent.state = TrainState.MOVING
        # design: distance is None when off map -- the agent is placed directly on the map here,
        # bypassing the state machine's own departure step, so bootstrap distance to 0 explicitly.
        env_agent.speed_counter.step(speed=env_agent.speed_counter.speed, crossing_completed=False)
        # design: actions applied at cell entry -- keep the current_entry_point/next_entry_point
        # invariant (both set and different while on-map) even for this direct test-harness setup.
        transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, env_agent.current_entry_point)
        assert transition is not None
        env_agent.next_entry_point = _sanitize_entry_point(transition)

    agent_halts = 0
    total_down_time = 0
    agent_old_position = env.agents[0].current_entry_point[0]

    # Move target to unreachable position in order to not interfere with test
    env.agents[0].targets = {((0, 0), d) for d in Grid4TransitionsEnum}

    # Add in max episode steps because schedule generator sets it to 0 for dummy data
    env._max_episode_steps = 200
    for step in range(100):
        actions = {}

        for i in range(len(obs)):
            actions[i] = np.argmax(obs[i]) + 1

        obs, all_rewards, done, _ = env.step(actions)
        if done["__all__"]:
            break

        if env.agents[0].malfunction_handler.malfunction_down_counter > 0:
            agent_malfunctioning = True
        else:
            agent_malfunctioning = False

        if agent_malfunctioning:
            # Check that agent is not moving while malfunctioning
            assert agent_old_position == env.agents[0].current_entry_point[0]

        agent_old_position = env.agents[0].current_entry_point[0]
        total_down_time += env.agents[0].malfunction_handler.malfunction_down_counter
    # Check that the appropriate number of malfunctions is achieved
    # Dipam: The number of malfunctions varies by seed
    assert env.agents[0].malfunction_handler.num_malfunctions == 28, "Actual {}".format(
        env.agents[0].malfunction_handler.num_malfunctions)

    # Check that malfunctioning data was standing around
    assert total_down_time > 0


def test_malfunction_process_statistically():
    """Tests that malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    stochastic_data = MalfunctionParameters(malfunction_rate=1 / 5,  # Rate of malfunction occurence
                                            min_duration=5,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )

    env.reset(True, True, random_seed=10)
    env._max_episode_steps = 1000

    env.agents[0].targets = {((0, 0), d) for d in Grid4TransitionsEnum}
    # Next line only for test generation
    agent_malfunction_list = [[] for i in range(2)]
    agent_malfunction_list = [[0, 0, 0, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 1],
                              [0, 0, 4, 3, 2, 1, 0, 0, 0, 0, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1]]

    for step in range(20):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent_idx in range(env.get_num_agents()):
            # We randomly select an action
            action_dict[agent_idx] = RailEnvActions.from_value(np.random.randint(4))
            # For generating tests only:
            # agent_malfunction_list[agent_idx].append(
            # env.agents[agent_idx].malfunction_handler.malfunction_down_counter)
            assert env.agents[agent_idx].malfunction_handler.malfunction_down_counter == \
                   agent_malfunction_list[agent_idx][step]
        env.step(action_dict)


def test_malfunction_before_entry():
    """Tests that malfunctions are working properly for agents before entering the environment!"""
    # Set fixed malfunction duration for this test
    stochastic_data = MalfunctionParameters(malfunction_rate=1 / 2,  # Rate of malfunction occurrence
                                            min_duration=10,  # Minimal duration of malfunction
                                            max_duration=10  # Max duration of malfunction
                                            )

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    env.reset(False, False, random_seed=10)
    env.agents[0].targets = {((0, 0), d) for d in Grid4TransitionsEnum}

    # Test initial malfunction values for all agents
    # we want some agents to be malfunctioning already and some to be working
    # we want different next_malfunction values for the agents
    malfunction_values = [env.malfunction_generator(env.np_random).num_broken_steps for _ in range(1000)]
    expected_value = (1 - np.exp(-0.5)) * 10
    assert np.allclose(np.mean(malfunction_values), expected_value, rtol=0.1), "Mean values of malfunction don't match rate"


def test_malfunction_values_and_behavior():
    """
    Test the malfunction counts down as desired
    Returns
    -------

    """
    # Set fixed malfunction duration for this test

    rail, rail_map, optionals = make_simple_rail2()
    action_dict: Dict[int, RailEnvActions] = {}
    stochastic_data = MalfunctionParameters(malfunction_rate=1 / 0.001,  # Rate of malfunction occurrence
                                            min_duration=10,  # Minimal duration of malfunction
                                            max_duration=10  # Max duration of malfunction
                                            )
    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=SingleAgentNavigationObs()
                  )

    env.reset(False, False, random_seed=10)

    env._max_episode_steps = 20

    # Assertions
    assert_list = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5]
    for time_step in range(15):
        # Move in the env
        _, _, dones, _ = env.step(action_dict)
        # Check that next_step decreases as expected
        assert env.agents[0].malfunction_handler.malfunction_down_counter == assert_list[time_step]
        if dones['__all__']:
            break


def test_initial_malfunction():
    stochastic_data = MalfunctionParameters(malfunction_rate=1 / 1000,  # Rate of malfunction occurrence
                                            min_duration=2,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static
    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 1000
    print(env.agents[0].malfunction_handler)
    env.agents[0].targets = {((0, 5), d) for d in Grid4TransitionsEnum}
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                set_malfunction=3,
                malfunction=3,
                distance=0.0,
                speed=1.0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,

                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            Replay(  # 1
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MALFUNCTION,
                distance=0.0,
                speed=0.0,
                malfunction=2,

                action=RailEnvActions.MOVE_FORWARD,  # SM: MALFUNCTION -> MOVING needs move action

                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action MOVE_FORWARD, agent should restart and move to the next cell
            Replay(  # 2
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MALFUNCTION,
                malfunction=1,

                action=RailEnvActions.MOVE_FORWARD,

                reward=env.step_penalty

            ),  # malfunctioning ends: starting and running at speed 1.0
            Replay(  # 3
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MALFUNCTION,
                malfunction=0,

                action=RailEnvActions.MOVE_FORWARD,  # SM: MALFUNCTION -> MOVING needs move action

                reward=env.start_penalty + env.step_penalty * 1.0  # running at speed 1.0
            ),
            # design: distance update with pre-step speed.
            Replay(  # 4
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                malfunction=0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,

                reward=env.step_penalty  # running at speed 1.0
            )
        ],
        speed=env.agents[0].speed_counter.speed,
        target=next(iter(env.agents[0].targets))[0],
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [replay_config], skip_reward_check=True, skip_action_required_check=True)


def test_initial_malfunction_stop_moving():
    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=SingleAgentNavigationObs())
    env.reset(False, False, random_seed=10)

    env._max_episode_steps = 1000

    position = env.agents[0].current_entry_point[0] if env.agents[0].current_entry_point is not None else None
    direction = env.agents[0].current_entry_point[1] if env.agents[0].current_entry_point is not None else None
    print(env.agents[0].initial_entry_point[0], direction, position, env.agents[0].state)

    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=None,
                direction=None,
                state=TrainState.READY_TO_DEPART,

                action=RailEnvActions.MOVE_FORWARD,

                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty,  # full step penalty when stopped
            ),
            Replay(  # 1
                position=None,
                direction=None,
                state=TrainState.MALFUNCTION_OFF_MAP,

                action=RailEnvActions.DO_NOTHING,

                malfunction=2,
                reward=env.step_penalty,  # full step penalty when stopped

            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action STOP_MOVING, agent should restart without moving
            #
            Replay(  # 2
                position=None,
                direction=None,
                state=TrainState.MALFUNCTION_OFF_MAP,

                action=RailEnvActions.STOP_MOVING,

                malfunction=1,
                reward=env.step_penalty,  # full step penalty while stopped
            ),
            # need valid movement action to enter the grid
            Replay(  # 3
                position=None,
                direction=None,
                state=TrainState.MALFUNCTION_OFF_MAP,

                action=RailEnvActions.MOVE_FORWARD,  # SM: MALFUNCTION_OFF_MAP -> MOVING needs move action

                malfunction=0,
                reward=env.step_penalty,  # full step penalty while stopped

            ),
            Replay(  # 4
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,

                action=RailEnvActions.STOP_MOVING,

                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0,  # full step penalty while stopped

            ),
            Replay(  # 5
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.STOPPED,

                action=RailEnvActions.MOVE_FORWARD,

                malfunction=0,
                reward=env.step_penalty * 1.0,  # full step penalty while stopped

            ),
            Replay(  # 6
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,

                action=RailEnvActions.STOP_MOVING,

                malfunction=0,
                reward=env.step_penalty * 1.0,  # full step penalty while stopped

            ),
            Replay(  # 7
                position=(3, 4),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.STOPPED,

                action=RailEnvActions.MOVE_FORWARD,

                malfunction=0,
                reward=env.step_penalty * 1.0,  # full step penalty while stopped
            )
        ],
        speed=env.agents[0].speed_counter.speed,
        target=next(iter(env.agents[0].targets))[0],
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )

    run_replay_config(env, [replay_config], activate_agents=False,
                      skip_reward_check=True, set_ready_to_depart=True, skip_action_required_check=True)


def test_stop_moving_crossing_completion_consistent_with_do_nothing():
    """
    STOP_MOVING and DO_NOTHING have the identical pre-step speed at the moment the agent's accumulated
    distance reaches the cell boundary, so per the "distance always advances by pre-step speed" design
    this tick's distance/position update is identical regardless of which action is given - only the
    speed that applies from the NEXT tick onward differs (fixed: issue #178, design D2a). The gating
    condition in rail_env.py step()'s (3b.5) POSITION UPDATE used to special-case
    `stop_action_given and candidate_speed == 0` and block the crossing outright for STOP_MOVING
    (candidate_entry_point was never even set to the next cell); it's now gated purely on
    `is_cell_exit`/`candidate_entry_point_independent`, same as every other action, so STOP_MOVING
    completes the crossing exactly like DO_NOTHING does - it just ends the tick STOPPED (already inside
    the newly-entered cell) rather than blocked short of it.
    """

    def build_env_at_critical_step():
        rail, rail_map, optionals = make_simple_rail2()
        env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                      line_generator=sparse_line_generator(), number_of_agents=1,
                      obs_builder_object=SingleAgentNavigationObs())
        env.reset(False, False, random_seed=10)
        env._max_episode_steps = 1000
        agent = env.agents[0]
        agent.current_entry_point = agent.initial_entry_point
        agent._set_state(TrainState.MOVING)
        # design: distance is None when off map -- the agent is placed directly on the map here,
        # bypassing the state machine's own departure step, so bootstrap distance to 0 explicitly.
        agent.speed_counter.step(speed=agent.speed_counter.speed, crossing_completed=False)
        # design: actions applied at cell entry
        transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, agent.current_entry_point)
        assert transition is not None
        agent.next_entry_point = _sanitize_entry_point(transition)
        # warm up to a tick where the agent's pre-step speed alone reaches the cell boundary
        for _ in range(3):
            env.step({0: RailEnvActions.MOVE_FORWARD})
        return env, agent

    env_stop, agent_stop = build_env_at_critical_step()
    env_nothing, agent_nothing = build_env_at_critical_step()

    # identical starting point for both branches
    pre_entry_point = agent_stop.current_entry_point
    pre_speed = agent_stop.speed_counter.speed
    pre_distance = agent_stop.speed_counter.distance
    assert (agent_nothing.current_entry_point, agent_nothing.speed_counter.speed, agent_nothing.speed_counter.distance) == \
           (pre_entry_point, pre_speed, pre_distance)
    assert pre_distance + pre_speed >= 1  # pre-step speed alone would reach/cross the boundary this tick

    env_stop.step({0: RailEnvActions.STOP_MOVING})
    env_nothing.step({0: RailEnvActions.DO_NOTHING})

    # same pre-step speed, same distance, same intended movement - STOP_MOVING now completes the
    # crossing exactly like DO_NOTHING does; only the resulting state differs (braking vs continuing)
    assert agent_stop.state == TrainState.STOPPED
    assert agent_stop.current_entry_point == agent_nothing.current_entry_point
    assert agent_stop.current_entry_point != pre_entry_point
    assert agent_stop.speed_counter.distance == agent_nothing.speed_counter.distance

    assert agent_nothing.state == TrainState.MOVING
    assert agent_nothing.current_entry_point != pre_entry_point
    assert agent_nothing.speed_counter.distance == 0


def test_stop_moving_wraps_overshoot_beyond_boundary():
    """
    Further consequence of the fix in test_stop_moving_crossing_completion_consistent_with_do_nothing:
    once STOP_MOVING completes an in-flight crossing like any other action, overshoot past the cell
    boundary is preserved (wrapped via `SpeedCounter._distance_update`'s `distance % SEGMENT_LENGTH`),
    not discarded - `distance + pre_speed` reaching `3/2` lands the agent `1/2` into the new cell, not
    capped at exactly the boundary.
    """
    rail, rail_map, optionals = make_simple_rail2()
    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=SingleAgentNavigationObs())
    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 1000
    env.acceleration_delta = Fraction(1, 2)
    env.braking_delta = -Fraction(1)  # full stop in one step, regardless of current speed
    agent = env.agents[0]
    agent.current_entry_point = agent.initial_entry_point
    agent._set_state(TrainState.MOVING)
    # design: actions applied at cell entry
    transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, agent.current_entry_point)
    assert transition is not None
    agent.next_entry_point = _sanitize_entry_point(transition)
    agent.speed_counter = SpeedCounter(speed=Fraction(1, 2), max_speed=Fraction(1, 1))
    # design: distance is None when off map -- the agent is placed directly on the map here,
    # bypassing the state machine's own departure step, so bootstrap distance to 0 explicitly.
    agent.speed_counter.step(speed=agent.speed_counter.speed, crossing_completed=False)

    env.step({0: RailEnvActions.MOVE_FORWARD})
    pre_speed = agent.speed_counter.speed
    pre_distance = agent.speed_counter.distance
    assert (pre_speed, pre_distance) == (Fraction(1), Fraction(1, 2))
    assert pre_distance + pre_speed == Fraction(3, 2)  # well past the boundary, not just reaching it
    pre_entry_point = agent.current_entry_point

    env.step({0: RailEnvActions.STOP_MOVING})

    # wrapped to the 1/2 remainder `(distance+pre_speed) mod 1`, not capped at exactly the boundary
    # and not left at 3/2 either - the crossing completes and the overshoot carries over
    assert agent.state == TrainState.STOPPED
    assert agent.current_entry_point != pre_entry_point
    assert agent.speed_counter.distance == Fraction(1, 2)


def test_initial_malfunction_do_nothing():
    stochastic_data = MalfunctionParameters(malfunction_rate=1 / 70,  # Rate of malfunction occurrence
                                            min_duration=2,  # Minimal duration of malfunction
                                            max_duration=5  # Max duration of malfunction
                                            )

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  # Malfunction data generator
                  )
    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 1000

    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=None,
                direction=None,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty,  # full step penalty while malfunctioning
                state=TrainState.READY_TO_DEPART
            ),
            Replay(  # 1
                position=None,
                direction=None,
                action=None,
                malfunction=2,
                reward=env.step_penalty,  # full step penalty while malfunctioning
                state=TrainState.MALFUNCTION_OFF_MAP
            ),
            Replay(  # 2
                position=None,
                direction=None,
                malfunction=1,
                state=TrainState.MALFUNCTION_OFF_MAP,

                action=None,

                reward=env.step_penalty,  # full step penalty while stopped
            ),
            Replay(  # 3
                position=None,
                direction=None,
                malfunction=0,
                # design: distance is None when off map
                distance=None,
                speed=1,  # irrelevant
                state=TrainState.MALFUNCTION_OFF_MAP,

                action=RailEnvActions.MOVE_FORWARD,  # SM: MALFUNCTION_OFF_MAP -> MOVING needs move action

                reward=env.step_penalty,  # full step penalty while stopped
            ),
            Replay(  # 4
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                distance=0.0,
                speed=1.0,

                malfunction=0,

                action=RailEnvActions.MOVE_FORWARD,

                reward=env.start_penalty + env.step_penalty * 1.0,  # start penalty + step penalty for speed 1.0
            ),  # we start to move forward --> should go to next cell now
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                malfunction=0,
                distance=0.0,
                speed=1.0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,

                reward=env.step_penalty * 1.0,  # step penalty for speed 1.0

            )
        ],
        speed=env.agents[0].speed_counter.speed,
        target=next(iter(env.agents[0].targets))[0],
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [replay_config], activate_agents=False,
                      skip_reward_check=True,
                      set_ready_to_depart=True,
                      skip_action_required_check=False
                      )


def tests_random_interference_from_outside():
    """Tests that malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    rail, rail_map, optionals = make_simple_rail2()
    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=2), number_of_agents=1, random_seed=1)
    env.reset()
    env.agents[0].speed_counter = SpeedCounter(speed=0.33)
    env.reset(False, False, random_seed=10)
    env_data = []

    for step in range(200):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # We randomly select an action
            action_dict[agent.handle] = RailEnvActions.MOVE_FORWARD

        _, reward, dones, _ = env.step(action_dict)
        # Append the rewards of the first trial
        position = env.agents[0].current_entry_point[0] if env.agents[0].current_entry_point is not None else None
        env_data.append((reward[0], position))
        assert reward[0] == env_data[step][0]
        assert position == env_data[step][1]
        if dones['__all__']:
            break
    # Run the same test as above but with an external random generator running
    # Check that the reward stays the same

    rail, rail_map, optionals = make_simple_rail2()
    random.seed(47)
    np.random.seed(1234)
    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=2), number_of_agents=1, random_seed=1)
    env.reset()
    env.agents[0].speed_counter = SpeedCounter(speed=0.33)
    env.reset(False, False, random_seed=10)

    dummy_list = [1, 2, 6, 7, 8, 9, 4, 5, 4]
    for step in range(200):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # We randomly select an action
            action_dict[agent.handle] = RailEnvActions.MOVE_FORWARD

            # Do dummy random number generations
            random.shuffle(dummy_list)
            np.random.rand()

        _, reward, dones, _ = env.step(action_dict)
        assert reward[0] == env_data[step][0]
        position = env.agents[0].current_entry_point[0] if env.agents[0].current_entry_point is not None else None
        assert position == env_data[step][1]
        if dones['__all__']:
            break


def test_last_malfunction_step():
    """
    Test to check that agent moves when it is not malfunctioning

    """

    # Set fixed malfunction duration for this test

    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=2), number_of_agents=1, random_seed=1)
    env.reset()
    env.agents[0].speed_counter = SpeedCounter(speed=1. / 3.)
    env.agents[0].initial_entry_point = ((6, 6), 2)
    env.agents[0].targets = {((0, 3), d) for d in Grid4TransitionsEnum}

    env._max_episode_steps = 1000

    env.reset(False, False, random_seed=10)
    assert len(set([a.initial_entry_point[0] for a in env.agents])) == 1
    for a_idx in range(len(env.agents)):
        env_agent = env.agents[a_idx]
        env_agent.current_entry_point = env_agent.initial_entry_point
        env_agent.state = TrainState.MOVING
        # design: distance is None when off map -- the agent is placed directly on the map here,
        # bypassing the state machine's own departure step, so bootstrap distance to 0 explicitly.
        env_agent.speed_counter.step(speed=env_agent.speed_counter.speed, crossing_completed=False)
        # design: actions applied at cell entry -- keep the current_entry_point/next_entry_point
        # invariant (both set and different while on-map) even for this direct test-harness setup.
        transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, env_agent.current_entry_point)
        assert transition is not None
        env_agent.next_entry_point = _sanitize_entry_point(transition)
    env.agents[0].malfunction_handler.malfunction_down_counter = 0

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    for step in range(20):
        # Go forward all the time
        action_dict: Dict[int, RailEnvActions] = {agent.handle: RailEnvActions.MOVE_FORWARD for agent in env.agents}

        if env.agents[0].malfunction_handler.malfunction_down_counter < 1:
            agent_can_move = True
        # Store the position before and after the step
        pre_position = env.agents[0].speed_counter.distance
        _, reward, _, _ = env.step(action_dict)
        # Check if the agent is still allowed to move in this step

        if env.agents[0].malfunction_handler.malfunction_down_counter > 0:
            agent_can_move = False
        post_position = env.agents[0].speed_counter.distance
        # Assert that the agent moved while it was still allowed
        if agent_can_move:
            assert pre_position != post_position
        else:
            assert post_position == pre_position
