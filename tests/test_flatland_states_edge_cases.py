from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.step_utils.states import TrainState
from flatland.utils.simple_rail import make_simple_rail


def test_return_to_ready_to_depart():
    """
    When going from ready to depart to malfunction off map, if do nothing is provided, should return to ready to depart
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 100

    for _ in range(3):
        env.step({0: RailEnvActions.DO_NOTHING})

    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    env.step({0: RailEnvActions.DO_NOTHING})

    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP

    for _ in range(2):
        env.step({0: RailEnvActions.DO_NOTHING})

    assert env.agents[0].state == TrainState.READY_TO_DEPART


def test_ready_to_depart_to_stopped():
    """
    When going from ready to depart to malfunction off map, if stopped is provided, should go to stopped
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 100

    for _ in range(3):
        env.step({0: RailEnvActions.STOP_MOVING})

    assert env.agents[0].state == TrainState.READY_TO_DEPART

    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    env.step({0: RailEnvActions.STOP_MOVING})

    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP

    for _ in range(2):
        env.step({0: RailEnvActions.STOP_MOVING})

    assert env.agents[0].state == TrainState.STOPPED


def test_malfunction_no_phase_through():
    """
    A moving train shouldn't phase through a malfunctioning train
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    for _ in range(5):
        env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    env.agents[1].malfunction_handler._set_malfunction_down_counter(10)

    for _ in range(3):
        env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})

    assert env.agents[0].state == TrainState.STOPPED
    assert env.agents[0].position == (3, 5)


def test_malfunction_off_map_not_on_map_with_stop_action_after_malfunction():
    """
    MALFUNCTION_OFF_MAP getting into map must respect without motion check.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_position = (6, 6)
    env.agents[0].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[0].target = (0, 3)
    env.agents[0].earliest_departure = 0

    env.agents[1].initial_position = (6, 6)
    env.agents[1].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[1].target = (0, 3)
    env.agents[1].earliest_departure = 0
    env.agents[1].malfunction_handler._set_malfunction_down_counter(2)

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP

    assert env.agents[1].malfunction_handler.malfunction_down_counter == 1

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MOVING

    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP
    assert env.agents[1].malfunction_handler.malfunction_down_counter == 0

    # step 3
    env.step({0: RailEnvActions.STOP_MOVING, 1: RailEnvActions.STOP_MOVING})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.STOPPED

    # / TEMPORARY FIX FOR MALFUNCTION_OFF_MAP getting into map without motion check
    #   WITHOUT FIX: STOPPED on map in the same cell as agent 0, not respecting motion check!
    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.READY_TO_DEPART
    # \ TEMPORARY FIX


def test_malfunction_motion_check_order_when_earliest_departure_is_not_reached():
    """
    Avoid adding agent to motion check as it can hinder other agents having earliest_departure_reached to start.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_position = (6, 6)
    env.agents[0].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[0].target = (0, 3)
    env.agents[0].earliest_departure = 55
    env.agents[0].malfunction_handler._set_malfunction_down_counter(1)

    env.agents[1].initial_position = (6, 6)
    env.agents[1].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[1].target = (0, 3)
    env.agents[1].earliest_departure = 2

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 0

    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.WAITING

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.WAITING
    # this is the root cause: the motion check for agent 0 returns OK, the action preprocessing converts to DO_NOTHING only in state WAITING, but not MALFUNCTION_OFF_MAP
    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.READY_TO_DEPART

    # step 3
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.WAITING

    # / TEMPORARY FIX as adding agent to motion check can hinder other agents having earliest_departure_reached to start
    # WITHOUT FIX: both agents are inserted into motion check, and the one with lower index wins:
    # - agent 0 with the lower index wins, despite not having reached earliest departure
    # - agent 1 is blocked, despite having reached earliest departure
    assert env.agents[1].position == (6, 6)
    assert env.agents[1].state == TrainState.MOVING
    # \ TEMPORARY FIX


def test_malfunction_motion_check_order_when_earliest_departure_reached_but_not_moving_action():
    """
    Avoid adding agent to motion check as it can hinder other agents having earliest_departure_reached to start.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_position = (6, 6)
    env.agents[0].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[0].target = (0, 3)
    env.agents[0].earliest_departure = 3
    env.agents[0].malfunction_handler._set_malfunction_down_counter(1)

    env.agents[1].initial_position = (6, 6)
    env.agents[1].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[1].target = (0, 3)
    env.agents[1].earliest_departure = 2

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 0

    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.WAITING

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.WAITING
    # this is the root cause: the motion check for agent 0 returns OK, the action preprocessing converts to DO_NOTHING only in state WAITING, but not MALFUNCTION_OFF_MAP
    assert env.agents[1].position == None
    assert env.agents[1].state == TrainState.READY_TO_DEPART

    # step 3
    env.step({0: RailEnvActions.DO_NOTHING, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    # / TEMPORARY FIX as adding agent to motion check can hinder other agents having earliest_departure_reached to start
    # WITHOUT FIX:
    # - agent 0 with the lower index wins, despite sending DO_NOTHING
    # - agent 1 is blocked, despite having reached earliest departure
    assert env.agents[1].position == (6, 6)
    assert env.agents[1].state == TrainState.MOVING
    # \ TEMPORARY FIX


def test_malfunction_to_moving_instead_of_stopped():
    """
    MALFUNCTION to MOVING without going to STOPPED unnecessarily
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_position = (6, 6)
    env.agents[0].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[0].target = (0, 3)
    env.agents[0].earliest_departure = 0
    env.agents[0].speed_counter._speed = 0.2
    env.agents[0].speed_counter._max_speed = 0.2

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert env.agents[0].speed_counter.speed == 0.2
    # N.B. no movement in first time step after READY_TO_DEPART or MALFUNCTION_OFF_MAP!
    assert env.agents[0].speed_counter.distance == 0.0

    # step 3
    env.agents[0].malfunction_handler._set_malfunction_down_counter(1)
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MALFUNCTION
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 0
    assert env.agents[0].speed_counter.speed == 0.0
    assert env.agents[0].speed_counter.distance == 0.0

    # step 4
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    # / TEMPORARY FIX avoid setting agent to STOPPED after malfunction unnecessarily
    # WITHOUT FIX: agent is STOPPED
    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert env.agents[0].speed_counter.speed == 0.2
    assert env.agents[0].speed_counter.distance == 0.2
    # \ TEMPORARY FIX


def test_stop_and_go():
    """
    Test stop and go.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_position = (6, 6)
    env.agents[0].initial_direction = Grid4TransitionsEnum.SOUTH
    env.agents[0].target = (0, 3)
    env.agents[0].earliest_departure = 0
    env.agents[0].speed_counter._speed = 0.2
    env.agents[0].speed_counter._max_speed = 0.2

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert env.agents[0].speed_counter.speed == 0.2
    # N.B. no movement in first time step after READY_TO_DEPART or MALFUNCTION_OFF_MAP!
    assert env.agents[0].speed_counter.distance == 0.0

    # step 3
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert env.agents[0].speed_counter.speed == 0.2
    assert env.agents[0].speed_counter.distance == 0.2

    # step 4
    env.step({0: RailEnvActions.STOP_MOVING, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.STOPPED
    assert env.agents[0].speed_counter.speed == 0.0
    assert env.agents[0].speed_counter.distance == 0.2

    # step 5
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].position == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert env.agents[0].speed_counter.speed == 0.2
    assert env.agents[0].speed_counter.distance == 0.4
