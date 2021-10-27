from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.simple_rail import make_simple_rail
from flatland.envs.step_utils.states import TrainState

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