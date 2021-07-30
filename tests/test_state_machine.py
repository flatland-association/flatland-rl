from test_env_step_utils import get_small_two_agent_env
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.envs.malfunction_generators import Malfunction

class NoMalfunctionGenerator:
    def generate(self, np_random):
        return Malfunction(0)

class AlwaysThreeStepMalfunction:
    def generate(self, np_random):
        return Malfunction(3)

def test_waiting_no_transition():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 0
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed-1):
        env.step({i_agent: RailEnvActions.MOVE_FORWARD})
        assert env.agents[i_agent].state == TrainState.WAITING
    
    
def test_waiting_to_ready_to_depart():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 0
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING})
    assert env.agents[i_agent].state == TrainState.READY_TO_DEPART


def test_ready_to_depart_to_moving():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 0
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING})

    env.step({i_agent: RailEnvActions.MOVE_FORWARD})
    assert env.agents[i_agent].state == TrainState.MOVING

def test_moving_to_stopped():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 0
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING})

    env.step({i_agent: RailEnvActions.MOVE_FORWARD})
    env.step({i_agent: RailEnvActions.STOP_MOVING})
    assert env.agents[i_agent].state == TrainState.STOPPED

def test_stopped_to_moving():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 0
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING})

    env.step({i_agent: RailEnvActions.MOVE_FORWARD})
    env.step({i_agent: RailEnvActions.STOP_MOVING})
    env.step({i_agent: RailEnvActions.MOVE_FORWARD})
    assert env.agents[i_agent].state == TrainState.MOVING

def test_moving_to_done():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 1
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING})

    for _ in range(50):
        env.step({i_agent: RailEnvActions.MOVE_FORWARD})
    assert env.agents[i_agent].state == TrainState.DONE

def test_waiting_to_malfunction():
    env = get_small_two_agent_env()
    env.malfunction_generator = AlwaysThreeStepMalfunction()
    i_agent = 1
    env.step({i_agent: RailEnvActions.DO_NOTHING})
    assert env.agents[i_agent].state == TrainState.MALFUNCTION_OFF_MAP


def test_ready_to_depart_to_malfunction_off_map():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 1
    env.step({i_agent: RailEnvActions.DO_NOTHING})
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING}) # This should get into ready to depart
        
    env.malfunction_generator = AlwaysThreeStepMalfunction()
    env.step({i_agent: RailEnvActions.DO_NOTHING})
    assert env.agents[i_agent].state == TrainState.MALFUNCTION_OFF_MAP


def test_malfunction_off_map_to_waiting():
    env = get_small_two_agent_env()
    env.malfunction_generator = NoMalfunctionGenerator()
    i_agent = 1
    env.step({i_agent: RailEnvActions.DO_NOTHING})
    ed = env.agents[i_agent].earliest_departure
    for _ in range(ed):
        env.step({i_agent: RailEnvActions.DO_NOTHING}) # This should get into ready to depart
        
    env.malfunction_generator = AlwaysThreeStepMalfunction()
    env.step({i_agent: RailEnvActions.DO_NOTHING})
    assert env.agents[i_agent].state == TrainState.MALFUNCTION_OFF_MAP