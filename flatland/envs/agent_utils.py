from flatland.envs.rail_trainrun_data_structures import Waypoint
import numpy as np

from itertools import starmap
from typing import Tuple, Optional, NamedTuple, List

from attr import attr, attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.timetable_utils import Line

from flatland.envs.step_utils.action_saver import ActionSaver
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState
from flatland.envs.step_utils.malfunction_handler import MalfunctionHandler

Agent = NamedTuple('Agent', [('initial_position', Tuple[int, int]),
                             ('initial_direction', Grid4TransitionsEnum),
                             ('direction', Grid4TransitionsEnum),
                             ('target', Tuple[int, int]),
                             ('moving', bool),
                             ('earliest_departure', int),
                             ('latest_arrival', int),
                             ('speed_data', dict),
                             ('malfunction_data', dict),
                             ('handle', int),
                             ('position', Tuple[int, int]),
                             ('arrival_time', int),
                             ('old_direction', Grid4TransitionsEnum),
                             ('old_position', Tuple[int, int]),
                             ('speed_counter', SpeedCounter),
                             ('action_saver', ActionSaver),
                             ('state', TrainState),
                             ('state_machine', TrainStateMachine),
                             ('malfunction_handler', MalfunctionHandler),
                             ])


@attrs
class EnvAgent:
    # INIT FROM HERE IN _from_line()
    initial_position = attrib(type=Tuple[int, int])
    initial_direction = attrib(type=Grid4TransitionsEnum)
    direction = attrib(type=Grid4TransitionsEnum)
    target = attrib(type=Tuple[int, int])
    moving = attrib(default=False, type=bool)

    # NEW : EnvAgent - Schedule properties
    earliest_departure = attrib(default=None, type=int)  # default None during _from_line()
    latest_arrival = attrib(default=None, type=int)  # default None during _from_line()

    # speed_data: speed is added to position_fraction on each moving step, until position_fraction>=1.0,
    # after which 'transition_action_on_cellexit' is executed (equivalent to executing that action in the previous
    # cell if speed=1, as default)
    # N.B. we need to use factory since default arguments are not recreated on each call!
    speed_data = attrib(
        default=Factory(lambda: dict({'position_fraction': 0.0, 'speed': 1.0, 'transition_action_on_cellexit': 0})))

    # if broken>0, the agent's actions are ignored for 'broken' steps
    # number of time the agent had to stop, since the last time it broke down
    malfunction_data = attrib(
        default=Factory(
            lambda: dict({'malfunction': 0, 'malfunction_rate': 0, 'next_malfunction': 0, 'nr_malfunctions': 0,
                          'moving_before_malfunction': False})))

    handle = attrib(default=None)
    # INIT TILL HERE IN _from_line()

    # Env step facelift
    speed_counter = attrib(default = None, type=SpeedCounter)
    action_saver = attrib(default = Factory(lambda: ActionSaver()), type=ActionSaver)
    state_machine = attrib(default= Factory(lambda: TrainStateMachine(initial_state=TrainState.WAITING)) , 
                           type=TrainStateMachine)
    malfunction_handler = attrib(default = Factory(lambda: MalfunctionHandler()), type=MalfunctionHandler)
    
    state = attrib(default=TrainState.WAITING, type=TrainState)

    position = attrib(default=None, type=Optional[Tuple[int, int]])

    # NEW : EnvAgent Reward Handling
    arrival_time = attrib(default=None, type=int)

    # used in rendering
    old_direction = attrib(default=None)
    old_position = attrib(default=None)


    def reset(self):
        """
        Resets the agents to their initial values of the episode. Called after ScheduleTime generation.
        """
        self.position = None
        # TODO: set direction to None: https://gitlab.aicrowd.com/flatland/flatland/issues/280
        self.direction = self.initial_direction
        self.old_position = None
        self.old_direction = None
        self.moving = False

        # Reset agent values for speed
        self.speed_data['position_fraction'] = 0.
        self.speed_data['transition_action_on_cellexit'] = 0.

        # Reset agent malfunction values
        self.malfunction_data['malfunction'] = 0
        self.malfunction_data['nr_malfunctions'] = 0
        self.malfunction_data['moving_before_malfunction'] = False

        self.action_saver.clear_saved_action()
        self.speed_counter.reset_counter()
        self.state_machine.reset()

    def to_agent(self) -> Agent:
        return Agent(initial_position=self.initial_position, 
                     initial_direction=self.initial_direction,
                     direction=self.direction,
                     target=self.target,
                     moving=self.moving,
                     earliest_departure=self.earliest_departure, 
                     latest_arrival=self.latest_arrival, 
                     speed_data=self.speed_data,
                     malfunction_data=self.malfunction_data, 
                     handle=self.handle, 
                     state=self.state,
                     position=self.position, 
                     old_direction=self.old_direction, 
                     old_position=self.old_position,
                     speed_counter=self.speed_counter,
                     action_saver=self.action_saver,
                     state_machine=self.state_machine,
                     malfunction_handler=self.malfunction_handler)

    @classmethod
    def from_line(cls, line: Line):
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        speed_datas = []
        speed_counters = []
        for i in range(len(line.agent_positions)):
            speed = line.agent_speeds[i] if line.agent_speeds is not None else 1.0
            speed_datas.append({'position_fraction': 0.0,
                                'speed': speed,
                                'transition_action_on_cellexit': 0})
            speed_counters.append( SpeedCounter(speed=speed) )

        malfunction_datas = []
        for i in range(len(line.agent_positions)):
            malfunction_datas.append({'malfunction': 0,
                                      'malfunction_rate': line.agent_malfunction_rates[
                                          i] if line.agent_malfunction_rates is not None else 0.,
                                      'next_malfunction': 0,
                                      'nr_malfunctions': 0})
        
        return list(starmap(EnvAgent, zip(line.agent_positions,  # TODO : Dipam - Really want to change this way of loading agents
                                          line.agent_directions,
                                          line.agent_directions,
                                          line.agent_targets, 
                                          [False] * len(line.agent_positions), 
                                          [None] * len(line.agent_positions), # earliest_departure
                                          [None] * len(line.agent_positions), # latest_arrival
                                          speed_datas,
                                          malfunction_datas,
                                          range(len(line.agent_positions)),
                                          speed_counters,
                                          )))

    @classmethod
    def load_legacy_static_agent(cls, static_agents_data: Tuple):
        agents = []
        for i, static_agent in enumerate(static_agents_data):
            if len(static_agent) >= 6:
                agent = EnvAgent(initial_position=static_agent[0], initial_direction=static_agent[1],
                                direction=static_agent[1], target=static_agent[2], moving=static_agent[3],
                                speed_data=static_agent[4], malfunction_data=static_agent[5], handle=i)
            else:
                agent = EnvAgent(initial_position=static_agent[0], initial_direction=static_agent[1],
                                direction=static_agent[1], target=static_agent[2], 
                                moving=False,
                                speed_data={"speed":1., "position_fraction":0., "transition_action_on_cell_exit":0.},
                                malfunction_data={
                                            'malfunction': 0,
                                            'nr_malfunctions': 0,
                                            'moving_before_malfunction': False
                                        },
                                handle=i)
            agents.append(agent)
        return agents
