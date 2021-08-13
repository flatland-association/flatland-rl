from flatland.envs.rail_trainrun_data_structures import Waypoint
import numpy as np

from enum import IntEnum
from itertools import starmap
from typing import Tuple, Optional, NamedTuple, List

from attr import attr, attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.timetable_utils import Line

class RailAgentStatus(IntEnum):
    WAITING = 0
    READY_TO_DEPART = 1  # not in grid yet (position is None) -> prediction as if it were at initial position
    ACTIVE = 2  # in grid (position is not None), not done -> prediction is remaining path
    DONE = 3  # in grid (position is not None), but done -> prediction is stay at target forever
    DONE_REMOVED = 4  # removed from grid (position is None) -> prediction is None


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
                             ('status', RailAgentStatus),
                             ('position', Tuple[int, int]),
                             ('arrival_time', int),
                             ('old_direction', Grid4TransitionsEnum),
                             ('old_position', Tuple[int, int])])


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

    status = attrib(default=RailAgentStatus.WAITING, type=RailAgentStatus)
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

        if (self.earliest_departure == 0):
            self.status = RailAgentStatus.READY_TO_DEPART
        else:
            self.status = RailAgentStatus.WAITING
            
        self.arrival_time = None

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

    # NEW : Callables
    def get_shortest_path(self, distance_map) -> List[Waypoint]:
        from flatland.envs.rail_env_shortest_paths import get_shortest_paths # Circular dep fix
        return get_shortest_paths(distance_map=distance_map, agent_handle=self.handle)[self.handle]
        
    def get_travel_time_on_shortest_path(self, distance_map) -> int:
        shortest_path = self.get_shortest_path(distance_map)
        if shortest_path is not None:
            distance = len(shortest_path)
        else:
            distance = 0
        speed = self.speed_data['speed']
        return int(np.ceil(distance / speed))

    def get_time_remaining_until_latest_arrival(self, elapsed_steps: int) -> int:
        return self.latest_arrival - elapsed_steps

    def get_current_delay(self, elapsed_steps: int, distance_map) -> int:
        '''
        +ve if arrival time is projected before latest arrival
        -ve if arrival time is projected after latest arrival
        '''
        return self.get_time_remaining_until_latest_arrival(elapsed_steps) - \
               self.get_travel_time_on_shortest_path(distance_map)

    def to_agent(self) -> Agent:
        return Agent(initial_position=self.initial_position, initial_direction=self.initial_direction, 
                     direction=self.direction, target=self.target, moving=self.moving, earliest_departure=self.earliest_departure, 
                     latest_arrival=self.latest_arrival, speed_data=self.speed_data, malfunction_data=self.malfunction_data, 
                     handle=self.handle, status=self.status, position=self.position, arrival_time=self.arrival_time, 
                     old_direction=self.old_direction, old_position=self.old_position)

    @classmethod
    def from_line(cls, line: Line):
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        speed_datas = []

        for i in range(len(line.agent_positions)):
            speed_datas.append({'position_fraction': 0.0,
                                'speed': line.agent_speeds[i] if line.agent_speeds is not None else 1.0,
                                'transition_action_on_cellexit': 0})

        malfunction_datas = []
        for i in range(len(line.agent_positions)):
            malfunction_datas.append({'malfunction': 0,
                                      'malfunction_rate': line.agent_malfunction_rates[
                                          i] if line.agent_malfunction_rates is not None else 0.,
                                      'next_malfunction': 0,
                                      'nr_malfunctions': 0})

        return list(starmap(EnvAgent, zip(line.agent_positions,
                                          line.agent_directions,
                                          line.agent_directions,
                                          line.agent_targets, 
                                          [False] * len(line.agent_positions), 
                                          [None] * len(line.agent_positions), # earliest_departure
                                          [None] * len(line.agent_positions), # latest_arrival
                                          speed_datas,
                                          malfunction_datas,
                                          range(len(line.agent_positions)))))

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
