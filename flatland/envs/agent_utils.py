from enum import IntEnum
from itertools import starmap
from typing import Tuple, Optional

from attr import attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.schedule_utils import Schedule


class RailAgentStatus(IntEnum):
    READY_TO_DEPART = 0  # not in grid yet (position is None) -> prediction as if it were at initial position
    ACTIVE = 1  # in grid (position is not None), not done -> prediction is remaining path
    DONE = 2  # in grid (position is not None), but done -> prediction is stay at target forever
    DONE_REMOVED = 3  # removed from grid (position is None) -> prediction is None


@attrs
class EnvAgent:

    initial_position = attrib(type=Tuple[int, int])
    initial_direction = attrib(type=Grid4TransitionsEnum)
    direction = attrib(type=Grid4TransitionsEnum)
    target = attrib(type=Tuple[int, int])
    moving = attrib(default=False, type=bool)

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

    status = attrib(default=RailAgentStatus.READY_TO_DEPART, type=RailAgentStatus)
    position = attrib(default=None, type=Optional[Tuple[int, int]])

    # used in rendering
    old_direction = attrib(default=None)
    old_position = attrib(default=None)

    def reset(self):
        self.position = None
        self.direction = self.initial_direction
        self.status = RailAgentStatus.READY_TO_DEPART
        self.old_position = None
        self.old_direction = None
        self.moving = False

    def to_list(self):
        return [self.initial_position, self.initial_direction, int(self.direction), self.target, int(self.moving),
                self.speed_data, self.malfunction_data, self.handle, self.status, self.position, self.old_direction,
                self.old_position]

    @classmethod
    def from_schedule(cls, schedule: Schedule):
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        speed_datas = []

        for i in range(len(schedule.agent_positions)):
            speed_datas.append({'position_fraction': 0.0,
                                'speed': schedule.agent_speeds[i] if schedule.agent_speeds is not None else 1.0,
                                'transition_action_on_cellexit': 0})

        malfunction_datas = []
        for i in range(len(schedule.agent_positions)):
            malfunction_datas.append({'malfunction': 0,
                                      'malfunction_rate': schedule.agent_malfunction_rates[
                                          i] if schedule.agent_malfunction_rates is not None else 0.,
                                      'next_malfunction': 0,
                                      'nr_malfunctions': 0})

        return list(starmap(EnvAgent, zip(schedule.agent_positions,
                                          schedule.agent_directions,
                                          schedule.agent_directions,
                                          schedule.agent_targets,
                                          [False] * len(schedule.agent_positions),
                                          speed_datas,
                                          malfunction_datas,
                                          range(len(schedule.agent_positions)))))
