from enum import IntEnum
from itertools import starmap
from typing import Tuple, Optional, NamedTuple

from attr import attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.schedule_utils import Schedule


class RailAgentStatus(IntEnum):
    READY_TO_DEPART = 0  # not in grid yet (position is None) -> prediction as if it were at initial position
    ACTIVE = 1  # in grid (position is not None), not done -> prediction is remaining path
    DONE = 2  # in grid (position is not None), but done -> prediction is stay at target forever
    DONE_REMOVED = 3  # removed from grid (position is None) -> prediction is None


Agent = NamedTuple('Agent', [('initial_position', Tuple[int, int]),
                             ('initial_direction', Grid4TransitionsEnum),
                             ('direction', Grid4TransitionsEnum),
                             ('target', Tuple[int, int]),
                             ('moving', bool),
                             ('speed_data', dict),
                             ('malfunction_data', dict),
                             ('handle', int),
                             ('status', RailAgentStatus),
                             ('position', Tuple[int, int]),
                             ('old_direction', Grid4TransitionsEnum),
                             ('old_position', Tuple[int, int])])


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
        """
        Resets the agents to their initial values of the episode
        """
        self.position = None
        # TODO: set direction to None: https://gitlab.aicrowd.com/flatland/flatland/issues/280
        self.direction = self.initial_direction
        self.status = RailAgentStatus.READY_TO_DEPART
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

    def to_agent(self) -> Agent:
        return Agent(initial_position=self.initial_position, initial_direction=self.initial_direction,
                     direction=self.direction, target=self.target, moving=self.moving, speed_data=self.speed_data,
                     malfunction_data=self.malfunction_data, handle=self.handle, status=self.status,
                     position=self.position, old_direction=self.old_direction, old_position=self.old_position)

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

    @classmethod
    def load_legacy_static_agent(cls, static_agents_data: Tuple):
        agents = []
        for i, static_agent in enumerate(static_agents_data):
            agent = EnvAgent(initial_position=static_agent[0], initial_direction=static_agent[1],
                             direction=static_agent[1], target=static_agent[2], moving=static_agent[3],
                             speed_data=static_agent[4], malfunction_data=static_agent[5], handle=i)
            agents.append(agent)
        return agents
