from enum import IntEnum
from itertools import starmap
from typing import Tuple, Optional

import numpy as np
from attr import attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum


class RailAgentStatus(IntEnum):
    READY_TO_DEPART = 0  # not in grid yet (position is None) -> prediction as if it were at initial position
    ACTIVE = 1  # in grid (position is not None), not done -> prediction is remaining path
    DONE = 2  # in grid (position is not None), but done -> prediction is stay at target forever
    DONE_REMOVED = 3  # removed from grid (position is None) -> prediction is None


@attrs
class EnvAgentStatic(object):
    """ EnvAgentStatic - Stores initial position, direction and target.
        This is like static data for the environment - it's where an agent starts,
        rather than where it is at the moment.
        The target should also be stored here.
    """
    initial_position = attrib(type=Tuple[int, int])
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

    status = attrib(default=RailAgentStatus.READY_TO_DEPART, type=RailAgentStatus)
    position = attrib(default=None, type=Optional[Tuple[int, int]])

    @classmethod
    def from_lists(cls, positions, directions, targets, speeds=None, malfunction_rates=None):
        """ Create a list of EnvAgentStatics from lists of positions, directions and targets
        """
        speed_datas = []

        for i in range(len(positions)):
            speed_datas.append({'position_fraction': 0.0,
                                'speed': speeds[i] if speeds is not None else 1.0,
                                'transition_action_on_cellexit': 0})

        # TODO: on initialization, all agents are re-set as non-broken. Perhaps it may be desirable to set
        # some as broken?

        malfunction_datas = []
        for i in range(len(positions)):
            malfunction_datas.append({'malfunction': 0,
                                      'malfunction_rate': malfunction_rates[i] if malfunction_rates is not None else 0.,
                                      'next_malfunction': 0,
                                      'nr_malfunctions': 0})

        return list(starmap(EnvAgentStatic, zip(positions,
                                                directions,
                                                targets,
                                                [False] * len(positions),
                                                speed_datas,
                                                malfunction_datas)))

    def to_list(self):

        # I can't find an expression which works on both tuples, lists and ndarrays
        # which converts them all to a list of native python ints.
        lPos = self.initial_position
        if type(lPos) is np.ndarray:
            lPos = lPos.tolist()

        lTarget = self.target
        if type(lTarget) is np.ndarray:
            lTarget = lTarget.tolist()

        return [lPos, int(self.direction), lTarget, int(self.moving), self.speed_data, self.malfunction_data]


@attrs
class EnvAgent(EnvAgentStatic):
    """ EnvAgent - replace separate agent_* lists with a single list
        of agent objects.  The EnvAgent represent's the environment's view
        of the dynamic agent state.
        We are duplicating target in the EnvAgent, which seems simpler than
        forcing the env to refer to it in the EnvAgentStatic
    """
    handle = attrib(default=None)
    old_direction = attrib(default=None)
    old_position = attrib(default=None)

    def to_list(self):
        return [
            self.position, self.direction, self.target, self.handle,
            self.old_direction, self.old_position, self.moving, self.speed_data, self.malfunction_data]

    @classmethod
    def from_static(cls, oStatic):
        """ Create an EnvAgent from the EnvAgentStatic,
        copying all the fields, and adding handle with the default 0.
        """
        return EnvAgent(*oStatic.__dict__, handle=0)

    @classmethod
    def list_from_static(cls, lEnvAgentStatic, handles=None):
        """ Create an EnvAgent from the EnvAgentStatic,
        copying all the fields, and adding handle with the default 0.
        """
        if handles is None:
            handles = range(len(lEnvAgentStatic))

        return [EnvAgent(**oEAS.__dict__, handle=handle)
                for handle, oEAS in zip(handles, lEnvAgentStatic)]
