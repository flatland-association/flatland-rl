
from attr import attrs, attrib
from itertools import starmap
import numpy as np
# from flatland.envs.rail_env import RailEnv


@attrs
class EnvDescription(object):
    """ EnvDescription - This is a description of a random env,
        based around the rail_generator and stats like size and n_agents.
        It mirrors the parameters given to the RailEnv constructor.
        Not currently used.
    """
    n_agents = attrib()
    height = attrib()
    width = attrib()
    rail_generator = attrib()
    obs_builder = attrib()   # not sure if this should closer to the agent than the env


@attrs
class EnvAgentStatic(object):
    """ EnvAgentStatic - Stores initial position, direction and target.
        This is like static data for the environment - it's where an agent starts,
        rather than where it is at the moment.
        The target should also be stored here.
    """
    position = attrib()
    direction = attrib()
    target = attrib()

    def __init__(self, position, direction, target):
        self.position = position
        self.direction = direction
        self.target = target

    @classmethod
    def from_lists(cls, positions, directions, targets):
        """ Create a list of EnvAgentStatics from lists of positions, directions and targets
        """
        return list(starmap(EnvAgentStatic, zip(positions, directions, targets)))

    def to_list(self):

        # I can't find an expression which works on both tuples, lists and ndarrays
        # which converts them all to a list of native python ints.
        lPos = self.position
        if type(lPos) is np.ndarray:
            lPos = lPos.tolist()

        lTarget = self.target
        if type(lTarget) is np.ndarray:
            lTarget = lTarget.tolist()

        return [lPos, int(self.direction), lTarget]


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

    def __init__(self, position, direction, target, handle, old_direction, old_position):
        super(EnvAgent, self).__init__(position, direction, target)
        self.handle = handle
        self.old_direction = old_direction
        self.old_position = old_position

    def to_list(self):
        return [
            self.position, self.direction, self.target, self.handle, 
            self.old_direction, self.old_position]

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
