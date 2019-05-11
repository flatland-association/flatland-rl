
from attr import attrs, attrib
from itertools import starmap
# from flatland.envs.rail_env import RailEnv


@attrs
class EnvDescription(object):
    n_agents = attrib()
    height = attrib()
    width = attrib()
    rail_generator = attrib()
    obs_builder = attrib()   # not sure if this should closer to the agent than the env


@attrs
class EnvAgentStatic(object):
    """ TODO: EnvAgentStatic - To store initial position, direction and target.
        This is like static data for the environment - it's where an agent starts,
        rather than where it is at the moment.
        The target should also be stored here.
    """
    position = attrib()
    direction = attrib()
    target = attrib()

    next_handle = 0  # this is not properly implemented

    @classmethod
    def from_lists(cls, positions, directions, targets):
        """ Create a list of EnvAgentStatics from lists of positions, directions and targets
        """
        return list(starmap(EnvAgentStatic, zip(positions, directions, targets)))
        

@attrs
class EnvAgent(EnvAgentStatic):
    """ EnvAgent - replace separate agent_* lists with a single list
        of agent objects.  The EnvAgent represent's the environment's view
        of the dynamic agent state.
        We are duplicating target in the EnvAgent, which seems simpler than
        forcing the env to refer to it in the EnvAgentStatic
    """
    handle = attrib(default=None)

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

