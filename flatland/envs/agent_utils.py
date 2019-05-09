
from attr import attrs, attrib
from itertools import starmap, count
import numpy as np

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
    handle = attrib()

    next_handle = 0

    @classmethod
    def from_lists(positions, directions, targets):
        """ Create a list of EnvAgentStatics from lists of positions, directions and targets
        """
        return starmap(EnvAgentStatic, zip(positions, directions, targets, count()))
        

class EnvAgent(EnvAgentStatic):
    """ TODO: EnvAgent - replace separate agent lists with a single list
        of agent objects.  The EnvAgent represent's the environment's view
        of the dynamic agent state.  So target is not part of it - target is
        static.
    """


class EnvManager(object):
    def __init__(self, env=None):
        self.env = env
        

    def load_env(self, sFilename):
        pass
    
    def save_env(self, sFilename):
        pass
    
    def regen_rail(self):
        pass

    def replace_agents(self):
        pass

    def add_agent_static(self, agent_static):
        """ Add a new agent_static
        """
        iAgent = self.number_of_agents

        if iDir is None:
            iDir = self.pick_agent_direction(rcPos, rcTarget)
        if iDir is None:
            print("Error picking agent direction at pos:", rcPos)
            return None

        self.agents_position.append(tuple(rcPos))  # ensure it's a tuple not a list
        self.agents_handles.append(max(self.agents_handles + [-1]) + 1)  # max(handles) + 1, starting at 0
        self.agents_direction.append(iDir)
        self.agents_target.append(rcPos)  # set the target to the origin initially
        self.number_of_agents += 1
        self.check_agent_lists()
        return iAgent



    def add_agent_old(self, rcPos=None, rcTarget=None, iDir=None):
        """ Add a new agent at position rcPos with target rcTarget and
            initial direction index iDir.
            Should also store this initial position etc as environment "meta-data"
            but this does not yet exist.
        """
        self.check_agent_lists()

        if rcPos is None:
            rcPos = np.random.choice(len(self.valid_positions))

        iAgent = self.number_of_agents

        if iDir is None:
            iDir = self.pick_agent_direction(rcPos, rcTarget)
        if iDir is None:
            print("Error picking agent direction at pos:", rcPos)
            return None

        self.agents_position.append(tuple(rcPos))  # ensure it's a tuple not a list
        self.agents_handles.append(max(self.agents_handles + [-1]) + 1)  # max(handles) + 1, starting at 0
        self.agents_direction.append(iDir)
        self.agents_target.append(rcPos)  # set the target to the origin initially
        self.number_of_agents += 1
        self.check_agent_lists()
        return iAgent

    def fill_valid_positions(self):
        ''' Populate the valid_positions list for the current TransitionMap.
            TODO: put this elsewhere
        '''
        self.env.valid_positions = valid_positions = []
        for r in range(self.env.height):
            for c in range(self.env.width):
                if self.env.rail.get_transitions((r, c)) > 0:
                    valid_positions.append((r, c))

    def check_agent_lists(self):
        ''' Check that the agent_handles, position and direction lists are all of length
            number_of_agents.
            (Suggest this is replaced with a single list of Agent objects :)
        '''
        for lAgents, name in zip(
                [self.env.agents_handles, self.env.agents_position, self.env.agents_direction],
                ["handles", "positions", "directions"]):
            assert self.env.number_of_agents == len(lAgents), "Inconsistent agent list:" + name

    def check_agent_locdirpath(self, iAgent):
        ''' Check that agent iAgent has a valid location and direction,
            with a path to its target.
            (Not currently used?)
        '''
        valid_movements = []
        for direction in range(4):
            position = self.env.agents_position[iAgent]
            moves = self.env.rail.get_transitions((position[0], position[1], direction))
            for move_index in range(4):
                if moves[move_index]:
                    valid_movements.append((direction, move_index))

        valid_starting_directions = []
        for m in valid_movements:
            new_position = self.env._new_position(self.env.agents_position[iAgent], m[1])
            if m[0] not in valid_starting_directions and \
                    self.env._path_exists(new_position, m[0], self.env.agents_target[iAgent]):
                valid_starting_directions.append(m[0])

        if len(valid_starting_directions) == 0:
            return False
        else:
            return True

    def pick_agent_direction(self, rcPos, rcTarget):
        """ Pick and return a valid direction index (0..3) for an agent starting at
            row,col rcPos with target rcTarget.
            Return None if no path exists.
            Picks random direction if more than one exists (uniformly).
        """
        valid_movements = []
        for direction in range(4):
            moves = self.env.rail.get_transitions((*rcPos, direction))
            for move_index in range(4):
                if moves[move_index]:
                    valid_movements.append((direction, move_index))
        # print("pos", rcPos, "targ", rcTarget, "valid movements", valid_movements)

        valid_starting_directions = []
        for m in valid_movements:
            new_position = self.env._new_position(rcPos, m[1])
            if m[0] not in valid_starting_directions and self.env._path_exists(new_position, m[0], rcTarget):
                valid_starting_directions.append(m[0])

        if len(valid_starting_directions) == 0:
            return None
        else:
            return valid_starting_directions[np.random.choice(len(valid_starting_directions), 1)[0]]

