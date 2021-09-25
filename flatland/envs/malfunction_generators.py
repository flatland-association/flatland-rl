"""Malfunction generators for rail systems"""

from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.envs import persistence


# why do we have both MalfunctionParameters and MalfunctionProcessData - they are both the same!
MalfunctionParameters = NamedTuple('MalfunctionParameters',
                                   [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])
MalfunctionProcessData = NamedTuple('MalfunctionProcessData',
                                    [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])

Malfunction = NamedTuple('Malfunction', [('num_broken_steps', int)])

# Why is the return value Optional?  We always return a Malfunction.
MalfunctionGenerator = Callable[[RandomState, bool], Malfunction]

def _malfunction_prob(rate: float) -> float:
    """
    Probability of a single agent to break. According to Poisson process with given rate
    :param rate:
    :return:
    """
    if rate <= 0:
        return 0.
    else:
        return 1 - np.exp(-rate)

class ParamMalfunctionGen(object):
    """ Preserving old behaviour of using MalfunctionParameters for constructor,
        but returning MalfunctionProcessData in get_process_data.  
        Data structure and content is the same.
    """
    def __init__(self, parameters: MalfunctionParameters):
        #self.mean_malfunction_rate = parameters.malfunction_rate
        #self.min_number_of_steps_broken = parameters.min_duration
        #self.max_number_of_steps_broken = parameters.max_duration
        self.MFP = parameters

    def generate(self, np_random: RandomState) -> Malfunction:

        if np_random.rand() < _malfunction_prob(self.MFP.malfunction_rate):
            num_broken_steps = np_random.randint(self.MFP.min_duration,
                                                    self.MFP.max_duration + 1) + 1
        else:
            num_broken_steps = 0
        return Malfunction(num_broken_steps)

    def get_process_data(self):
        return MalfunctionProcessData(*self.MFP)


class NoMalfunctionGen(ParamMalfunctionGen):
    def __init__(self):
        super().__init__(MalfunctionParameters(0,0,0))

class FileMalfunctionGen(ParamMalfunctionGen):
    def __init__(self, env_dict=None, filename=None, load_from_package=None):
        """ uses env_dict if populated, otherwise tries to load from file / package.
        """
        if env_dict is None:
             env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)

        if env_dict.get('malfunction') is not None:
            oMFP = MalfunctionParameters(*env_dict["malfunction"])
        else:
            oMFP = MalfunctionParameters(0,0,0)  # no malfunctions
        super().__init__(oMFP)


################################################################################################
# OLD / DEPRECATED generator functions below. To be removed.

def no_malfunction_generator() -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """
    Malfunction generator which generates no malfunctions

    Parameters
    ----------
    Nothing

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    print("DEPRECATED - use NoMalfunctionGen instead of no_malfunction_generator")
    # Mean malfunction in number of time steps
    mean_malfunction_rate = 0.

    # Uniform distribution parameters for malfunction duration
    min_number_of_steps_broken = 0
    max_number_of_steps_broken = 0

    def generator(np_random: RandomState = None) -> Malfunction:
        return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)



def single_malfunction_generator(earlierst_malfunction: int, malfunction_duration: int) -> Tuple[
    MalfunctionGenerator, MalfunctionProcessData]:
    """
    Malfunction generator which guarantees exactly one malfunction during an episode of an ACTIVE agent.

    Parameters
    ----------
    earlierst_malfunction: Earliest possible malfunction onset
    malfunction_duration: The duration of the single malfunction

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    # Mean malfunction in number of time steps
    mean_malfunction_rate = 0.

    # Uniform distribution parameters for malfunction duration
    min_number_of_steps_broken = 0
    max_number_of_steps_broken = 0

    # Keep track of the total number of malfunctions in the env
    global_nr_malfunctions = 0

    # Malfunction calls per agent
    malfunction_calls = dict()

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        # We use the global variable to assure only a single malfunction in the env
        nonlocal global_nr_malfunctions
        nonlocal malfunction_calls

        # Reset malfunciton generator
        if reset:
            nonlocal global_nr_malfunctions
            nonlocal malfunction_calls
            global_nr_malfunctions = 0
            malfunction_calls = dict()
            return Malfunction(0)

        # No more malfunctions if we already had one, ignore all updates
        if global_nr_malfunctions > 0:
            return Malfunction(0)

        # Update number of calls per agent
        if agent.handle in malfunction_calls:
            malfunction_calls[agent.handle] += 1
        else:
            malfunction_calls[agent.handle] = 1

        # Break an agent that is active at the time of the malfunction
        if (agent.state == TrainState.MOVING or agent.state == TrainState.STOPPED) \
            and malfunction_calls[agent.handle] >= earlierst_malfunction: #TODO : Dipam : Is this needed?
            global_nr_malfunctions += 1
            return Malfunction(malfunction_duration)
        else:
            return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)


def malfunction_from_file(filename: str, load_from_package=None) -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """

    print("DEPRECATED - use FileMalfunctionGen instead of malfunction_from_file")

    env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)
    # TODO: make this better by using namedtuple in the pickle file. See issue 282
    if env_dict.get('malfunction') is not None:
        env_dict['malfunction'] = oMPD = MalfunctionProcessData._make(env_dict['malfunction'])
    else:
        oMPD = None
    if oMPD is not None:
        # Mean malfunction in number of time steps
        mean_malfunction_rate = oMPD.malfunction_rate

        # Uniform distribution parameters for malfunction duration
        min_number_of_steps_broken = oMPD.min_duration
        max_number_of_steps_broken = oMPD.max_duration
    else:
        # Mean malfunction in number of time steps
        mean_malfunction_rate = 0.
        # Uniform distribution parameters for malfunction duration
        min_number_of_steps_broken = 0
        max_number_of_steps_broken = 0

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        """
        Generate malfunctions for agents
        Parameters
        ----------
        agent
        np_random

        Returns
        -------
        int: Number of time steps an agent is broken
        """

        # Dummy reset function as we don't implement specific seeding here
        if reset:
            return Malfunction(0)

        if agent.malfunction_handler.malfunction_down_counter < 1:
            if np_random.rand() < _malfunction_prob(mean_malfunction_rate):
                num_broken_steps = np_random.randint(min_number_of_steps_broken,
                                                     max_number_of_steps_broken + 1) + 1
                return Malfunction(num_broken_steps)
        return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)


def malfunction_from_params(parameters: MalfunctionParameters) -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """
    Utility to load malfunction from parameters

    Parameters
    ----------

    parameters : contains all the parameters of the malfunction
        malfunction_rate : float rate per timestep at which each agent malfunctions
        min_duration : int minimal duration of a failure
        max_number_of_steps_broken : int maximal duration of a failure

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    
    print("DEPRECATED - use ParamMalfunctionGen instead of malfunction_from_params")

    mean_malfunction_rate = parameters.malfunction_rate
    min_number_of_steps_broken = parameters.min_duration
    max_number_of_steps_broken = parameters.max_duration

    def generator(np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        """
        Generate malfunctions for agents
        Parameters
        ----------
        agent
        np_random

        Returns
        -------
        int: Number of time steps an agent is broken
        """

        # Dummy reset function as we don't implement specific seeding here
        if reset:
            return Malfunction(0)

        if np_random.rand() < _malfunction_prob(mean_malfunction_rate):
            num_broken_steps = np_random.randint(min_number_of_steps_broken,
                                                    max_number_of_steps_broken + 1)
            return Malfunction(num_broken_steps)
        return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)

