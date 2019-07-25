"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ Reset() is called after each environment reset, to allow for pre-computing relevant data.

+ Get() is called whenever an observation has to be computed, potentially for each agent independently in
case of multi-agent environments.
"""
import numpy as np


class ObservationBuilder:
    """
    ObservationBuilder base class.

    Derived objects must implement and `observation_space' attribute as a tuple with the dimensions of the returned
    observations.
    """

    def __init__(self):
        self.observation_space = ()

    def _set_env(self, env):
        self.env = env

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get_many(self, handles=[]):
        """
        Called whenever an observation has to be computed for the `env' environment, for each agent with handle
        in the `handles' list.

        Parameters
        -------
        handles : list of handles (optional)
            List with the handles of the agents for which to compute the observation vector.

        Returns
        -------
        function
            A dictionary of observation structures, specific to the corresponding environment, with handles from
            `handles' as keys.
        """
        observations = {}
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle=0):
        """
        Called whenever an observation has to be computed for the `env' environment, possibly
        for each agent independently (agent id `handle').

        Parameters
        -------
        handle : int (optional)
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            An observation structure, specific to the corresponding environment.
        """
        raise NotImplementedError()

    def _get_one_hot_for_agent_direction(self, agent):
        """Retuns the agent's direction to one-hot encoding."""
        direction = np.zeros(4)
        direction[agent.direction] = 1
        return direction


class DummyObservationBuilder(ObservationBuilder):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        self.observation_space = ()

    def _set_env(self, env):
        self.env = env

    def reset(self):
        pass

    def get_many(self, handles=[]):
        return True

    def get(self, handle=0):
        return True
