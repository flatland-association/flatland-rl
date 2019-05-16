"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ Reset() is called after each environment reset, to allow for pre-computing relevant data.

+ Get() is called whenever an observation has to be computed, potentially for each agent independently in
case of multi-agent environments.
"""


class ObservationBuilder:
    """
    ObservationBuilder base class.
    """

    def __init__(self):
        pass

    def _set_env(self, env):
        self.env = env

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

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
