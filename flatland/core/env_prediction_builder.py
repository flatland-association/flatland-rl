"""
PredictionBuilder objects are objects that can be passed to environments designed for customizability.
The PredictionBuilder-derived custom classes implement 2 functions, reset() and get([handle]).
If predictions are not required in every step or not for all agents, then

+ Reset() is called after each environment reset, to allow for pre-computing relevant data.

+ Get() is called whenever an step has to be computed, potentially for each agent independently in
case of multi-agent environments.
"""


class PredictionBuilder:
    """
    PredictionBuilder base class.

    """

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth

    def _set_env(self, env):
        self.env = env

    def reset(self):
        """
        Called after each environment reset.
        """
        pass

    def get(self, custom_args=None, handle=0):
        """
        Called whenever get_many in the observation build is called.

        Parameters
        -------
        custom_args: dict
            Implementation-dependent custom arguments, see the sub-classes.

        handle : int (optional)
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            A prediction structure, specific to the corresponding environment.
        """
        raise NotImplementedError()
