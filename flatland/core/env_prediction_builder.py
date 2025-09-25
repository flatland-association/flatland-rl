"""
PredictionBuilder objects are objects that can be passed to environments designed for customizability.
The PredictionBuilder-derived custom classes implement 2 functions, reset() and get([handle]).
If predictions are not required in every step or not for all agents, then

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an step has to be computed, potentially for each agent independently in \
case of multi-agent environments.
"""
from typing import Generic, TypeVar

PredictionType = TypeVar('PredictionType')
EnvType = TypeVar('EnvType')


class PredictionBuilder(Generic[EnvType, PredictionType]):
    """
    PredictionBuilder base class.

    """

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.env: EnvType = None

    def set_env(self, env: EnvType):
        self.env = env

    def reset(self):
        """
        Called after each environment reset.
        """
        pass

    def get(self, handle: int = 0) -> PredictionType:
        """
        Called whenever get_many in the observation builder is called.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            A prediction structure, specific to the corresponding environment.
        """
        raise NotImplementedError()
