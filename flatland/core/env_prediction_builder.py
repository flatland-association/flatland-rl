"""
PredictionBuilder objects are objects that can be passed to environments designed for customizability.
The PredictionBuilder-derived custom classes implement 2 functions, reset(env) and get([handle]).
If predictions are not required in every step or not for all agents, then

+ `reset(env)` is called after each environment reset, to allow for pre-computing relevant data. It receives the
  (possibly newly generated) env instance, so any instantiations depending on env parameters should be done here
  rather than in `__init__`.

+ `get()` is called whenever an step has to be computed, potentially for each agent independently in \
case of multi-agent environments.
"""
from typing import Generic, TypeVar

PredictionT = TypeVar('PredictionT')
EnvT = TypeVar('EnvT')


class PredictionBuilder(Generic[EnvT, PredictionT]):
    """
    PredictionBuilder base class.

    """

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.env: EnvT = None

    def reset(self, env: EnvT):
        """
        Called after each environment reset. Receives the env instance so prediction-builder-specific
        instantiations depending on env parameters can be made here.

        Parameters
        ----------
        env : EnvT
            the (possibly newly generated) environment instance
        """
        self.env = env

    def get(self, handle: int = 0) -> PredictionT:
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
