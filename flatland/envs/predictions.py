"""
Collection of environment-specific PredictionBuilder.
"""

from flatland.core.env_prediction_builder import PredictionBuilder


class DummyPredictorForRailEnv(PredictionBuilder):
    """
    DummyPredictorForRailEnv object.

    This object returns predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def get(self, handle=0):
        return {}
