import abc

from flatland.envs.rail_env import RailEnv


class WrappedRailEnv(abc.ABC):
    """
    An API for ml envs wrapping RailEnv.
    """

    @abc.abstractmethod
    def wrap(self) -> RailEnv:
        raise NotImplementedError()
