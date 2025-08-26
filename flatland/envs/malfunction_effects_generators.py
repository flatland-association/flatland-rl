from typing import Callable, List

from flatland.core.effects_generator import EffectsGenerator
from flatland.core.grid.grid_utils import IntVector2D
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs.step_utils.states import TrainState


class MalfunctionEffectsGenerator(EffectsGenerator["RailEnv"]):

    def __init__(self, malfunction_generator):
        super().__init__()
        self.malfunction_generator = malfunction_generator

    def on_episode_step_start(self, env: "RailEnv", *args, **kwargs) -> "RailEnv":
        for agent in env.agents:
            agent.malfunction_handler.generate_malfunction(self.malfunction_generator, env.np_random)
        return env


MalfunctionCondition = Callable[["EnvAgent", int], bool]


class ConditionalMalfunctionEffectsGenerator(EffectsGenerator["RailEnv"]):
    def __init__(self,
                 malfunction_rate: float = None,
                 min_duration: float = None,
                 max_duration: float = None,
                 condition: MalfunctionCondition = None,
                 ):
        """
        Generate agent malfunctions conditionally with conditional rate and duration.

        Parameters
        ----------
        malfunction_rate : int
            Poisson process with given rate.
        min_duration : int
            If malfunction, duration uniformly in [min_duration,max_duration].
        max_duration : int
            If malfunction, duration uniformly in [min_duration,max_duration].
        """
        super().__init__()

        self._malfunction_rate = malfunction_rate if malfunction_rate is not None else 0
        self._min_duration = min_duration if min_duration is not None else 1
        self._max_duration = max_duration if max_duration is not None else 1

        self._malfunction_generator = mal_gen.ParamMalfunctionGen(
            mal_gen.MalfunctionParameters(malfunction_rate=self._malfunction_rate, min_duration=self._min_duration, max_duration=self._max_duration)
        )
        self._condition = condition

    def on_episode_step_start(self, env: "RailEnv", *args, **kwargs) -> "RailEnv":
        if self._condition is None:
            return env
        for agent in env.agents:
            if self._condition(agent, env._elapsed_steps):
                agent.malfunction_handler.generate_malfunction(self._malfunction_generator, env.np_random)
        return env


def make_multi_malfunction_condition(conditions: List[MalfunctionCondition]) -> MalfunctionCondition:
    """
    Disjunctively wrap multiple MalfunctionCondition into one.

    Parameters
    ----------
    conditions : List[MalfunctionCondition]
        list of disjunctive conditions

    Returns
    -------
    MalfunctionCondition

    """

    def _condition(agent: "EnvAgent", elapsed_steps: int):
        for c in conditions:
            if c(agent, elapsed_steps):
                return True
        return False

    return _condition


def condition_stopped_intermediate_and_range(start_step_incl: int, end_step_excl: int) -> MalfunctionCondition:
    """
    Malfunction condition: stopped at an intermediate waypoint and in range of timesteps.


    Parameters
    ----------
    start_step_incl : int
        start step of positive range (incl.)
    end_step_excl : int
        end step of positive range (excl.)

    Returns
    -------
    MalfunctionCondition

    """

    def _condition(agent: "EnvAgent", elapsed_steps: int):
        return ((agent.position in {w.position for w in agent.waypoints[1:-1]})
                and agent.state_machine.state == TrainState.STOPPED and elapsed_steps >= start_step_incl and elapsed_steps < end_step_excl)

    return _condition


def condition_stopped_cells_and_range(start_step_incl: int, end_step_excl: int, cells: List[IntVector2D]) -> MalfunctionCondition:
    """
    Malfunction condition: stopped on any given cell and during range of timesteps.


    Parameters
    ----------
    start_step_incl : int
        start step of positive range (incl.)
    end_step_excl : int
        end step of positive range (excl.)

    Returns
    -------
    MalfunctionCondition

    """

    def _condition(agent: "EnvAgent", elapsed_steps: int):
        return agent.position in cells and agent.state_machine.state == TrainState.STOPPED and elapsed_steps >= start_step_incl and elapsed_steps < end_step_excl

    return _condition
