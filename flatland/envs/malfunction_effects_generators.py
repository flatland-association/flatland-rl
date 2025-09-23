import importlib
from typing import Callable, List

from flatland.core.effects_generator import EffectsGenerator
from flatland.core.grid.grid_utils import IntVector2D
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs.agent_utils import EnvAgent
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


def on_map_state_condition(env_agent: EnvAgent, elapsed_steps: int) -> bool:
    return env_agent.state_machine.state.is_on_map_state()


class ConditionalMalfunctionEffectsGenerator(EffectsGenerator["RailEnv"]):
    def __init__(self,
                 malfunction_rate: float = None,
                 min_duration: float = None,
                 max_duration: float = None,
                 earliest_malfunction: int = None,
                 max_num_malfunctions: int = None,
                 condition: MalfunctionCondition = None,
                 condition_pkg: str = None,
                 condition_cls: str = None,
                 ):
        """
        Generate agent malfunctions conditionally with conditional rate and duration.

        Parameters
        ----------
        malfunction_rate : float
            Poisson process with given rate.
        min_duration : int
            If malfunction, duration uniformly in [min_duration,max_duration].
        max_duration : int
            If malfunction, duration uniformly in [min_duration,max_duration].
        earliest_malfunction : int
            Defaults to `None`.
        max_num_malfunctions : int
            Defaults to `None`.
        condition : MalfunctionCondition
            Additional condition. Defaults to None.
        condition_pkg : str
            Additional condition to be created instead of instance via `condition`. Defaults to None.
        condition_cls : str
            Additional condition to be created instead of instance via `condition`. Defaults to None.
        """
        super().__init__()

        self._malfunction_rate = float(malfunction_rate)
        self._min_duration = int(min_duration)
        self._max_duration = int(max_duration)

        self._malfunction_generator = mal_gen.ParamMalfunctionGen(
            mal_gen.MalfunctionParameters(malfunction_rate=self._malfunction_rate, min_duration=self._min_duration, max_duration=self._max_duration)
        )
        self._earliest_condition = int(earliest_malfunction) if earliest_malfunction is not None else None
        self._max_num_malfunctions = int(max_num_malfunctions) if max_num_malfunctions is not None else None
        self._num_malfunctions = 0
        self._condition = condition
        if condition_pkg is not None and condition_cls is not None:
            module = importlib.import_module(condition_pkg)
            self._condition = getattr(module, condition_cls)

    def on_episode_step_start(self, env: "RailEnv", *args, **kwargs) -> "RailEnv":
        if self._earliest_condition is not None and env._elapsed_steps < self._earliest_condition:
            return env
        if self._max_num_malfunctions is not None and self._num_malfunctions >= self._max_num_malfunctions:
            return env
        for agent in env.agents:
            if self._condition is None or self._condition(agent, env._elapsed_steps):
                in_malfunction_before = agent.malfunction_handler.in_malfunction
                agent.malfunction_handler.generate_malfunction(self._malfunction_generator, env.np_random)
                in_malfunction_after = agent.malfunction_handler.in_malfunction
                if in_malfunction_after and not in_malfunction_before:
                    self._num_malfunctions += 1
                    if self._max_num_malfunctions is not None and self._num_malfunctions >= self._max_num_malfunctions:
                        return env
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
        return ((agent.position in {w.position for ws in agent.waypoints[1:-1] for w in ws})
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
