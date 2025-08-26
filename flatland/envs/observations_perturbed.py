from typing import Optional, List, Dict

import numpy as np
from numpy.random import RandomState

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder, AgentHandle
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.step_utils.malfunction_handler import MalfunctionHandler


def perturbation_tree_observation_builder_wrapper(
    builder: TreeObsForRailEnv, np_random: RandomState, perturbation_rate: float = None, min_duration: float = None, max_duration: float = None, blank=-np.inf,
) -> ObservationBuilder[Node]:
    """
    Make some trains blind for some time according to Poisson process.

    Parameters
    ----------
    builder : TreeObsForRailEnv
        the wrapped observation builder
    np_random : RandomState
    perturbation_rate : int
        Poisson process with given rate.
    min_duration : int
        If perturbed, duration uniformly in [min_duration,max_duration].
    max_duration : int
        If perturbed, duration uniformly in [min_duration,max_duration].
    blank : float
        value to insert for perturbed trains.

    Returns
    -------
    Observations with some trains not seeing anything..
    """

    class _PerturbedTreeObsForRailEnv(ObservationBuilder[Node]):
        def __init__(self, builder: TreeObsForRailEnv):
            super().__init__()
            self._malfunction_rate = perturbation_rate if perturbation_rate is not None else 0
            self._min_duration = min_duration if min_duration is not None else 1
            self._max_duration = max_duration if max_duration is not None else 1
            self._builder = builder
            self._np_random = np_random
            self._malfunction_handlers: Dict[AgentHandle, MalfunctionHandler] = {}
            self._malfunction_generator = ParamMalfunctionGen(
                MalfunctionParameters(malfunction_rate=self._malfunction_rate, min_duration=self._min_duration, max_duration=self._max_duration)
            )
            self._blank = blank

        def set_env(self, env: Environment):
            super().set_env(env)
            self._builder.set_env(env)

        def reset(self):
            self._builder.reset()
            self._malfunction_handlers = {
                i: MalfunctionHandler() for i in self.env.get_agent_handles()
            }

        def get_many(self, handles: Optional[List[AgentHandle]] = None) -> Dict[AgentHandle, Node]:
            obs = self._builder.get_many(handles)
            for handle in self.env.get_agent_handles():
                self._malfunction_handlers[handle].generate_malfunction(self._malfunction_generator, self._np_random)
                if self._malfunction_handlers[handle].in_malfunction:
                    # agent invisible for all others
                    obs[handle] = Node(dist_own_target_encountered=self._blank,
                                       dist_other_target_encountered=self._blank,
                                       dist_other_agent_encountered=self._blank,
                                       dist_potential_conflict=self._blank,
                                       dist_unusable_switch=self._blank,
                                       dist_to_next_branch=self._blank,
                                       dist_min_to_target=self._blank,
                                       num_agents_same_direction=self._blank,
                                       num_agents_opposite_direction=self._blank,
                                       num_agents_malfunctioning=self._blank,
                                       speed_min_fractional=self._blank,
                                       num_agents_ready_to_depart=self._blank,
                                       childs={})
            return obs

    return _PerturbedTreeObsForRailEnv(builder)
