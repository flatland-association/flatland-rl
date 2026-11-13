from dataclasses import dataclass
from typing import List, Optional, Tuple

from flatland.core.env_observation_builder import AgentHandle, ObservationBuilder
from flatland.envs.agent_utils import EntryPointT
from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_policies import ShortestPathObservation
from flatland.envs.travelwise.timetable_predictor import ProjectedEntryPoint, TimetablePredictor, UnilateralTimetablePredictor


@dataclass(frozen=True)
class VehiclePositionsAndPlans:
    """
    The positions and plans of all vehicles, by vehicle handle: one immutable object, referenced by every vehicle
    observation, not copied per vehicle. A vehicle's position is its current entry point (`None` off the map), its plan
    its predicted remaining route (see `TimetablePredictor`) - empty after `reset()` and once it has arrived.
    """
    positions: Tuple[Optional[EntryPointT], ...]
    plans: Tuple[Tuple[ProjectedEntryPoint, ...], ...]


@dataclass(frozen=True)
class VehicleObservation(ShortestPathObservation):
    """ A vehicle's `ShortestPathObservation`, which `ShortestPathPolicy` acts on, and the positions and plans of all vehicles. """
    vehicles: VehiclePositionsAndPlans


class VehicleObservationBuilder(ObservationBuilder[AbstractRailEnv, VehicleObservation]):
    """
    The rail env's observation builder in `PassengerEnv`: builds each vehicle's `VehicleObservation`. It owns the
    vehicles' predictor (as `TreeObsForRailEnv` owns its predictor), default `UnilateralTimetablePredictor`, and calls
    it once per step in `get_many()`. After `reset()`, no vehicle has a position or a plan yet.
    """

    def __init__(self, predictor: Optional[TimetablePredictor] = None):
        super().__init__()
        self.predictor = predictor if predictor is not None else UnilateralTimetablePredictor()
        self._vehicles: Optional[VehiclePositionsAndPlans] = None

    def reset(self, env: AbstractRailEnv):
        super().reset(env)
        self.predictor.reset(env)
        self._vehicles = None

    def get_many(self, handles: Optional[List[AgentHandle]] = None):
        plans = self.predictor.get() if self.env._elapsed_steps > 0 else {}
        self._vehicles = VehiclePositionsAndPlans(positions=tuple(agent.current_entry_point for agent in self.env.agents),
                                                  plans=tuple(tuple(plans.get(agent.handle, ())) for agent in self.env.agents))
        return super().get_many(handles)

    def get(self, handle: AgentHandle = 0) -> VehicleObservation:
        return VehicleObservation(**vars(ShortestPathObservation.from_env(self.env, handle)), vehicles=self._vehicles)
