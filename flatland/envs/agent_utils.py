import sys
import warnings
from typing import Tuple, NamedTuple, List

import numpy as np
from attr import attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.action_saver import ActionSaver
from flatland.envs.step_utils.malfunction_handler import MalfunctionHandler
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Line
from flatland.envs.timetable_utils import Timetable


class Agent(NamedTuple):
    initial_position: Tuple[int, int]
    initial_direction: Grid4TransitionsEnum
    direction: Grid4TransitionsEnum
    target: Tuple[int, int]
    moving: bool
    earliest_departure: int
    latest_arrival: int
    handle: int
    position: Tuple[int, int]
    arrival_time: int
    old_direction: Grid4TransitionsEnum
    old_position: Tuple[int, int]
    speed_counter: SpeedCounter
    action_saver: ActionSaver
    state_machine: TrainStateMachine
    malfunction_handler: MalfunctionHandler
    waypoints: List[List[Waypoint]] = None
    waypoints_earliest_departure: List[int] = None
    waypoints_latest_arrival: List[int] = None


def load_env_agent(agent_tuple: Agent):
    return EnvAgent(
        initial_configuration=(agent_tuple.initial_position, agent_tuple.initial_direction),
        current_configuration=(agent_tuple.position, agent_tuple.direction),
        old_configuration=(agent_tuple.old_position, agent_tuple.old_direction),
        target=agent_tuple.target,
        moving=agent_tuple.moving,
        earliest_departure=agent_tuple.earliest_departure,
        latest_arrival=agent_tuple.latest_arrival,
        handle=agent_tuple.handle,
        arrival_time=agent_tuple.arrival_time,
        speed_counter=agent_tuple.speed_counter,
        action_saver=agent_tuple.action_saver,
        state_machine=agent_tuple.state_machine,
        malfunction_handler=agent_tuple.malfunction_handler,
        waypoints=agent_tuple.waypoints if agent_tuple.waypoints is not None else [Waypoint(agent_tuple.initial_position, agent_tuple.initial_direction),
                                                                                   Waypoint(agent_tuple.target, None)],
        waypoints_earliest_departure=agent_tuple.waypoints_earliest_departure if agent_tuple.waypoints_earliest_departure is not None else [
            agent_tuple.earliest_departure, None],
        waypoints_latest_arrival=agent_tuple.waypoints_latest_arrival if agent_tuple.waypoints_latest_arrival is not None else [None,
                                                                                                                                agent_tuple.latest_arrival],
    )


@attrs
class EnvAgent:
    # backwards compatibility
    # TODO drop setters, keep only getters for backwards compatibility but get core free of setters
    @property
    def initial_position(self):
        return self.initial_configuration[0]

    @initial_position.setter
    def initial_position(self, value):
        self.initial_configuration = (value, self.initial_direction)

    @property
    def initial_direction(self):
        return self.initial_configuration[1]

    @initial_direction.setter
    def initial_direction(self, value):
        self.initial_configuration = (self.initial_position, value)

    @property
    def position(self):
        return self.current_configuration[0]

    @position.setter
    def position(self, value):
        self.current_configuration = (value, self.direction)

    @property
    def direction(self):
        return self.current_configuration[1]

    @direction.setter
    def direction(self, value):
        self.current_configuration = (self.position, value)

    # used in rendering
    @property
    def old_position(self):
        return self.old_configuration[0]

    @old_position.setter
    def old_position(self, value):
        self.old_configuration = (value, self.old_direction)

    @property
    def old_direction(self):
        return self.old_configuration[1]

    @old_direction.setter
    def old_direction(self, value):
        self.old_configuration = (self.old_position, value)

    # INIT FROM HERE IN _from_line()
    initial_configuration = attrib(type=Tuple[Tuple[int, int], int])
    # TODO make optional
    current_configuration = attrib(type=Tuple[Tuple[int, int], int])

    target = attrib(type=Tuple[int, int])

    moving = attrib(default=False, type=bool)

    # NEW : EnvAgent - Schedule properties
    earliest_departure = attrib(default=0, type=int)
    latest_arrival = attrib(default=sys.maxsize, type=int)

    # including initial and target, routing flexibility
    waypoints = attrib(type=List[List[Waypoint]], default=Factory(lambda: [[]]))
    # None at target, same for all in routing flexibility
    waypoints_earliest_departure = attrib(type=List[int], default=Factory(lambda: []))
    # None at initial, same for all in routing flexibility
    waypoints_latest_arrival = attrib(type=List[int], default=Factory(lambda: []))

    handle = attrib(default=None)
    # INIT TILL HERE IN _from_line()

    # Env step facelift
    speed_counter = attrib(default=Factory(lambda: SpeedCounter(1.0)), type=SpeedCounter)
    action_saver = attrib(default=Factory(lambda: ActionSaver()), type=ActionSaver)
    state_machine = attrib(default=Factory(lambda: TrainStateMachine(initial_state=TrainState.WAITING)),
                           type=TrainStateMachine)
    malfunction_handler = attrib(default=Factory(lambda: MalfunctionHandler()), type=MalfunctionHandler)

    # NEW : EnvAgent Reward Handling
    arrival_time = attrib(default=None, type=int)

    # TODO make optional instead
    old_configuration = attrib(type=Tuple[Tuple[int, int], int], default=Factory(lambda: (None, None)))

    def reset(self):
        """
        Resets the agents to their initial values of the episode. Called after ScheduleTime generation.
        """
        # TODO: set direction to None
        self.current_configuration = (None, self.initial_direction)
        self.old_configuration = (None, None)
        self.moving = False
        self.arrival_time = None

        self.malfunction_handler.reset()

        self.action_saver.clear_saved_action()
        self.speed_counter.reset()
        self.state_machine.reset()

    def to_agent(self) -> Agent:
        return Agent(initial_position=self.initial_position,
                     initial_direction=self.initial_direction,
                     direction=self.direction,
                     target=self.target,
                     moving=self.moving,
                     earliest_departure=self.earliest_departure,
                     latest_arrival=self.latest_arrival,
                     handle=self.handle,
                     position=self.position,
                     old_direction=self.old_direction,
                     old_position=self.old_position,
                     speed_counter=self.speed_counter,
                     action_saver=self.action_saver,
                     arrival_time=self.arrival_time,
                     state_machine=self.state_machine,
                     malfunction_handler=self.malfunction_handler,
                     waypoints=self.waypoints,
                     waypoints_earliest_departure=self.waypoints_earliest_departure,
                     waypoints_latest_arrival=self.waypoints_latest_arrival,
                     )

    def get_shortest_path(self, distance_map) -> List[Waypoint]:
        return distance_map.get_shortest_paths(agent_handle=self.handle)[self.handle]

    def get_travel_time_on_shortest_path(self, distance_map) -> int:
        shortest_path = self.get_shortest_path(distance_map)
        if shortest_path is not None:
            distance = len(shortest_path)
        else:
            distance = 0
        speed = self.speed_counter.max_speed
        return int(np.ceil(distance / speed))

    def get_time_remaining_until_latest_arrival(self, elapsed_steps: int) -> int:
        return self.latest_arrival - elapsed_steps

    def get_current_delay(self, elapsed_steps: int, distance_map) -> int:
        '''
        +ve if arrival time is projected before latest arrival
        -ve if arrival time is projected after latest arrival
        '''
        return self.get_time_remaining_until_latest_arrival(elapsed_steps) - \
            self.get_travel_time_on_shortest_path(distance_map)

    @classmethod
    def from_line(cls, line: Line):
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        num_agents = len(line.agent_waypoints)

        agent_list = []
        for i_agent in range(num_agents):
            speed = line.agent_speeds[i_agent] if line.agent_speeds is not None else 1.0

            agent = EnvAgent(
                initial_configuration=(line.agent_waypoints[i_agent][0][0].position, line.agent_waypoints[i_agent][0][0].direction),
                current_configuration=(line.agent_waypoints[i_agent][0][0].position, line.agent_waypoints[i_agent][0][0].direction),
                old_configuration=(None, None),
                target=line.agent_waypoints[i_agent][-1][0].position,
                waypoints=line.agent_waypoints[i_agent],
                moving=False,
                earliest_departure=None,
                latest_arrival=None,
                waypoints_earliest_departure=None,
                waypoints_latest_arrival=None,
                handle=i_agent,
                speed_counter=SpeedCounter(speed=speed))
            agent_list.append(agent)

        return agent_list

    @classmethod
    def load_legacy_static_agent(cls, static_agents_data: Tuple):
        agents = []
        for i, static_agent in enumerate(static_agents_data):
            if len(static_agent) >= 6:
                speed = static_agent[4]['speed']
                speed = 1 / (round(1 / speed))
                agent = EnvAgent(
                    initial_configuration=(static_agent[0], static_agent[1]),
                    current_configuration=(static_agent[0], static_agent[1]),
                    old_configuration=(None, None),
                    target=static_agent[2], moving=static_agent[3],
                    speed_counter=SpeedCounter(speed), handle=i,
                )
            else:
                agent = EnvAgent(
                    initial_configuration=(static_agent[0], static_agent[1]),
                    current_configuration=(static_agent[0], static_agent[1]),
                    old_configuration=(None, None),
                    target=static_agent[2],
                    moving=False,
                    speed_counter=SpeedCounter(1.0),
                    handle=i,
                )
            agents.append(agent)
        return agents

    def __str__(self):
        return (
            f"EnvAgent(\n"
            f"\thandle={self.handle},\n"
            f"\tinitial_position={self.initial_position},\n"
            f"\tinitial_direction={self.initial_direction},\n"
            f"\tposition={self.position},\n"
            f"\tdirection={self.direction if self.direction is None else Grid4TransitionsEnum(self.direction).value},\n"
            f"\ttarget={self.target},\n"
            f"\told_position={self.old_position},\n"
            f"\told_direction={self.old_direction},\n"
            f"\tearliest_departure={self.earliest_departure},\n"
            f"\tlatest_arrival={self.latest_arrival},\n"
            f"\tstate_machine={str(self.state_machine)},\n"
            f"\tmalfunction_handler={self.malfunction_handler},\n"
            f"\twaypoints={self.waypoints},\n"
            f"\twaypoints_earliest_departure={self.waypoints_earliest_departure},\n"
            f"\twaypoints_latest_arrival={self.waypoints_latest_arrival},\n"
            f")"
        )

    @property
    def state(self):
        return self.state_machine.state

    @state.setter
    def state(self, state):
        self._set_state(state)

    def _set_state(self, state):
        warnings.warn("Not recommended to set the state with this function unless completely required")
        self.state_machine.set_state(state)

    @property
    def malfunction_data(self):
        raise ValueError("agent.malunction_data is deprecated, please use agent.malfunction_hander instead")

    @property
    def speed_data(self):
        raise ValueError("agent.speed_data is deprecated, please use agent.speed_counter instead")

    @classmethod
    def apply_timetable(cls, agents: List["EnvAgent"], timetable: Timetable):
        for agent_i, agent in enumerate(agents):
            # TODO revise design: should we set state to READY_TO_DEPART if earliest_departure == 0? See `test_known_flatland_bugs.test_earliest_departure_zero_bug`.
            agent.earliest_departure = timetable.earliest_departures[agent_i][0]
            agent.latest_arrival = timetable.latest_arrivals[agent_i][-1]
            agent.waypoints_earliest_departure = timetable.earliest_departures[agent_i]
            agent.waypoints_latest_arrival = timetable.latest_arrivals[agent_i]
