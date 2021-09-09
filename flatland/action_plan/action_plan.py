import pprint
from typing import Dict, List, Optional, NamedTuple

import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_action_for_move
from flatland.envs.rail_trainrun_data_structures import Waypoint, Trainrun, TrainrunWaypoint

# ---- ActionPlan ---------------
# an action plan element represents the actions to be taken by an agent at the given time step
ActionPlanElement = NamedTuple('ActionPlanElement', [
    ('scheduled_at', int),
    ('action', RailEnvActions)
])
# an action plan gathers all the the actions to be taken by a single agent at the corresponding time steps
ActionPlan = List[ActionPlanElement]

# An action plan dict gathers all the actions for every agent identified by the dictionary key = agent_handle
ActionPlanDict = Dict[int, ActionPlan]


class ControllerFromTrainruns():
    """Takes train runs, derives the actions from it and re-acts them."""
    pp = pprint.PrettyPrinter(indent=4)

    def __init__(self,
                 env: RailEnv,
                 trainrun_dict: Dict[int, Trainrun]):

        self.env: RailEnv = env
        self.trainrun_dict: Dict[int, Trainrun] = trainrun_dict
        self.action_plan: ActionPlanDict = [self._create_action_plan_for_agent(agent_id, chosen_path)
                                            for agent_id, chosen_path in trainrun_dict.items()]

    def get_waypoint_before_or_at_step(self, agent_id: int, step: int) -> Waypoint:
        """
        Get the way point point from which the current position can be extracted.

        Parameters
        ----------
        agent_id
        step

        Returns
        -------
        WalkingElement

        """
        trainrun = self.trainrun_dict[agent_id]
        entry_time_step = trainrun[0].scheduled_at
        # the agent has no position before and at choosing to enter the grid (one tick elapses before the agent enters the grid)
        if step <= entry_time_step:
            return Waypoint(position=None, direction=self.env.agents[agent_id].initial_direction)

        # the agent has no position as soon as the target is reached
        exit_time_step = trainrun[-1].scheduled_at
        if step >= exit_time_step:
            # agent loses position as soon as target cell is reached
            return Waypoint(position=None, direction=trainrun[-1].waypoint.direction)

        waypoint = None
        for trainrun_waypoint in trainrun:
            if step < trainrun_waypoint.scheduled_at:
                return waypoint
            if step >= trainrun_waypoint.scheduled_at:
                waypoint = trainrun_waypoint.waypoint
        assert waypoint is not None
        return waypoint

    def get_action_at_step(self, agent_id: int, current_step: int) -> Optional[RailEnvActions]:
        """
        Get the current action if any is defined in the `ActionPlan`.
        ASSUMPTION we assume the env has `remove_agents_at_target=True` and `activate_agents=False`!!

        Parameters
        ----------
        agent_id
        current_step

        Returns
        -------
        WalkingElement, optional

        """
        for action_plan_element in self.action_plan[agent_id]:
            scheduled_at = action_plan_element.scheduled_at
            if scheduled_at > current_step:
                return None
            elif current_step == scheduled_at:
                return action_plan_element.action
        return None

    def act(self, current_step: int) -> Dict[int, RailEnvActions]:
        """
        Get the action dictionary to be replayed at the current step.
        Returns only action where required (no action for done agents or those not at the beginning of the cell).

        ASSUMPTION we assume the env has `remove_agents_at_target=True` and `activate_agents=False`!!

        Parameters
        ----------
        current_step: int

        Returns
        -------
        Dict[int, RailEnvActions]

        """
        action_dict = {}
        for agent_id in range(len(self.env.agents)):
            action: Optional[RailEnvActions] = self.get_action_at_step(agent_id, current_step)
            if action is not None:
                action_dict[agent_id] = action
        return action_dict

    def print_action_plan(self):
        """Pretty-prints `ActionPlanDict` of this `ControllerFromTrainruns`  to stdout."""
        self.__class__.print_action_plan_dict(self.action_plan)

    @staticmethod
    def print_action_plan_dict(action_plan: ActionPlanDict):
        """Pretty-prints `ActionPlanDict` to stdout."""
        for agent_id, plan in enumerate(action_plan):
            print("{}: ".format(agent_id))
            for step in plan:
                print("  {}".format(step))

    @staticmethod
    def assert_actions_plans_equal(expected_action_plan: ActionPlanDict, actual_action_plan: ActionPlanDict):
        assert len(expected_action_plan) == len(actual_action_plan)
        for k in range(len(expected_action_plan)):
            assert len(expected_action_plan[k]) == len(actual_action_plan[k]), \
                "len for agent {} should be the same.\n\n  expected ({}) = {}\n\n  actual ({}) = {}".format(
                    k,
                    len(expected_action_plan[k]),
                    ControllerFromTrainruns.pp.pformat(expected_action_plan[k]),
                    len(actual_action_plan[k]),
                    ControllerFromTrainruns.pp.pformat(actual_action_plan[k]))
            for i in range(len(expected_action_plan[k])):
                assert expected_action_plan[k][i] == actual_action_plan[k][i], \
                    "not the same at agent {} at step {}\n\n  expected = {}\n\n  actual = {}".format(
                        k, i,
                        ControllerFromTrainruns.pp.pformat(expected_action_plan[k][i]),
                        ControllerFromTrainruns.pp.pformat(actual_action_plan[k][i]))
        assert expected_action_plan == actual_action_plan, \
            "expected {}, found {}".format(expected_action_plan, actual_action_plan)

    def _create_action_plan_for_agent(self, agent_id, trainrun) -> ActionPlan:
        action_plan = []
        agent = self.env.agents[agent_id]
        minimum_cell_time = agent.speed_counter.max_count + 1
        for path_loop, trainrun_waypoint in enumerate(trainrun):
            trainrun_waypoint: TrainrunWaypoint = trainrun_waypoint

            position = trainrun_waypoint.waypoint.position

            if Vec2d.is_equal(agent.target, position):
                break

            next_trainrun_waypoint: TrainrunWaypoint = trainrun[path_loop + 1]
            next_position = next_trainrun_waypoint.waypoint.position

            if path_loop == 0:
                self._add_action_plan_elements_for_first_path_element_of_agent(
                    action_plan,
                    trainrun_waypoint,
                    next_trainrun_waypoint,
                    minimum_cell_time
                )
                continue

            just_before_target = Vec2d.is_equal(agent.target, next_position)

            self._add_action_plan_elements_for_current_path_element(
                action_plan,
                minimum_cell_time,
                trainrun_waypoint,
                next_trainrun_waypoint)

            # add a final element
            if just_before_target:
                self._add_action_plan_elements_for_target_at_path_element_just_before_target(
                    action_plan,
                    minimum_cell_time,
                    trainrun_waypoint,
                    next_trainrun_waypoint)
        return action_plan

    def _add_action_plan_elements_for_current_path_element(self,
                                                           action_plan: ActionPlan,
                                                           minimum_cell_time: int,
                                                           trainrun_waypoint: TrainrunWaypoint,
                                                           next_trainrun_waypoint: TrainrunWaypoint):
        scheduled_at = trainrun_waypoint.scheduled_at
        next_entry_value = next_trainrun_waypoint.scheduled_at

        position = trainrun_waypoint.waypoint.position
        direction = trainrun_waypoint.waypoint.direction
        next_position = next_trainrun_waypoint.waypoint.position
        next_direction = next_trainrun_waypoint.waypoint.direction
        next_action = get_action_for_move(position,
                                          direction,
                                          next_position,
                                          next_direction,
                                          self.env.rail)

        # if the next entry is later than minimum_cell_time, then stop here and
        # move minimum_cell_time before the exit
        # we have to do this since agents in the RailEnv are processed in the step() in the order of their handle
        if next_entry_value > scheduled_at + minimum_cell_time:
            action = ActionPlanElement(scheduled_at, RailEnvActions.STOP_MOVING)
            action_plan.append(action)

            action = ActionPlanElement(next_entry_value - minimum_cell_time, next_action)
            action_plan.append(action)
        else:
            action = ActionPlanElement(scheduled_at, next_action)
            action_plan.append(action)

    def _add_action_plan_elements_for_target_at_path_element_just_before_target(self,
                                                                                action_plan: ActionPlan,
                                                                                minimum_cell_time: int,
                                                                                trainrun_waypoint: TrainrunWaypoint,
                                                                                next_trainrun_waypoint: TrainrunWaypoint):
        scheduled_at = trainrun_waypoint.scheduled_at

        action = ActionPlanElement(scheduled_at + minimum_cell_time, RailEnvActions.STOP_MOVING)
        action_plan.append(action)

    def _add_action_plan_elements_for_first_path_element_of_agent(self,
                                                                  action_plan: ActionPlan,
                                                                  trainrun_waypoint: TrainrunWaypoint,
                                                                  next_trainrun_waypoint: TrainrunWaypoint,
                                                                  minimum_cell_time: int):
        scheduled_at = trainrun_waypoint.scheduled_at
        position = trainrun_waypoint.waypoint.position
        direction = trainrun_waypoint.waypoint.direction
        next_position = next_trainrun_waypoint.waypoint.position
        next_direction = next_trainrun_waypoint.waypoint.direction

        # add intial do nothing if we do not enter immediately, actually not necessary
        if scheduled_at > 0:
            action = ActionPlanElement(0, RailEnvActions.DO_NOTHING)
            action_plan.append(action)

        # add action to enter the grid
        action = ActionPlanElement(scheduled_at, RailEnvActions.MOVE_FORWARD)
        action_plan.append(action)

        next_action = get_action_for_move(position,
                                          direction,
                                          next_position,
                                          next_direction,
                                          self.env.rail)

        # if the agent is blocked in the cell, we have to call stop upon entering!
        if next_trainrun_waypoint.scheduled_at > scheduled_at + 1 + minimum_cell_time:
            action = ActionPlanElement(scheduled_at + 1, RailEnvActions.STOP_MOVING)
            action_plan.append(action)

        # execute the action exactly minimum_cell_time before the entry into the next cell
        action = ActionPlanElement(next_trainrun_waypoint.scheduled_at - minimum_cell_time, next_action)
        action_plan.append(action)
