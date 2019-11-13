import pprint
from typing import Dict, List, Optional, NamedTuple

import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_action_for_move
from flatland.envs.rail_train_run_data_structures import WayPoint, TrainRun, TrainRunWayPoint
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

# ---- ActionPlan ---------------
# an action plan element represents the actions to be taken by an agent at the given time step
ActionPlanElement = NamedTuple('ActionPlanElement', [
    ('scheduled_at', int),
    ('action', RailEnvActions)
])
# an action plan gathers all the the actions to be taken by a single agent at the corresponding time steps
ActionPlan = List[ActionPlanElement]

# An action plan dict gathers all the actions for every agent identified by the dictionary key
ActionPlanDict = Dict[int, ActionPlan]


class DeterministicController():
    pp = pprint.PrettyPrinter(indent=4)

    def __init__(self,
                 env: RailEnv,
                 train_run_dict: Dict[int, TrainRun]):

        self.env = env
        self.train_run_dict: Dict[int, TrainRun] = train_run_dict
        self.action_plan = [self._create_action_plan_for_agent(agent_id, chosen_path)
                            for agent_id, chosen_path in train_run_dict.items()]

    def get_way_point_before_or_at_step(self, agent_id: int, step: int) -> WayPoint:
        """
        Get the walking element from which the current position can be extracted.

        Parameters
        ----------
        agent_id
        step

        Returns
        -------
        WalkingElement

        """
        train_run = self.train_run_dict[agent_id]
        entry_time_step = train_run[0].scheduled_at
        # the agent has no position before and at choosing to enter the grid (one tick elapses before the agent enters the grid)
        if step <= entry_time_step:
            return WayPoint(position=None, direction=self.env.agents[agent_id].initial_direction)

        # the agent has no position as soon as the target is reached
        exit_time_step = train_run[-1].scheduled_at
        if step >= exit_time_step:
            # agent loses position as soon as target cell is reached
            return WayPoint(position=None, direction=train_run[-1].way_point.direction)

        way_point = None
        for train_run_way_point in train_run:
            if step < train_run_way_point.scheduled_at:
                return way_point
            if step >= train_run_way_point.scheduled_at:
                way_point = train_run_way_point.way_point
        assert way_point is not None
        return way_point

    def get_action_at_step(self, agent_id: int, current_step: int) -> Optional[RailEnvActions]:
        """
        Get the current action if any is defined in the `ActionPlan`.

        Parameters
        ----------
        agent_id
        current_step

        Returns
        -------
        WalkingElement, optional

        """
        for action_plan_element in self.action_plan[agent_id]:
            action_plan_element: ActionPlanElement = action_plan_element
            scheduled_at = action_plan_element.scheduled_at
            if scheduled_at > current_step:
                return None
            elif current_step == scheduled_at:
                return action_plan_element.action
        return None

    def act(self, current_step: int) -> Dict[int, RailEnvActions]:
        """
        Get the action dictionary to be replayed at the current step.

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
        for agent_id, plan in enumerate(self.action_plan):
            print("{}: ".format(agent_id))
            for step in plan:
                print("  {}".format(step))

    @staticmethod
    def compare_action_plans(expected_action_plan: ActionPlanDict, actual_action_plan: ActionPlanDict):
        assert len(expected_action_plan) == len(actual_action_plan)
        for k in range(len(expected_action_plan)):
            assert len(expected_action_plan[k]) == len(actual_action_plan[k]), \
                "len for agent {} should be the same.\n\n  expected ({}) = {}\n\n  actual ({}) = {}".format(
                    k,
                    len(expected_action_plan[k]),
                    DeterministicController.pp.pformat(expected_action_plan[k]),
                    len(actual_action_plan[k]),
                    DeterministicController.pp.pformat(actual_action_plan[k]))
            for i in range(len(expected_action_plan[k])):
                assert expected_action_plan[k][i] == actual_action_plan[k][i], \
                    "not the same at agent {} at step {}\n\n  expected = {}\n\n  actual = {}".format(
                        k, i,
                        DeterministicController.pp.pformat(expected_action_plan[k][i]),
                        DeterministicController.pp.pformat(actual_action_plan[k][i]))
        assert expected_action_plan == actual_action_plan, \
            "expected {}, found {}".format(expected_action_plan, actual_action_plan)

    def _create_action_plan_for_agent(self, agent_id, agent_path_new) -> ActionPlan:
        action_plan = []
        agent = self.env.agents[agent_id]
        minimum_cell_time = int(np.ceil(1.0 / agent.speed_data['speed']))
        for path_loop, train_run_way_point in enumerate(agent_path_new):
            train_run_way_point: TrainRunWayPoint = train_run_way_point

            position = train_run_way_point.way_point.position

            if Vec2d.is_equal(agent.target, position):
                break

            next_train_run_way_point: TrainRunWayPoint = agent_path_new[path_loop + 1]
            next_position = next_train_run_way_point.way_point.position

            if path_loop == 0:
                self._add_action_plan_elements_for_first_path_element_of_agent(
                    action_plan,
                    train_run_way_point,
                    next_train_run_way_point)
                continue

            just_before_target = Vec2d.is_equal(agent.target, next_position)

            self._add_action_plan_elements_for_current_path_element(
                action_plan,
                minimum_cell_time,
                train_run_way_point,
                next_train_run_way_point)

            # add a final element
            if just_before_target:
                self._add_action_plan_elements_for_target_at_path_element_just_before_target(
                    action_plan,
                    minimum_cell_time,
                    train_run_way_point,
                    next_train_run_way_point)
        return action_plan

    def _add_action_plan_elements_for_current_path_element(self,
                                                           action_plan: ActionPlan,
                                                           minimum_cell_time: int,
                                                           train_run_way_point: TrainRunWayPoint,
                                                           next_train_run_way_point: TrainRunWayPoint):
        scheduled_at = train_run_way_point.scheduled_at
        next_entry_value = next_train_run_way_point.scheduled_at

        position = train_run_way_point.way_point.position
        direction = train_run_way_point.way_point.direction
        next_position = next_train_run_way_point.way_point.position
        next_direction = next_train_run_way_point.way_point.direction
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
                                                                                train_run_way_point: TrainRunWayPoint,
                                                                                next_train_run_way_point: TrainRunWayPoint):
        scheduled_at = train_run_way_point.scheduled_at

        action = ActionPlanElement(scheduled_at + minimum_cell_time, RailEnvActions.STOP_MOVING)
        action_plan.append(action)

    def _add_action_plan_elements_for_first_path_element_of_agent(self,
                                                                  action_plan: ActionPlan,
                                                                  train_run_way_point: TrainRunWayPoint,
                                                                  next_train_run_way_point: TrainRunWayPoint):
        scheduled_at = train_run_way_point.scheduled_at
        position = train_run_way_point.way_point.position
        direction = train_run_way_point.way_point.direction
        next_position = next_train_run_way_point.way_point.position
        next_direction = next_train_run_way_point.way_point.direction

        # add intial do nothing if we do not enter immediately
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

        # now, we have a position need to perform the action
        action = ActionPlanElement(scheduled_at + 1, next_action)
        action_plan.append(action)


class DeterministicControllerReplayer():
    """Allows to verify a `DeterministicController` by replaying it against a FLATland env without malfunction."""

    @staticmethod
    def replay_verify(max_episode_steps: int, ctl: DeterministicController, env: RailEnv, rendering: bool):
        """Replays this deterministic `ActionPlan` and verifies whether it is feasible."""
        if rendering:
            renderer = RenderTool(env, gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                                  show_debug=True,
                                  clear_debug_text=True,
                                  screen_height=1000,
                                  screen_width=1000)
            renderer.render_env(show=True, show_observations=False, show_predictions=False)
        i = 0
        while not env.dones['__all__'] and i <= max_episode_steps:
            for agent_id, agent in enumerate(env.agents):
                way_point: WayPoint = ctl.get_way_point_before_or_at_step(agent_id, i)
                assert agent.position == way_point.position, \
                    "before {}, agent {} at {}, expected {}".format(i, agent_id, agent.position,
                                                                    way_point.position)
            actions = ctl.act(i)
            print("actions for {}: {}".format(i, actions))

            obs, all_rewards, done, _ = env.step(actions)

            if rendering:
                renderer.render_env(show=True, show_observations=False, show_predictions=False)

            i += 1
