import pprint
from typing import Dict, List, Optional, NamedTuple

import numpy as np
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import WalkingElement, get_action_for_move
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


#---- Input Data Structures (graph representation)  ---------------------------------------------
#  A cell pin represents the one of the four pins in which the cell at row,column may be entered.
CellPin = NamedTuple('CellPin', [('r', int), ('c', int), ('d', int)])

# A path schedule element represents the entry time of agent at a cell pin.
PathScheduleElement = NamedTuple('PathScheduleElement', [
    ('scheduled_at', int),
    ('cell_pin', CellPin)
])
# A path schedule is the list of an agent's cell pin entries
PathSchedule = List[PathScheduleElement]


#---- Output Data Structures (FLATland representation) ---------------------------------------------
# An action plan element represents the actions to be taken by an agent at deterministic time steps
#  plus the position before the action
ActionPlanElement = NamedTuple('ActionPlanElement', [
    ('scheduled_at', int),
    ('walking_element', WalkingElement)
])
# An action plan deterministically represents all the actions to be taken by an agent
#  plus its position before the actions are taken
ActionPlan = Dict[int, List[ActionPlanElement]]



class ActionPlanReplayer():
    """Allows to deduce an `ActionPlan` from the agents' `PathSchedule` and
    to be replayed/verified in a FLATland env without malfunction."""

    pp = pprint.PrettyPrinter(indent=4)

    def __init__(self,
                 env: RailEnv,
                 chosen_path_dict: Dict[int, PathSchedule]):

        self.env = env
        self.action_plan = [[] for _ in range(self.env.get_num_agents())]

        for agent_id, chosen_path in chosen_path_dict.items():
            self._add_aggent_to_action_plan(self.action_plan, agent_id, chosen_path)

    def get_walking_element_before_or_at_step(self, agent_id: int, step: int) -> WalkingElement:
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
        walking_element = None
        for action in self.action_plan[agent_id]:
            if step < action.scheduled_at:
                return walking_element
            if step >= action.scheduled_at:
                walking_element = action.walking_element
        assert walking_element is not None
        return walking_element

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
        for action_plan_step in self.action_plan[agent_id]:
            action_plan_step: ActionPlanElement = action_plan_step
            scheduled_at = action_plan_step.scheduled_at
            walking_element: WalkingElement = action_plan_step.walking_element
            if scheduled_at > current_step:
                return None
            elif np.isclose(current_step, scheduled_at):
                return walking_element.next_action
        return None

    def get_action_dict_for_step_replay(self, current_step: int) -> Dict[int, RailEnvActions]:
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
        for agent_id, agent in enumerate(self.env.agents):
            action: Optional[RailEnvActions] = self.get_action_at_step(agent_id, current_step)
            if action is not None:
                action_dict[agent_id] = action
        return action_dict

    def replay_verify(self, MAX_EPISODE_STEPS: int, env: RailEnv, rendering: bool):
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
        while not env.dones['__all__'] and i <= MAX_EPISODE_STEPS:
            for agent_id, agent in enumerate(env.agents):
                walking_element: WalkingElement = self.get_walking_element_before_or_at_step(agent_id, i)
                assert agent.position == walking_element.position, \
                    "before {}, agent {} at {}, expected {}".format(i, agent_id, agent.position,
                                                                    walking_element.position)
            actions = self.get_action_dict_for_step_replay(i)
            print("actions for {}: {}".format(i, actions))

            obs, all_rewards, done, _ = env.step(actions)

            if rendering:
                renderer.render_env(show=True, show_observations=False, show_predictions=False)

            i += 1

    def print_action_plan(self):
        for agent_id, plan in enumerate(self.action_plan):
            print("{}: ".format(agent_id))
            for step in plan:
                print("  {}".format(step))

    @staticmethod
    def compare_action_plans(expected_action_plan: ActionPlan, actual_action_plan: ActionPlan):
        assert len(expected_action_plan) == len(actual_action_plan)
        for k in range(len(expected_action_plan)):
            assert len(expected_action_plan[k]) == len(actual_action_plan[k]), \
                "len for agent {} should be the same.\n\n  expected ({}) = {}\n\n  actual ({}) = {}".format(
                    k,
                    len(expected_action_plan[k]),
                    ActionPlanReplayer.pp.pformat(expected_action_plan[k]),
                    len(actual_action_plan[k]),
                    ActionPlanReplayer.pp.pformat(actual_action_plan[k]))
            for i in range(len(expected_action_plan[k])):
                assert expected_action_plan[k][i] == actual_action_plan[k][i], \
                    "not the same at agent {} at step {}\n\n  expected = {}\n\n  actual = {}".format(
                        k, i,
                        ActionPlanReplayer.pp.pformat(expected_action_plan[k][i]),
                        ActionPlanReplayer.pp.pformat(actual_action_plan[k][i]))

    def _add_aggent_to_action_plan(self, action_plan, agent_id, agent_path_new):
        agent = self.env.agents[agent_id]
        minimum_cell_time = int(np.ceil(1.0 / agent.speed_data['speed']))
        for path_loop, path_schedule_element in enumerate(agent_path_new):
            path_schedule_element: PathScheduleElement = path_schedule_element

            position = (path_schedule_element.cell_pin.r, path_schedule_element.cell_pin.c)

            if Vec2d.is_equal(agent.target, position):
                break

            next_path_schedule_element: PathScheduleElement = agent_path_new[path_loop + 1]
            next_position = (next_path_schedule_element.cell_pin.r, next_path_schedule_element.cell_pin.c)

            if path_loop == 0:
                self._create_action_plan_for_first_path_element_of_agent(
                    action_plan,
                    agent_id,
                    path_schedule_element,
                    next_path_schedule_element)
                continue

            just_before_target = Vec2d.is_equal(agent.target, next_position)

            self._create_action_plan_for_current_path_element(
                action_plan,
                agent_id,
                minimum_cell_time,
                path_schedule_element,
                next_path_schedule_element)

            # add a final element
            if just_before_target:
                self._create_action_plan_for_target_at_path_element_just_before_target(
                    action_plan,
                    agent_id,
                    minimum_cell_time,
                    path_schedule_element,
                    next_path_schedule_element)

    def _create_action_plan_for_current_path_element(self,
                                                     action_plan: ActionPlan,
                                                     agent_id: int,
                                                     minimum_cell_time: int,
                                                     path_schedule_element: PathScheduleElement,
                                                     next_path_schedule_element: PathScheduleElement):
        scheduled_at = path_schedule_element.scheduled_at
        next_entry_value = next_path_schedule_element.scheduled_at

        position = (path_schedule_element.cell_pin.r, path_schedule_element.cell_pin.c)
        direction = path_schedule_element.cell_pin.d
        next_position = next_path_schedule_element.cell_pin.r, next_path_schedule_element.cell_pin.c
        next_direction = next_path_schedule_element.cell_pin.d
        next_action = get_action_for_move(position,
                                          direction,
                                          next_position,
                                          next_direction,
                                          self.env.rail)

        walking_element = WalkingElement(position, direction, next_action)

        # if the next entry is later than minimum_cell_time, then stop here and
        # move minimum_cell_time before the exit
        # we have to do this since agents in the RailEnv are processed in the step() in the order of their handle
        if next_entry_value > scheduled_at + minimum_cell_time:
            action = ActionPlanElement(scheduled_at,
                                       WalkingElement(
                                           position=position,
                                           direction=direction,
                                           next_action=RailEnvActions.STOP_MOVING))
            action_plan[agent_id].append(action)

            action = ActionPlanElement(next_entry_value - minimum_cell_time, walking_element)
            action_plan[agent_id].append(action)
        else:
            action = ActionPlanElement(scheduled_at, walking_element)
            action_plan[agent_id].append(action)

    def _create_action_plan_for_target_at_path_element_just_before_target(self,
                                                                          action_plan: ActionPlan,
                                                                          agent_id: int,
                                                                          minimum_cell_time: int,
                                                                          path_schedule_element: PathScheduleElement,
                                                                          next_path_schedule_element: PathScheduleElement):
        scheduled_at = path_schedule_element.scheduled_at
        next_path_schedule_element.cell_pin

        action = ActionPlanElement(scheduled_at + minimum_cell_time,
                                   WalkingElement(
                                       position=None,
                                       direction=next_path_schedule_element.cell_pin.d,
                                       next_action=RailEnvActions.STOP_MOVING))
        action_plan[agent_id].append(action)

    def _create_action_plan_for_first_path_element_of_agent(self,
                                                            action_plan: ActionPlan,
                                                            agent_id: int,
                                                            path_schedule_element: PathScheduleElement,
                                                            next_path_schedule_element: PathScheduleElement):
        scheduled_at = path_schedule_element.scheduled_at
        position = (path_schedule_element.cell_pin.r, path_schedule_element.cell_pin.c)
        direction = path_schedule_element.cell_pin.d
        next_position = next_path_schedule_element.cell_pin.r, next_path_schedule_element.cell_pin.c
        next_direction = next_path_schedule_element.cell_pin.d

        # add intial do nothing if we do not enter immediately
        if scheduled_at > 0:
            action = ActionPlanElement(0,
                                       WalkingElement(
                                           position=None,
                                           direction=direction,
                                           next_action=RailEnvActions.DO_NOTHING))
            action_plan[agent_id].append(action)
        # add action to enter the grid
        action = ActionPlanElement(scheduled_at,
                                   WalkingElement(
                                       position=None,
                                       direction=direction,
                                       next_action=RailEnvActions.MOVE_FORWARD))
        action_plan[agent_id].append(action)

        next_action = get_action_for_move(position,
                                          direction,
                                          next_position,
                                          next_direction,
                                          self.env.rail)

        # now, we have a position need to perform the action
        action = ActionPlanElement(scheduled_at + 1,
                                   WalkingElement(
                                       position=position,
                                       direction=direction,
                                       next_action=next_action))
        action_plan[agent_id].append(action)
