from flatland.action_plan.action_plan import TrainrunWaypoint, ControllerFromTrainrunsReplayer, ActionPlanElement, \
    ControllerFromTrainruns
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_train_run_data_structures import Waypoint
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail


def test_action_plan(rendering: bool = False):
    """Tests ActionPlanReplayer: does action plan generation and replay work as expected."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=77),
                  number_of_agents=2,
                  obs_builder_object=GlobalObsForRailEnv(),
                  remove_agents_at_target=True
                  )
    env.reset()
    env.agents[0].initial_position = (3, 0)
    env.agents[0].target = (3, 8)
    env.agents[0].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].initial_position = (3, 8)
    env.agents[1].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].target = (0, 3)
    env.agents[1].speed_data['speed'] = 0.5  # two
    env.reset(False, False, False)
    for handle, agent in enumerate(env.agents):
        print("[{}] {} -> {}".format(handle, agent.initial_position, agent.target))

    chosen_path_dict = {0: [TrainrunWaypoint(scheduled_at=0, way_point=Waypoint(position=(3, 0), direction=3)),
                            TrainrunWaypoint(scheduled_at=2, way_point=Waypoint(position=(3, 1), direction=1)),
                            TrainrunWaypoint(scheduled_at=3, way_point=Waypoint(position=(3, 2), direction=1)),
                            TrainrunWaypoint(scheduled_at=14, way_point=Waypoint(position=(3, 3), direction=1)),
                            TrainrunWaypoint(scheduled_at=15, way_point=Waypoint(position=(3, 4), direction=1)),
                            TrainrunWaypoint(scheduled_at=16, way_point=Waypoint(position=(3, 5), direction=1)),
                            TrainrunWaypoint(scheduled_at=17, way_point=Waypoint(position=(3, 6), direction=1)),
                            TrainrunWaypoint(scheduled_at=18, way_point=Waypoint(position=(3, 7), direction=1)),
                            TrainrunWaypoint(scheduled_at=19, way_point=Waypoint(position=(3, 8), direction=1)),
                            TrainrunWaypoint(scheduled_at=20, way_point=Waypoint(position=(3, 8), direction=5))],
                        1: [TrainrunWaypoint(scheduled_at=0, way_point=Waypoint(position=(3, 8), direction=3)),
                            TrainrunWaypoint(scheduled_at=3, way_point=Waypoint(position=(3, 7), direction=3)),
                            TrainrunWaypoint(scheduled_at=5, way_point=Waypoint(position=(3, 6), direction=3)),
                            TrainrunWaypoint(scheduled_at=7, way_point=Waypoint(position=(3, 5), direction=3)),
                            TrainrunWaypoint(scheduled_at=9, way_point=Waypoint(position=(3, 4), direction=3)),
                            TrainrunWaypoint(scheduled_at=11, way_point=Waypoint(position=(3, 3), direction=3)),
                            TrainrunWaypoint(scheduled_at=13, way_point=Waypoint(position=(2, 3), direction=0)),
                            TrainrunWaypoint(scheduled_at=15, way_point=Waypoint(position=(1, 3), direction=0)),
                            TrainrunWaypoint(scheduled_at=17, way_point=Waypoint(position=(0, 3), direction=0))]}
    expected_action_plan = [[
        # take action to enter the grid
        ActionPlanElement(0, RailEnvActions.MOVE_FORWARD),
        # take action to enter the cell properly
        ActionPlanElement(1, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(2, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(3, RailEnvActions.STOP_MOVING),
        ActionPlanElement(13, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(14, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(15, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(16, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(17, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(18, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(19, RailEnvActions.STOP_MOVING)

    ], [
        ActionPlanElement(0, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(1, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(3, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(5, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(7, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(9, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(11, RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(13, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(15, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(17, RailEnvActions.STOP_MOVING),

    ]]

    deterministic_controller = ControllerFromTrainruns(env, chosen_path_dict)
    deterministic_controller.print_action_plan()
    ControllerFromTrainruns.assert_actions_plans_equal(expected_action_plan, deterministic_controller.action_plan)
    ControllerFromTrainrunsReplayer.replay_verify(deterministic_controller, env, rendering)
