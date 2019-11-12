from flatland.action_plan.action_plan import PathScheduleElement, CellPin, ActionPlanReplayer
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import WalkingElement
from flatland.envs.rail_generators import rail_from_grid_transition_map
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
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
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

    chosen_path_dict = {0: [PathScheduleElement(scheduled_at=0, cell_pin=CellPin(r=3, c=0, d=3)),
                            PathScheduleElement(scheduled_at=2, cell_pin=CellPin(r=3, c=1, d=1)),
                            PathScheduleElement(scheduled_at=3, cell_pin=CellPin(r=3, c=2, d=1)),
                            PathScheduleElement(scheduled_at=14, cell_pin=CellPin(r=3, c=3, d=1)),
                            PathScheduleElement(scheduled_at=15, cell_pin=CellPin(r=3, c=4, d=1)),
                            PathScheduleElement(scheduled_at=16, cell_pin=CellPin(r=3, c=5, d=1)),
                            PathScheduleElement(scheduled_at=17, cell_pin=CellPin(r=3, c=6, d=1)),
                            PathScheduleElement(scheduled_at=18, cell_pin=CellPin(r=3, c=7, d=1)),
                            PathScheduleElement(scheduled_at=19, cell_pin=CellPin(r=3, c=8, d=1)),
                            PathScheduleElement(scheduled_at=20, cell_pin=CellPin(r=3, c=8, d=5))],
                        1: [PathScheduleElement(scheduled_at=0, cell_pin=CellPin(r=3, c=8, d=3)),
                            PathScheduleElement(scheduled_at=3, cell_pin=CellPin(r=3, c=7, d=3)),
                            PathScheduleElement(scheduled_at=5, cell_pin=CellPin(r=3, c=6, d=3)),
                            PathScheduleElement(scheduled_at=7, cell_pin=CellPin(r=3, c=5, d=3)),
                            PathScheduleElement(scheduled_at=9, cell_pin=CellPin(r=3, c=4, d=3)),
                            PathScheduleElement(scheduled_at=11, cell_pin=CellPin(r=3, c=3, d=3)),
                            PathScheduleElement(scheduled_at=13, cell_pin=CellPin(r=2, c=3, d=0)),
                            PathScheduleElement(scheduled_at=15, cell_pin=CellPin(r=1, c=3, d=0)),
                            PathScheduleElement(scheduled_at=17, cell_pin=CellPin(r=0, c=3, d=0)),
                            PathScheduleElement(scheduled_at=18, cell_pin=CellPin(r=0, c=3, d=5))]}
    expected_action_plan = [[
        # take action to enter the grid
        (0, WalkingElement(position=None, direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        # take action to enter the cell properly
        (1, WalkingElement(position=(3, 0), direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (2, WalkingElement(position=(3, 1), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (3, WalkingElement(position=(3, 2), direction=1, next_action=RailEnvActions.STOP_MOVING)),
        (13, WalkingElement(position=(3, 2), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (14, WalkingElement(position=(3, 3), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (15, WalkingElement(position=(3, 4), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (16, WalkingElement(position=(3, 5), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (17, WalkingElement(position=(3, 6), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (18, WalkingElement(position=(3, 7), direction=1, next_action=RailEnvActions.MOVE_FORWARD)),
        (19, WalkingElement(position=None, direction=1, next_action=RailEnvActions.STOP_MOVING))

    ], [
        (0, WalkingElement(position=None, direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (1, WalkingElement(position=(3, 8), direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (3, WalkingElement(position=(3, 7), direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (5, WalkingElement(position=(3, 6), direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (7, WalkingElement(position=(3, 5), direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (9, WalkingElement(position=(3, 4), direction=3, next_action=RailEnvActions.MOVE_FORWARD)),
        (11, WalkingElement(position=(3, 3), direction=3, next_action=RailEnvActions.MOVE_RIGHT)),
        (13, WalkingElement(position=(2, 3), direction=0, next_action=RailEnvActions.MOVE_FORWARD)),
        (15, WalkingElement(position=(1, 3), direction=0, next_action=RailEnvActions.MOVE_FORWARD)),
        (17, WalkingElement(position=None, direction=0, next_action=RailEnvActions.STOP_MOVING)),

    ]]

    MAX_EPISODE_STEPS = 50

    actual_action_plan = ActionPlanReplayer(env, chosen_path_dict)
    actual_action_plan.print_action_plan()
    ActionPlanReplayer.compare_action_plans(expected_action_plan, actual_action_plan.action_plan)
    assert actual_action_plan.action_plan == expected_action_plan, \
        "expected {}, found {}".format(expected_action_plan, actual_action_plan.action_plan)

    actual_action_plan.replay_verify(MAX_EPISODE_STEPS, env, rendering)
