from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.step_utils.states import TrainState
from flatland.utils.simple_rail import make_simple_rail
from tests.test_utils import ReplayConfig, Replay, run_replay_config, set_penalties_for_replay


def test_variablespeed_actions_no_malfunction_no_blocking():
    """Test that actions are correctly performed on cell exit for a single agent."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    env._max_episode_steps = 1000
    env.acceleration_delta = 0.2
    env.braking_delta = -0.2

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                speed=0.5,
                distance=0.0,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 1
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                speed=0.7,
                distance=0.7,

                action=None,
            ),
            Replay(  # 2
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                speed=0.7,
                distance=0.4,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 3
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                speed=0.9,
                distance=0.3,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 4
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                speed=1.0,
                distance=0.3,

                action=RailEnvActions.MOVE_LEFT,

            ),
            Replay(  # 5
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                speed=1.0,
                distance=0.3,
                state=TrainState.MOVING,

                action=RailEnvActions.STOP_MOVING,
            ),
            #
            Replay(  # 6
                position=(5, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                speed=0.8,
                distance=0.1,

                action=RailEnvActions.STOP_MOVING,
            ),
            Replay(  # 7
                position=(5, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                speed=0.6,
                distance=0.7,

                action=RailEnvActions.MOVE_RIGHT,  # must not accelerate/brake!
            ),
            Replay(  # 8
                position=(6, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                speed=0.6,
                distance=0.3,

                action=RailEnvActions.DO_NOTHING,
            ),
            Replay(  # 9
                position=(6, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                speed=0.6,
                distance=0.9,

                action=RailEnvActions.DO_NOTHING,
            ),
        ],
        target=(3, 0),  # west dead-end
        speed=0.5,
        max_speed=1.0,
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [test_config], skip_reward_check=True, skip_action_required_check=True)
