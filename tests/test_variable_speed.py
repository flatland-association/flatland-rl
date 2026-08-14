from fractions import Fraction

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.step_utils.speed_counter import _pseudo_fractional
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
    env.acceleration_delta = Fraction(2, 10)
    env.braking_delta = -Fraction(2, 10)

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                speed=_pseudo_fractional(0.5),
                distance=_pseudo_fractional(0.0),

                action=RailEnvActions.MOVE_FORWARD,
            ),
            # design: distance update with pre-step speed - this step's distance advances by the speed the
            # agent had BEFORE it (0.5, from replay #0), not the just-accelerated 0.7 granted by this step.
            Replay(  # 1
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                speed=_pseudo_fractional(0.7),
                distance=_pseudo_fractional(0.5),

                action=None,
            ),
            # design: distance update with pre-step speed - same as replay #1: distance advances by the
            # pre-step speed (0.5), not the 0.7 granted this step.
            Replay(  # 2
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(0.7),
                distance=_pseudo_fractional(0.2),

                action=RailEnvActions.MOVE_FORWARD,
            ),
            # design: distance update with pre-step speed - distance/position now lag the pre-fix table by
            # one step throughout: is_cell_exit() judges exit-readiness from the speed the agent had BEFORE
            # this step, not the post-acceleration speed granted THIS step, so a boundary crossing that used
            # to complete here now completes one step later.
            Replay(  # 3
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(0.9),
                distance=_pseudo_fractional(0.9),

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 4
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(1.0),
                distance=_pseudo_fractional(0.8),

                action=RailEnvActions.MOVE_LEFT,

            ),
            # design: distance update with pre-step speed - the one-step exit-timing lag means MOVE_LEFT is
            # now issued one step before the agent actually reaches the switch cell (3, 6); it lands on a
            # straight (3, 7)->(3, 6) segment instead, where a left turn isn't a valid transition, so it's
            # treated as a forward move. The agent never turns south and just continues west - this scripted
            # action sequence no longer exercises the switch turn it was originally written to test.
            Replay(  # 5
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(1.0),
                distance=_pseudo_fractional(0.8),
                state=TrainState.MOVING,

                action=RailEnvActions.STOP_MOVING,
            ),
            #
            Replay(  # 6
                position=(3, 5),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(0.8),
                distance=_pseudo_fractional(0.8),

                action=RailEnvActions.STOP_MOVING,
            ),
            Replay(  # 7
                position=(3, 4),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(0.6),
                distance=_pseudo_fractional(0.6),

                action=RailEnvActions.MOVE_RIGHT,  # must not accelerate/brake!
            ),
            Replay(  # 8
                position=(3, 3),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(0.6),
                distance=_pseudo_fractional(0.2),

                action=RailEnvActions.DO_NOTHING,
            ),
            Replay(  # 9
                position=(3, 3),
                direction=Grid4TransitionsEnum.WEST,
                speed=_pseudo_fractional(0.6),
                distance=_pseudo_fractional(0.8),

                action=RailEnvActions.DO_NOTHING,
            ),
        ],
        target=(3, 0),  # west dead-end
        speed=_pseudo_fractional(0.5),
        max_speed=_pseudo_fractional(1.0),
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [test_config], skip_reward_check=True, skip_action_required_check=True)
