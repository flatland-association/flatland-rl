from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.step_utils.states import TrainState
from flatland.utils.simple_rail import make_simple_rail
from tests.test_utils import ReplayConfig, Replay, run_replay_config, set_penalties_for_replay


def test_initial_status():
    """Test that agent lifecycle works correctly ready-to-depart -> active -> done."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=False)
    env.reset()

    env._max_episode_steps = 1000

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=None,  # not entered grid yet
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,

                action=RailEnvActions.DO_NOTHING,
            ),
            Replay(  # 1
                position=None,  # not entered grid yet before step
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,

                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(  # 2
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                distance=0.0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(  # 3
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                distance=0.5,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 4
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                distance=0.0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=None,
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            # Replay(
            #     position=(3, 5),
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE
            # ),
            # Replay(
            #     position=(3, 5),
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE
            # )

        ],
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
        target=(3, 5),
        speed=0.5
    )

    run_replay_config(env, [test_config], activate_agents=False, skip_reward_check=True, skip_action_required_check=True,
                      set_ready_to_depart=True)

    assert env.agents[0].state == TrainState.DONE


def test_status_done_remove():
    """Test that agent lifecycle works correctly ready-to-depart -> active -> done."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=True)
    env.reset()

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    env._max_episode_steps = 1000

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=None,  # not entered grid yet
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.DO_NOTHING,

            ),
            Replay(  # 1
                position=None,  # not entered grid yet before step
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(  # 2
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 3
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=None,
            ),
            Replay(  # 4
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 5
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=None,

            ),
            Replay(  # 6
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                state=TrainState.MOVING
            ),
            Replay(  # 7
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            Replay(  # 8
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            # Replay(
            #     position=None,
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE_REMOVED
            # ),
            # Replay(
            #     position=None,
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE_REMOVED
            # )

        ],
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
        target=(3, 5),
        speed=0.5
    )

    run_replay_config(env, [test_config], activate_agents=False, skip_reward_check=True, skip_action_required_check=True,
                      set_ready_to_depart=True)
    assert env.agents[0].state == TrainState.DONE
