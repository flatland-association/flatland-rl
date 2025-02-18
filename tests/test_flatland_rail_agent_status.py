from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.simple_rail import make_simple_rail
from tests.test_utils import ReplayConfig, Replay, run_replay_config, set_penalties_for_replay
from flatland.envs.step_utils.states import TrainState

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
        env.step({}) # DO_NOTHING for all agents

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(
                position=None,  # not entered grid yet
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.DO_NOTHING,
                reward=env.step_penalty * 0.5,

            ),
            Replay(
                position=None,  # not entered grid yet before step
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.MOVE_LEFT,
                reward=env.step_penalty * 0.5,  # auto-correction left to forward without penalty!
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_LEFT,
                reward=env.start_penalty + env.step_penalty * 0.5,  # running at speed 0.5
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=None,
                reward=env.step_penalty * 0.5,  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5,  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=None,
                reward=env.step_penalty * 0.5,  # running at speed 0.5

            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5,  # running at speed 0.5
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5,  # wrong action is corrected to forward without penalty!
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                reward=env.step_penalty * 0.5,  #
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.global_reward,  #
                state=TrainState.MOVING
            ),
            # Replay(
            #     position=(3, 5),
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.global_reward,  # already done
            #     status=RailAgentStatus.DONE
            # ),
            # Replay(
            #     position=(3, 5),
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.global_reward,  # already done
            #     status=RailAgentStatus.DONE
            # )

        ],
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
        target=(3, 5),
        speed=0.5
    )

    run_replay_config(env, [test_config], activate_agents=False, skip_reward_check=True,
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
        env.step({}) # DO_NOTHING for all agents

    env._max_episode_steps = 1000

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(
                position=None,  # not entered grid yet
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.DO_NOTHING,
                reward=env.step_penalty * 0.5,

            ),
            Replay(
                position=None,  # not entered grid yet before step
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.MOVE_LEFT,
                reward=env.step_penalty * 0.5,  # auto-correction left to forward without penalty!
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.start_penalty + env.step_penalty * 0.5,  # running at speed 0.5
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=None,
                reward=env.step_penalty * 0.5,  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5,  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=None,
                reward=env.step_penalty * 0.5,  # running at speed 0.5

            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                reward=env.step_penalty * 0.5,  # running at speed 0.5
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5,  # wrong action is corrected to forward without penalty!
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5,  # done
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.global_reward,  # already done
                state=TrainState.MOVING
            ),
            # Replay(
            #     position=None,
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.global_reward,  # already done
            #     status=RailAgentStatus.DONE_REMOVED
            # ),
            # Replay(
            #     position=None,
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.global_reward,  # already done
            #     status=RailAgentStatus.DONE_REMOVED
            # )

        ],
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
        target=(3, 5),
        speed=0.5
    )

    run_replay_config(env, [test_config], activate_agents=False, skip_reward_check=True,
                      set_ready_to_depart=True)
    assert env.agents[0].state == TrainState.DONE
