import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import complex_rail_generator, rail_from_grid_transition_map
from flatland.envs.schedule_generators import complex_schedule_generator, random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail
from test_utils import ReplayConfig, Replay, run_replay_config, set_penalties_for_replay


# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment
#


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice([1, 2, 3])

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


def test_multi_speed_init():
    env = RailEnv(width=50, height=50,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999,
                                                        seed=1), schedule_generator=complex_schedule_generator(),
                  number_of_agents=5)
    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    agent = RandomAgent(218, 4)

    # Empty dictionary for all agent action
    action_dict = dict()

    # Set all the different speeds
    # Reset environment and get initial observations for all agents
    env.reset(False, False, True)

    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository
    old_pos = []
    for i_agent in range(env.get_num_agents()):
        env.agents[i_agent].speed_data['speed'] = 1. / (i_agent + 1)
        old_pos.append(env.agents[i_agent].position)

    # Run episode
    for step in range(100):

        # Choose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(0)
            action_dict.update({a: action})

            # Check that agent did not move in between its speed updates
            assert old_pos[a] == env.agents[a].position

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether they are done
        _, _, _, _ = env.step(action_dict)

        # Update old position whenever an agent was allowed to move
        for i_agent in range(env.get_num_agents()):
            if (step + 1) % (i_agent + 1) == 0:
                print(step, i_agent, env.agents[i_agent].position)
                old_pos[i_agent] = env.agents[i_agent].position


def test_multispeed_actions_no_malfunction_no_blocking():
    """Test that actions are correctly performed on cell exit for a single agent."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.start_penalty + env.step_penalty * 0.5  # starting and running at speed 0.5
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_LEFT,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.STOP_MOVING,
                reward=env.stop_penalty + env.step_penalty * 0.5  # stopping and step penalty
            ),
            #
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.STOP_MOVING,
                reward=env.step_penalty * 0.5  # step penalty for speed 0.5 when stopped
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.start_penalty + env.step_penalty * 0.5  # starting + running at speed 0.5
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(5, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
        ],
        target=(3, 0),  # west dead-end
        speed=0.5,
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
    )

    run_replay_config(env, [test_config])


def test_multispeed_actions_no_malfunction_blocking():
    """The second agent blocks the first because it is slower."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()
    set_penalties_for_replay(env)
    test_configs = [
        ReplayConfig(
            replay=[
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.start_penalty + env.step_penalty * 1.0 / 3.0  # starting and running at speed 1/3
                ),
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),

                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),

                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),

                Replay(
                    position=(3, 5),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 5),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                ),
                Replay(
                    position=(3, 5),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 1.0 / 3.0  # running at speed 1/3
                )
            ],
            target=(3, 0),  # west dead-end
            speed=1 / 3,
            initial_position=(3, 8),
            initial_direction=Grid4TransitionsEnum.WEST,
        ),
        ReplayConfig(
            replay=[
                Replay(
                    position=(3, 9),  # east dead-end
                    direction=Grid4TransitionsEnum.EAST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.start_penalty + env.step_penalty * 0.5  # starting and running at speed 0.5
                ),
                Replay(
                    position=(3, 9),
                    direction=Grid4TransitionsEnum.EAST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                # blocked although fraction >= 1.0
                Replay(
                    position=(3, 9),
                    direction=Grid4TransitionsEnum.EAST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),

                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                # blocked although fraction >= 1.0
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),

                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                # blocked although fraction >= 1.0
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),

                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_LEFT,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
                # not blocked, action required!
                Replay(
                    position=(4, 6),
                    direction=Grid4TransitionsEnum.SOUTH,
                    action=RailEnvActions.MOVE_FORWARD,
                    reward=env.step_penalty * 0.5  # running at speed 0.5
                ),
            ],
            target=(3, 0),  # west dead-end
            speed=0.5,
            initial_position=(3, 9),  # east dead-end
            initial_direction=Grid4TransitionsEnum.EAST,
        )

    ]
    run_replay_config(env, test_configs)


def test_multispeed_actions_malfunction_no_blocking():
    """Test on a single agent whether action on cell exit work correctly despite malfunction."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.start_penalty + env.step_penalty * 0.5  # starting and running at speed 0.5
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            # add additional step in the cell
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                set_malfunction=2,  # recovers in two steps from now!,
                malfunction=2,
                reward=env.step_penalty * 0.5  # step penalty for speed 0.5 when malfunctioning
            ),
            # agent recovers in this step
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                malfunction=1,
                reward=env.step_penalty * 0.5  # recovered: running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                set_malfunction=2,  # recovers in two steps from now!
                malfunction=2,
                reward=env.step_penalty * 0.5  # step penalty for speed 0.5 when malfunctioning
            ),
            # agent recovers in this step; since we're at the beginning, we provide a different action although we're broken!
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                malfunction=1,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.STOP_MOVING,
                reward=env.stop_penalty + env.step_penalty * 0.5  # stopping and step penalty for speed 0.5
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.STOP_MOVING,
                reward=env.step_penalty * 0.5  # step penalty for speed 0.5 while stopped
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.start_penalty + env.step_penalty * 0.5  # starting and running at speed 0.5
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            # DO_NOTHING keeps moving!
            Replay(
                position=(3, 5),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.DO_NOTHING,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 5),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 4),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),

        ],
        target=(3, 0),  # west dead-end
        speed=0.5,
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [test_config])


# TODO invalid action penalty seems only given when forward is not possible - is this the intended behaviour?
def test_multispeed_actions_no_malfunction_invalid_actions():
    """Test that actions are correctly performed on cell exit for a single agent."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_LEFT,
                reward=env.start_penalty + env.step_penalty * 0.5  # auto-correction left to forward without penalty!
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                reward=env.step_penalty * 0.5  # wrong action is corrected to forward without penalty!
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),
            Replay(
                position=(3, 5),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                reward=env.step_penalty * 0.5  # wrong action is corrected to forward without penalty!
            ), Replay(
                position=(3, 5),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                reward=env.step_penalty * 0.5  # running at speed 0.5
            ),

        ],
        target=(3, 0),  # west dead-end
        speed=0.5,
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
    )

    run_replay_config(env, [test_config])
