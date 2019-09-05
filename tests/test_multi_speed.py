from typing import List

import numpy as np
from attr import attrib, attrs

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent, EnvAgentStatic
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import complex_rail_generator, rail_from_grid_transition_map
from flatland.envs.schedule_generators import complex_schedule_generator, random_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail

np.random.seed(1)


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
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999,
                                                        seed=0),
                  schedule_generator=complex_schedule_generator(),
                  number_of_agents=5)
    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    agent = RandomAgent(218, 4)

    # Empty dictionary for all agent action
    action_dict = dict()

    # Set all the different speeds
    # Reset environment and get initial observations for all agents
    env.reset()
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


@attrs
class Replay(object):
    position = attrib()
    direction = attrib()
    action = attrib(type=RailEnvActions)
    malfunction = attrib(default=0, type=int)


@attrs
class TestConfig(object):
    replay = attrib(type=List[Replay])
    target = attrib()
    speed = attrib(type=float)


def test_multispeed_actions_no_malfunction_no_blocking(rendering=True):
    """Test that actions are correctly performed on cell exit for a single agent."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )

    # initialize agents_static
    env.reset()

    # reset to set agents from agents_static
    env.reset(False, False)

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")

    test_config = TestConfig(
        replay=[
            Replay(
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                action=None
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_LEFT
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.STOP_MOVING
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.STOP_MOVING
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=None
            ),
            Replay(
                position=(5, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.MOVE_FORWARD
            ),

        ],
        target=(3, 0),  # west dead-end
        speed=0.5
    )

    # TODO test penalties!
    agentStatic: EnvAgentStatic = env.agents_static[0]
    info_dict = {
        'action_required': [True]
    }
    for i, replay in enumerate(test_config.replay):
        if i == 0:
            # set the initial position
            agentStatic.position = replay.position
            agentStatic.direction = replay.direction
            agentStatic.target = test_config.target
            agentStatic.moving = True
            agentStatic.speed_data['speed'] = test_config.speed

            # reset to set agents from agents_static
            env.reset(False, False)

        def _assert(actual, expected, msg):
            assert actual == expected, "[{}] {}:  actual={}, expected={}".format(i, msg, actual, expected)

        agent: EnvAgent = env.agents[0]

        _assert(agent.position, replay.position, 'position')
        _assert(agent.direction, replay.direction, 'direction')

        if replay.action:
            assert info_dict['action_required'][0] == True, "[{}] expecting action_required={}".format(i, True)
            _, _, _, info_dict = env.step({0: replay.action})

        else:
            assert info_dict['action_required'][0] == False, "[{}] expecting action_required={}".format(i, False)
            _, _, _, info_dict = env.step({})

        if rendering:
            renderer.render_env(show=True, show_observations=True)


def test_multispeed_actions_no_malfunction_blocking(rendering=True):
    """The second agent blocks the first because it is slower."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )

    # initialize agents_static
    env.reset()

    # reset to set agents from agents_static
    env.reset(False, False)

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")

    test_configs = [
        TestConfig(
            replay=[
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),

                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),

                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),

                Replay(
                    position=(3, 5),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 5),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                Replay(
                    position=(3, 5),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                )
            ],
            target=(3, 0),  # west dead-end
            speed=1 / 3),
        TestConfig(
            replay=[
                Replay(
                    position=(3, 9),  # east dead-end
                    direction=Grid4TransitionsEnum.EAST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 9),
                    direction=Grid4TransitionsEnum.EAST,
                    action=None
                ),
                # blocked although fraction >= 1.0
                Replay(
                    position=(3, 9),
                    direction=Grid4TransitionsEnum.EAST,
                    action=None
                ),

                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                # blocked although fraction >= 1.0
                Replay(
                    position=(3, 8),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),

                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_FORWARD
                ),
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                # blocked although fraction >= 1.0
                Replay(
                    position=(3, 7),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),

                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=RailEnvActions.MOVE_LEFT
                ),
                Replay(
                    position=(3, 6),
                    direction=Grid4TransitionsEnum.WEST,
                    action=None
                ),
                # not blocked, action required!
                Replay(
                    position=(4, 6),
                    direction=Grid4TransitionsEnum.SOUTH,
                    action=RailEnvActions.MOVE_FORWARD
                ),
            ],
            target=(3, 0),  # west dead-end
            speed=0.5
        )

    ]

    # TODO test penalties!
    info_dict = {
        'action_required': [True for _ in test_configs]
    }
    for step in range(len(test_configs[0].replay)):
        if step == 0:
            for a, test_config in enumerate(test_configs):
                agentStatic: EnvAgentStatic = env.agents_static[a]
                replay = test_config.replay[0]
                # set the initial position
                agentStatic.position = replay.position
                agentStatic.direction = replay.direction
                agentStatic.target = test_config.target
                agentStatic.moving = True
                agentStatic.speed_data['speed'] = test_config.speed

            # reset to set agents from agents_static
            env.reset(False, False)

        def _assert(a, actual, expected, msg):
            assert actual == expected, "[{}] {} {}:  actual={}, expected={}".format(step, a, msg, actual, expected)

        action_dict = {}

        for a, test_config in enumerate(test_configs):
            agent: EnvAgent = env.agents[a]
            replay = test_config.replay[step]

            _assert(a, agent.position, replay.position, 'position')
            _assert(a, agent.direction, replay.direction, 'direction')



            if replay.action:
                assert info_dict['action_required'][a] == True, "[{}] agent {} expecting action_required={}".format(step, a, True)
                action_dict[a] = replay.action
            else:
                assert info_dict['action_required'][a] == False, "[{}] agent {} expecting action_required={}".format(step, a, False)
        _, _, _, info_dict = env.step(action_dict)

        if rendering:
            renderer.render_env(show=True, show_observations=True)


def test_multispeed_actions_malfunction_no_blocking(rendering=True):
    """Test on a single agent whether action on cell exit work correctly despite malfunction."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )

    # initialize agents_static
    env.reset()

    # reset to set agents from agents_static
    env.reset(False, False)

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")

    test_config = TestConfig(
        replay=[
            Replay(
                position=(3, 9),  # east dead-end
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                action=None
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD
            ),
            # add additional step in the cell
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                malfunction=2 # recovers in two steps from now!
            ),
            # agent recovers in this step
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                action=None
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=2 # recovers in two steps from now!
            ),
            # agent recovers in this step; since we're at the beginning, we provide a different action although we're broken!
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.STOP_MOVING
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.STOP_MOVING
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.MOVE_FORWARD
            ),
            Replay(
                position=(4, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=None
            ),
            Replay(
                position=(5, 6),
                direction=Grid4TransitionsEnum.SOUTH,
                action=RailEnvActions.MOVE_FORWARD
            ),

        ],
        target=(3, 0),  # west dead-end
        speed=0.5
    )

    # TODO test penalties!
    agentStatic: EnvAgentStatic = env.agents_static[0]
    info_dict = {
        'action_required': [True]
    }
    for i, replay in enumerate(test_config.replay):
        if i == 0:
            # set the initial position
            agentStatic.position = replay.position
            agentStatic.direction = replay.direction
            agentStatic.target = test_config.target
            agentStatic.moving = True
            agentStatic.speed_data['speed'] = test_config.speed

            # reset to set agents from agents_static
            env.reset(False, False)

        def _assert(actual, expected, msg):
            assert actual == expected, "[{}] {}:  actual={}, expected={}".format(i, msg, actual, expected)

        agent: EnvAgent = env.agents[0]

        _assert(agent.position, replay.position, 'position')
        _assert(agent.direction, replay.direction, 'direction')

        if replay.malfunction:
            agent.malfunction_data['malfunction'] = 2

        if replay.action:
            assert info_dict['action_required'][0] == True, "[{}] expecting action_required={}".format(i, True)
            _, _, _, info_dict = env.step({0: replay.action})

        else:
            assert info_dict['action_required'][0] == False, "[{}] expecting action_required={}".format(i, False)
            _, _, _, info_dict = env.step({})

        if rendering:
            renderer.render_env(show=True, show_observations=True)
