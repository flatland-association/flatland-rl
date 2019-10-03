import random
from typing import Dict, List

import numpy as np
from test_utils import Replay, ReplayConfig, run_replay_config, set_penalties_for_replay

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import complex_rail_generator, sparse_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator, sparse_schedule_generator


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def test_malfunction_process():
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 1000,
                       'min_duration': 3,
                       'max_duration': 3}
    np.random.seed(5)

    env = RailEnv(width=20,
                  height=20,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
                                                        seed=0),
                  schedule_generator=complex_schedule_generator(),
                  number_of_agents=2,
                  obs_builder_object=SingleAgentNavigationObs(),
                  stochastic_data=stochastic_data)

    obs = env.reset()

    # Check that a initial duration for malfunction was assigned
    assert env.agents[0].malfunction_data['next_malfunction'] > 0

    agent_halts = 0
    total_down_time = 0
    agent_old_position = env.agents[0].position
    for step in range(100):
        actions = {}
        for i in range(len(obs)):
            actions[i] = np.argmax(obs[i]) + 1

        if step % 5 == 0:
            # Stop the agent and set it to be malfunctioning
            env.agents[0].malfunction_data['malfunction'] = -1
            env.agents[0].malfunction_data['next_malfunction'] = 0
            agent_halts += 1

        obs, all_rewards, done, _ = env.step(actions)

        if env.agents[0].malfunction_data['malfunction'] > 0:
            agent_malfunctioning = True
        else:
            agent_malfunctioning = False

        if agent_malfunctioning:
            # Check that agent is not moving while malfunctioning
            assert agent_old_position == env.agents[0].position

        agent_old_position = env.agents[0].position
        total_down_time += env.agents[0].malfunction_data['malfunction']

    # Check that the appropriate number of malfunctions is achieved
    assert env.agents[0].malfunction_data['nr_malfunctions'] == 21

    # Check that 20 stops where performed
    assert agent_halts == 20

    # Check that malfunctioning data was standing around
    assert total_down_time > 0


def test_malfunction_process_statistically():
    """Tests hat malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 2,
                       'min_duration': 3,
                       'max_duration': 3}
    np.random.seed(5)
    random.seed(0)

    env = RailEnv(width=20,
                  height=20,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
                                                        seed=0),
                  schedule_generator=complex_schedule_generator(),
                  number_of_agents=2,
                  obs_builder_object=SingleAgentNavigationObs(),
                  stochastic_data=stochastic_data)

    env.reset()
    nb_malfunction = 0
    for step in range(100):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            if agent.malfunction_data['malfunction'] > 0:
                nb_malfunction += 1
            # We randomly select an action
            action_dict[agent.handle] = RailEnvActions(np.random.randint(4))

        env.step(action_dict)

    # check that generation of malfunctions works as expected
    assert nb_malfunction == 156, "nb_malfunction={}".format(nb_malfunction)


def test_initial_malfunction():
    random.seed(0)
    np.random.seed(0)

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.,  # Fast freight train
                        1. / 3.: 0.,  # Slow commuter train
                        1. / 4.: 0.}  # Slow freight train

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=sparse_rail_generator(max_num_cities=5,
                                                       # Number of cities in map (where train stations are)
                                                       num_intersections=4,
                                                       # Number of intersections (no start / target)
                                                       num_trainstations=25,  # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,  # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities/intersections
                                                       seed=215545,  # Random seed
                                                       grid_mode=True
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=2,
                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action MOVE_FORWARD, agent should restart and move to the next cell
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=1,
                reward=env.start_penalty + env.step_penalty * 1.0
                # malfunctioning ends: starting and running at speed 1.0
            ),
            Replay(
                position=(28, 4),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0  # running at speed 1.0
            ),
            Replay(
                position=(27, 4),
                direction=Grid4TransitionsEnum.NORTH,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0  # running at speed 1.0
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target
    )
    run_replay_config(env, [replay_config])


def test_initial_malfunction_stop_moving():
    random.seed(0)
    np.random.seed(0)

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.,  # Fast freight train
                        1. / 3.: 0.,  # Slow commuter train
                        1. / 4.: 0.}  # Slow freight train

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=sparse_rail_generator(max_num_cities=5,
                                                       # Number of cities in map (where train stations are)
                                                       num_intersections=4,
                                                       # Number of intersections (no start / target)
                                                       num_trainstations=25,  # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,  # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities/intersections
                                                       seed=215545,  # Random seed
                                                       grid_mode=True,
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty  # full step penalty when stopped
            ),
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=2,
                reward=env.step_penalty  # full step penalty when stopped
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action STOP_MOVING, agent should restart without moving
            #
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.STOP_MOVING,
                malfunction=1,
                reward=env.step_penalty  # full step penalty while stopped
            ),
            # we have stopped and do nothing --> should stand still
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=0,
                reward=env.step_penalty  # full step penalty while stopped
            ),
            # we start to move forward --> should go to next cell now
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0  # full step penalty while stopped
            ),
            Replay(
                position=(28, 4),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0  # full step penalty while stopped
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target
    )

    run_replay_config(env, [replay_config])


def test_initial_malfunction_do_nothing():
    random.seed(0)
    np.random.seed(0)

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.,  # Fast freight train
                        1. / 3.: 0.,  # Slow commuter train
                        1. / 4.: 0.}  # Slow freight train

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=sparse_rail_generator(max_num_cities=5,
                                                       # Number of cities in map (where train stations are)
                                                       num_intersections=4,
                                                       # Number of intersections (no start / target)
                                                       num_trainstations=25,  # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,  # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities/intersections
                                                       seed=215545,  # Random seed
                                                       grid_mode=True,
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[Replay(
            position=(28, 5),
            direction=Grid4TransitionsEnum.EAST,
            action=RailEnvActions.DO_NOTHING,
            set_malfunction=3,
            malfunction=3,
            reward=env.step_penalty  # full step penalty while malfunctioning
        ),
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=2,
                reward=env.step_penalty  # full step penalty while malfunctioning
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action DO_NOTHING, agent should restart without moving
            #
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=1,
                reward=env.step_penalty  # full step penalty while stopped
            ),
            # we haven't started moving yet --> stay here
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=0,
                reward=env.step_penalty  # full step penalty while stopped
            ),
            # we start to move forward --> should go to next cell now
            Replay(
                position=(28, 5),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0  # start penalty + step penalty for speed 1.0
            ),
            Replay(
                position=(28, 4),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0  # step penalty for speed 1.0
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target
    )

    run_replay_config(env, [replay_config])


def test_initial_nextmalfunction_not_below_zero():
    random.seed(0)
    np.random.seed(0)

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 0.5,  # Rate of malfunction occurence
                       'min_duration': 5,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.,  # Fast freight train
                        1. / 3.: 0.,  # Slow commuter train
                        1. / 4.: 0.}  # Slow freight train

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=sparse_rail_generator(max_num_cities=5,
                                                       # Number of cities in map (where train stations are)
                                                       num_intersections=4,
                                                       # Number of intersections (no start / target)
                                                       num_trainstations=25,  # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,  # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities/intersections
                                                       seed=215545,  # Random seed
                                                       grid_mode=True,
                                                       enhance_intersection=False
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    agent = env.agents[0]
    env.step({})
    # was next_malfunction was -1 befor the bugfix https://gitlab.aicrowd.com/flatland/flatland/issues/186
    assert agent.malfunction_data['next_malfunction'] >= 0, \
        "next_malfunction should be >=0, found {}".format(agent.malfunction_data['next_malfunction'])
