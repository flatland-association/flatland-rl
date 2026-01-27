import tempfile

import numpy as np
import pytest

from flatland.env_generation.env_generator import env_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map, sparse_rail_generator
from flatland.utils.simple_rail import make_simple_rail2


def ndom_seeding():
    # Set fixed malfunction duration for this test
    rail, rail_map, optionals = make_simple_rail2()

    # Move target to unreachable position in order to not interfere with test
    for idx in range(100):
        env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                      line_generator=sparse_line_generator(seed=12), number_of_agents=10)
        env.reset(True, True, random_seed=1)

        env.agents[0].target = (0, 0)
        for step in range(10):
            actions = {}
            actions[0] = 2
            env.step(actions)
        agent_positions = []

        env.agents[0].initial_position == (3, 2)
        env.agents[1].initial_position == (3, 5)
        env.agents[2].initial_position == (3, 6)
        env.agents[3].initial_position == (5, 6)
        env.agents[4].initial_position == (3, 4)
        env.agents[5].initial_position == (3, 1)
        env.agents[6].initial_position == (3, 9)
        env.agents[7].initial_position == (4, 6)
        env.agents[8].initial_position == (0, 3)
        env.agents[9].initial_position == (3, 7)
        # Test generation print
        # for a in range(env.get_num_agents()):
        #    print("env.agents[{}].initial_position == {}".format(a,env.agents[a].initial_position))
        # print("env.agents[0].initial_position == {}".format(env.agents[0].initial_position))
        # print("assert env.agents[0].position ==  {}".format(env.agents[0].position))


def test_seeding_and_observations():
    # Test if two different instances diverge with different observations
    rail, rail_map, optionals = make_simple_rail2()
    optionals['agents_hints']['num_agents'] = 10
    # Make two seperate envs with different observation builders
    # Global Observation
    env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=12), number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())
    # Tree Observation
    env2 = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                   line_generator=sparse_line_generator(seed=12), number_of_agents=10,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))

    env.reset(False, False, random_seed=12)
    env2.reset(False, False, random_seed=12)
    # Check that both environments produce the same initial start positions
    assert env.agents[0].initial_position == env2.agents[0].initial_position
    assert env.agents[1].initial_position == env2.agents[1].initial_position
    assert env.agents[2].initial_position == env2.agents[2].initial_position
    assert env.agents[3].initial_position == env2.agents[3].initial_position
    assert env.agents[4].initial_position == env2.agents[4].initial_position
    assert env.agents[5].initial_position == env2.agents[5].initial_position
    assert env.agents[6].initial_position == env2.agents[6].initial_position
    assert env.agents[7].initial_position == env2.agents[7].initial_position
    assert env.agents[8].initial_position == env2.agents[8].initial_position
    assert env.agents[9].initial_position == env2.agents[9].initial_position

    action_dict = {}
    for step in range(10):
        for a in range(env.get_num_agents()):
            action = np.random.randint(4)
            action_dict[a] = action
        env.step(action_dict)
        env2.step(action_dict)
    # Check that both environments end up in the same position
    assert env.agents[0].position == env2.agents[0].position
    assert env.agents[1].position == env2.agents[1].position
    assert env.agents[2].position == env2.agents[2].position
    assert env.agents[3].position == env2.agents[3].position
    assert env.agents[4].position == env2.agents[4].position
    assert env.agents[5].position == env2.agents[5].position
    assert env.agents[6].position == env2.agents[6].position
    assert env.agents[7].position == env2.agents[7].position
    assert env.agents[8].position == env2.agents[8].position
    assert env.agents[9].position == env2.agents[9].position
    for a in range(env.get_num_agents()):
        print("assert env.agents[{}].position == env2.agents[{}].position".format(a, a))


def test_seeding_and_malfunction():
    # Test if two different instances diverge with different observations
    rail, rail_map, optionals = make_simple_rail2()
    optionals['agents_hints']['num_agents'] = 10
    stochastic_data = {'prop_malfunction': 0.4,
                       'malfunction_rate': 2,
                       'min_duration': 10,
                       'max_duration': 10}
    # Make two seperate envs with different and see if the exhibit the same malfunctions
    # Global Observation
    for tests in range(1, 100):
        env = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                      line_generator=sparse_line_generator(), number_of_agents=10,
                      obs_builder_object=GlobalObsForRailEnv())

        # Tree Observation
        env2 = RailEnv(width=25, height=30, rail_generator=rail_from_grid_transition_map(rail, optionals),
                       line_generator=sparse_line_generator(), number_of_agents=10,
                       obs_builder_object=GlobalObsForRailEnv())

        env.reset(True, False, random_seed=tests)
        env2.reset(True, False, random_seed=tests)

        # Check that both environments produce the same initial start positions
        assert env.agents[0].initial_position == env2.agents[0].initial_position
        assert env.agents[1].initial_position == env2.agents[1].initial_position
        assert env.agents[2].initial_position == env2.agents[2].initial_position
        assert env.agents[3].initial_position == env2.agents[3].initial_position
        assert env.agents[4].initial_position == env2.agents[4].initial_position
        assert env.agents[5].initial_position == env2.agents[5].initial_position
        assert env.agents[6].initial_position == env2.agents[6].initial_position
        assert env.agents[7].initial_position == env2.agents[7].initial_position
        assert env.agents[8].initial_position == env2.agents[8].initial_position
        assert env.agents[9].initial_position == env2.agents[9].initial_position

        action_dict = {}
        for step in range(10):
            for a in range(env.get_num_agents()):
                action = np.random.randint(4)
                action_dict[a] = action
                # print("----------------------")
                # print(env.agents[a].malfunction_handler, env.agents[a].status)
                # print(env2.agents[a].malfunction_handler, env2.agents[a].status)

            _, reward1, done1, _ = env.step(action_dict)
            _, reward2, done2, _ = env2.step(action_dict)
            for a in range(env.get_num_agents()):
                assert reward1[a] == reward2[a]
                assert done1[a] == done2[a]
        # Check that both environments end up in the same position

        assert env.agents[0].position == env2.agents[0].position
        assert env.agents[1].position == env2.agents[1].position
        assert env.agents[2].position == env2.agents[2].position
        assert env.agents[3].position == env2.agents[3].position
        assert env.agents[4].position == env2.agents[4].position
        assert env.agents[5].position == env2.agents[5].position
        assert env.agents[6].position == env2.agents[6].position
        assert env.agents[7].position == env2.agents[7].position
        assert env.agents[8].position == env2.agents[8].position
        assert env.agents[9].position == env2.agents[9].position


def test_reproducability_env():
    """
    Test that no random generators are present within the env that get influenced by external np random
    """
    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.,  # Fast freight train
                        1. / 3.: 0.,  # Slow commuter train
                        1. / 4.: 0.}  # Slow freight train

    env = RailEnv(width=25, height=30, rail_generator=sparse_rail_generator(max_num_cities=5,
                                                                            max_rails_between_cities=3,
                                                                            seed=10,  # Random seed
                                                                            grid_mode=True
                                                                            ),
                  line_generator=sparse_line_generator(speed_ration_map), number_of_agents=1)
    env.reset(True, True, random_seed=1)
    excpeted_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 16386, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [16386, 17411, 1025, 5633, 17411, 3089, 1025, 1097, 5633, 17411, 1025, 5633, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 5633,
                      17411, 1025, 4608],
                     [32800, 32800, 0, 72, 3089, 5633, 1025, 17411, 1097, 2064, 0, 72, 1025, 1025, 1025, 1025, 1097, 3089, 1025, 1025, 1025, 1097, 3089, 1025,
                      37408],
                     [32800, 32800, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [32800, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 34864],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [72, 37408, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
                     [0, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 37408],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32872, 1025, 5633, 17411, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 5633,
                      17411, 1025, 34864],
                     [0, 72, 1025, 1097, 3089, 1025, 1025, 1025, 1097, 3089, 1025, 1025, 1025, 1025, 1025, 1025, 1097, 3089, 1025, 1025, 1025, 1097, 3089, 1025,
                      2064],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    assert env.rail.grid.tolist() == excpeted_grid

    # Test that we don't have interference from calling mulitple function outisde
    env2 = RailEnv(width=25, height=30, rail_generator=sparse_rail_generator(max_num_cities=5,
                                                                             max_rails_between_cities=3,
                                                                             seed=10,  # Random seed
                                                                             grid_mode=True
                                                                             ),
                   line_generator=sparse_line_generator(speed_ration_map), number_of_agents=1)
    np.random.seed(1)
    for i in range(10):
        np.random.randn()
    env2.reset(True, True, random_seed=1)
    assert env2.rail.grid.tolist() == excpeted_grid


@pytest.mark.parametrize("reset_seed_generated,reset_seed_persisted,expected_equal", [
    (None, None, True),
    (None, 1001, False),  # this is why offline evluation currently has different result
    (1001, 1001, True),  # TODO this is how we can fix offline evaluation
])
def test_env_generator_online_evaluation_service(reset_seed_generated, reset_seed_persisted, expected_equal):
    # ,test_id,env_id,n_agents,x_dim,y_dim,n_cities,max_rail_pairs_in_city,n_envs_run,seed,grid_mode,max_rails_between_cities,malfunction_duration_min,malfunction_duration_max,malfunction_interval,speed_ratios
    # 0,Test_00,Level_0,7,30,30,2,2,10,42,False,2,20,50,540,"{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}"
    env_generated, _, _ = env_generator(seed=42, max_rail_pairs_in_city=2)
    if reset_seed_generated is not None:
        env_generated.reset(random_seed=reset_seed_generated, regenerate_rail=False, regenerate_schedule=False)

    # online evaluation:
    # generated and persisted with the above params

    env_persisted, _, _ = env_generator(seed=42, max_rail_pairs_in_city=2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(env_persisted, f"{tmpdirname}/Test0_Level0.pkl")
        env_persisted, _ = RailEnvPersister.load_new(f"{tmpdirname}/Test0_Level0.pkl")
    if reset_seed_persisted is not None:
        # has rail and line generator from file, so generators will not consume random numbers
        env_persisted.reset(random_seed=reset_seed_persisted)

    assert env_generated.rail.grid.tolist() == env_persisted.rail.grid.tolist()

    assert (env_generated.np_random.get_state()[1] == env_persisted.np_random.get_state()[1]).all() == expected_equal
