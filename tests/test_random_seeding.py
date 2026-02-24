import tempfile

import numpy as np
import pytest

from flatland.env_generation.env_generator import env_generator, env_generator_legacy
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
    excpeted_grid = env.rail.grid.tolist()

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


def test_reset_idempotent():
    # new behaviour: reset with random_seed is idempotent
    env, _, _ = env_generator(seed=42)
    env.num_resets = 0
    env.reset(random_seed=1001)

    # rail generator seeded with this seed
    env2, _, _ = env_generator(seed=43)
    env2.num_resets = 0
    # rail generator ignores random_seed
    env2.reset(random_seed=1001)

    print(env.agents)
    print(env2.agents)
    assert env.agents == env2.agents
    assert (env.rail.grid == env2.rail.grid).all()

    print(env.random_seed)
    print(env.np_random.get_state())
    print(env2.np_random.get_state())
    assert (env.np_random.get_state()[1] == env2.np_random.get_state()[1]).all()

    print(env.malfunction_process_data)
    print(env2.malfunction_process_data)
    assert env.malfunction_process_data == env2.malfunction_process_data


def test_env_loading_no_seed_new_behaviour():
    """
    Illustrate same:
    - `env_generator` with `seed=<s1>`
    - `RailEnvPersister`
        - with save from env generated with `seed=<s1>`
        - `load_new`
    """
    # Flatland 3 envs:
    # ,test_id,env_id,n_agents,x_dim,y_dim,n_cities,max_rail_pairs_in_city,n_envs_run,seed,grid_mode,max_rails_between_cities,malfunction_duration_min,malfunction_duration_max,malfunction_interval,speed_ratios
    # 0,Test_00,Level_0,7,30,30,2,2,10,42,False,2,20,50,540,"{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}"

    env_generated, _, _ = env_generator(seed=42, max_rail_pairs_in_city=2)

    env_persisted, _, _ = env_generator(seed=42, max_rail_pairs_in_city=2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(env_persisted, f"{tmpdirname}/Test0_Level0.pkl")
        env_persisted, _ = RailEnvPersister.load_new(f"{tmpdirname}/Test0_Level0.pkl")

    assert env_generated.rail.grid.tolist() == env_persisted.rail.grid.tolist()
    assert (env_generated.np_random.get_state()[1] == env_persisted.np_random.get_state()[1]).all()


def test_env_loading_no_seed_new_behaviour_with_post_seed():
    """
    Illustrate same:
    - `env_generator` with `seed=<s1>` and `post_seed=<s2>`
    - `RailEnvPersister`
        - with save from env generated with `seed=<s1>`
        - `load_new` and `reset(random_seed=<s2>)`
    """
    # Flatland 3 envs:
    # ,test_id,env_id,n_agents,x_dim,y_dim,n_cities,max_rail_pairs_in_city,n_envs_run,seed,grid_mode,max_rails_between_cities,malfunction_duration_min,malfunction_duration_max,malfunction_interval,speed_ratios
    # 0,Test_00,Level_0,7,30,30,2,2,10,42,False,2,20,50,540,"{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}"

    env_generated, _, _ = env_generator(seed=42, max_rail_pairs_in_city=2, post_seed=44)

    env_persisted, _, _ = env_generator(seed=42, max_rail_pairs_in_city=2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(env_persisted, f"{tmpdirname}/Test0_Level0.pkl")
        env_persisted, _ = RailEnvPersister.load_new(f"{tmpdirname}/Test0_Level0.pkl")

    # env loaded with state after rail/line/schedule generation - different from post_seed
    assert not (env_generated.np_random.get_state()[1] == env_persisted.np_random.get_state()[1]).all()

    env_persisted.reset(random_seed=44)
    assert env_generated.rail.grid.tolist() == env_persisted.rail.grid.tolist()
    assert (env_generated.np_random.get_state()[1] == env_persisted.np_random.get_state()[1]).all()


def test_env_generator_no_seed():
    """
    `env_generator` without `seed` nor `post_seed` behaves
    - non-deterministically for rail generation
    - non-deterministically for initialization of env's random state
    """
    env_generated_1, _, _ = env_generator()
    env_generated_2, _, _ = env_generator()
    assert not (env_generated_1.rail.grid == env_generated_2.rail.grid).all()
    assert not (env_generated_1.np_random.get_state()[1] == env_generated_2.np_random.get_state()[1]).all()


def test_env_generator_no_seed_but_post_seed():
    """
    `env_generator` without `seed` but `post_seed` behaves
    - non-deterministically for rail generation
    - deterministically for initialization of env's random state
    """
    env_generated_1, _, _ = env_generator(post_seed=55)
    env_generated_2, _, _ = env_generator(post_seed=55)
    assert not (env_generated_1.rail.grid == env_generated_2.rail.grid).all()
    assert (env_generated_1.np_random.get_state()[1] == env_generated_2.np_random.get_state()[1]).all()


@pytest.mark.parametrize(
    "package, resource, seed",
    [
        ("env_data.tests.service_test.Test_0", "Level_0.pkl", 335971),
        ("env_data.tests.service_test.Test_0", "Level_1.pkl", 335972),
    ])
def test_env_loading_no_seed_old_behaviour(package, resource, seed):
    """
    Old pickles (without seed persisted) and no `random_seed` passed to `reset` behave
    - deterministically for rail generation
    - non-deterministically for initialization of env's random state
    """
    env_loaded, _ = RailEnvPersister.load_new(resource, load_from_package=package)
    env_loaded.reset(
        regenerate_rail=True,  # no effect, loaded from file
        regenerate_schedule=True,  # no effect, loaded from file
        random_seed=None
    )
    env_loaded2, _ = RailEnvPersister.load_new(resource, load_from_package=package)
    env_loaded2.reset(
        regenerate_rail=True,  # no effect, loaded from file
        regenerate_schedule=True,  # no effect, loaded from file
        random_seed=None
    )
    assert (env_loaded.rail.grid == env_loaded2.rail.grid).all()
    # old pickles without seed persisted!
    assert not (env_loaded.np_random.get_state()[1] == env_loaded2.np_random.get_state()[1]).all()


@pytest.mark.parametrize(
    "package, resource, seed",
    [
        ("env_data.tests.service_test.Test_0", "Level_0.pkl", 335971),
        ("env_data.tests.service_test.Test_0", "Level_1.pkl", 335972),
    ])
def test_env_loading_seed(package, resource, seed):
    """
    Old pickles (without seed persisted) and no `random_seed` passed to `reset` behave
    - deterministically for rail generation
    - deterministically for initialization of env's random state
    """
    env_loaded, _ = RailEnvPersister.load_new(resource, load_from_package=package)
    env_loaded.reset(
        regenerate_rail=True,  # no effect, loaded from file
        regenerate_schedule=True,  # no effect, loaded from file
        random_seed=42
    )
    env_loaded2, _ = RailEnvPersister.load_new(resource, load_from_package=package)
    env_loaded2.reset(
        regenerate_rail=True,  # no effect, loaded from file
        regenerate_schedule=True,  # no effect, loaded from file
        random_seed=42
    )
    assert (env_loaded.rail.grid == env_loaded2.rail.grid).all()
    assert (env_loaded.np_random.get_state()[1] == env_loaded2.np_random.get_state()[1]).all()


@pytest.mark.parametrize(
    "package, resource, seed",
    [
        ("env_data.tests.service_test.Test_0", "Level_0.pkl", 335971),
        ("env_data.tests.service_test.Test_0", "Level_1.pkl", 335972),
    ])
def test_backwards_compatibility_post_seed(package, resource, seed):
    """
    Illustrates that the random state issued by:
    - generated env with `post_seed` option
    - loaded env with `random_seed` to `reset`
    """
    env_loaded, _ = RailEnvPersister.load_new(resource, load_from_package=package)
    env_loaded.reset(
        regenerate_rail=True,  # no effect, loaded from file
        regenerate_schedule=True,  # no effect, loaded from file
        random_seed=1001
    )
    # as shown below, the generated env's rail is different:
    # - deprecated direct seeding of sparse_rail_generator
    # - modified astar
    env_generated, _, _ = env_generator(max_rail_pairs_in_city=2, seed=seed, post_seed=1001)

    assert not (env_loaded.rail.grid == env_generated.rail.grid).all()
    assert (env_generated.np_random.get_state()[1] == env_loaded.np_random.get_state()[1]).all()


@pytest.mark.parametrize(
    "package, resource, seed, expected",
    [
        ("env_data.tests.service_test.Test_0", "Level_0.pkl", 335971,
         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 16386, 1025, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 0, 49186, 34864, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 16386, 34864, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 32800, 32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 72, 37408, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 0, 32872, 37408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 0, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 32800, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 72, 1025, 1025, 1025, 1025, 38505, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 72, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32872, 4608, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 49186, 34864, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 16386, 34864, 32872, 4608, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 32800, 32800, 32800, 32800, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 72, 37408, 49186, 2064, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32872, 37408, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 49186, 2064, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        ("env_data.tests.service_test.Test_0", "Level_1.pkl", 335972,
         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 17411, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32872, 37408, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49186, 34864, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 34864, 32872, 4608, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800, 32800, 32800, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 37408, 49186, 2064, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32872, 37408, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49186, 34864, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 38505, 17411, 1025, 1025, 35889, 4608, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 2136, 52275, 5633, 1025, 2064, 72, 4608, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 32872, 37408, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 49186, 34864, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 16386, 34864, 32872, 4608, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 32800, 32800, 32800, 32800, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 72, 37408, 49186, 2064, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 32872, 37408, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 49186, 34864, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 32800, 32800, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 1025, 2064, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    ])
@pytest.mark.skip("Not backwards compatible due to new astar implementation.")
def test_env_gen(package, resource, seed, expected):
    """
    Failing regression test for sparse_rail_generator().
    Disabled - keep for documentation purposes.
    """
    env, _, _ = env_generator_legacy(x_dim=30, y_dim=30, n_agents=7, max_rail_pairs_in_city=2, seed=seed)
    assert env.rail.grid.tolist() == expected
