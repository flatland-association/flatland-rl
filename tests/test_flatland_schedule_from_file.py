from test_utils import create_and_save_env

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, random_rail_generator, complex_rail_generator, \
    rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator, random_schedule_generator, \
    complex_schedule_generator, schedule_from_file


def test_schedule_from_file_sparse():
    """
    Test to see that all parameters are loaded as expected
    Returns
    -------

    """
    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # Generate Sparse test env
    rail_generator = sparse_rail_generator(max_num_cities=5,
                                           seed=1,
                                           grid_mode=False,
                                           max_rails_between_cities=3,
                                           max_rails_in_city=6,
                                           )
    schedule_generator = sparse_schedule_generator(speed_ration_map)

    create_and_save_env(file_name="./sparse_env_test.pkl", rail_generator=rail_generator,
                        schedule_generator=schedule_generator)


    # Sparse generator
    rail_generator = rail_from_file("./sparse_env_test.pkl")
    schedule_generator = schedule_from_file("./sparse_env_test.pkl")
    sparse_env_from_file = RailEnv(width=1, height=1, rail_generator=rail_generator,
                                   schedule_generator=schedule_generator)
    sparse_env_from_file.reset(True, True)

    # Assert loaded agent number is correct
    assert sparse_env_from_file.get_num_agents() == 10

    # Assert max steps is correct
    assert sparse_env_from_file._max_episode_steps == 500



def test_schedule_from_file_random():
    """
    Test to see that all parameters are loaded as expected
    Returns
    -------

    """
    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # Generate random test env
    rail_generator = random_rail_generator()
    schedule_generator = random_schedule_generator(speed_ration_map)

    create_and_save_env(file_name="./random_env_test.pkl", rail_generator=rail_generator,
                        schedule_generator=schedule_generator)


    # Random generator
    rail_generator = rail_from_file("./random_env_test.pkl")
    schedule_generator = schedule_from_file("./random_env_test.pkl")
    random_env_from_file = RailEnv(width=1, height=1, rail_generator=rail_generator,
                                   schedule_generator=schedule_generator)
    random_env_from_file.reset(True, True)

    # Assert loaded agent number is correct
    assert random_env_from_file.get_num_agents() == 10

    # Assert max steps is correct
    assert random_env_from_file._max_episode_steps == 1350




def test_schedule_from_file_complex():
    """
    Test to see that all parameters are loaded as expected
    Returns
    -------

    """
    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # Generate complex test env
    rail_generator = complex_rail_generator(nr_start_goal=10,
                                            nr_extra=1,
                                            min_dist=8,
                                            max_dist=99999)
    schedule_generator = complex_schedule_generator(speed_ration_map)

    create_and_save_env(file_name="./complex_env_test.pkl", rail_generator=rail_generator,
                        schedule_generator=schedule_generator)

    # Load the different envs and check the parameters


    # Complex generator
    rail_generator = rail_from_file("./complex_env_test.pkl")
    schedule_generator = schedule_from_file("./complex_env_test.pkl")
    complex_env_from_file = RailEnv(width=1, height=1, rail_generator=rail_generator,
                                    schedule_generator=schedule_generator)
    complex_env_from_file.reset(True, True)

    # Assert loaded agent number is correct
    assert complex_env_from_file.get_num_agents() == 10

    # Assert max steps is correct
    assert complex_env_from_file._max_episode_steps == 1350
