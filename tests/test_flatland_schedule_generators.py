from test_utils import create_and_save_env

from flatland.envs.rail_generators import sparse_rail_generator, random_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, random_schedule_generator, \
    complex_schedule_generator


def test_schedule_from_file():
    """
    Test to see that all parameters are loaded as expected
    Returns
    -------

    """
    # Generate Sparse test env
    rail_generator = sparse_rail_generator(max_num_cities=5,
                                           seed=1,
                                           grid_mode=False,
                                           max_rails_between_cities=3,
                                           max_rails_in_city=6,
                                           )

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    schedule_generator = sparse_schedule_generator(
        speed_ration_map)

    create_and_save_env(file_name="./sparse_env_test.pkl", rail_generator=rail_generator,
                        schedule_generator=schedule_generator)

    # Generate random test env
    rail_generator = random_rail_generator()

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    schedule_generator = random_schedule_generator(
        speed_ration_map)

    create_and_save_env(file_name="./random_env_test.pkl", rail_generator=rail_generator,
                        schedule_generator=schedule_generator)

    # Generate complex test env
    rail_generator = complex_rail_generator(nr_start_goal=10,
                                            nr_extra=1,
                                            min_dist=8,
                                            max_dist=99999)

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    schedule_generator = complex_schedule_generator(
        speed_ration_map)

    create_and_save_env(file_name="./complex_env_test.pkl", rail_generator=rail_generator,
                        schedule_generator=schedule_generator)

# def test_sparse_schedule_generator():


# def test_random_schedule_generator():


# def test_complex_schedule_generator():
