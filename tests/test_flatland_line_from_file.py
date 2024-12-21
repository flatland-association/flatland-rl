from flatland.envs.line_generators import sparse_line_generator, line_from_file
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file
from tests.test_utils import create_and_save_env


def test_line_from_file_sparse():
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
                                           max_rail_pairs_in_city=3,
                                           )
    line_generator = sparse_line_generator(speed_ration_map)

    env = create_and_save_env(file_name="./sparse_env_test.pkl", rail_generator=rail_generator,
                        line_generator=line_generator)
    old_num_steps = env._max_episode_steps
    old_num_agents = len(env.agents)


    # Sparse generator
    rail_generator = rail_from_file("./sparse_env_test.pkl")
    line_generator = line_from_file("./sparse_env_test.pkl")
    sparse_env_from_file = RailEnv(width=1, height=1, rail_generator=rail_generator,
                                   line_generator=line_generator)
    sparse_env_from_file.reset(True, True)

    # Assert loaded agent number is correct
    assert sparse_env_from_file.get_num_agents() == old_num_agents

    # Assert max steps is correct
    assert sparse_env_from_file._max_episode_steps == old_num_steps
