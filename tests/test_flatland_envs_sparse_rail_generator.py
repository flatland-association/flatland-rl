import unittest
import warnings

import numpy as np
from numpy.random import RandomState

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool


def test_sparse_rail_generator():
    generate = sparse_rail_generator(max_num_cities=10, max_rails_between_cities=3, grid_mode=False)
    actual_grid_map, _ = generate(width=50, height=50, num_agents=10, np_random=RandomState(1))
    # for r in range(env.height):
    #     for c in range(env.width):
    #         if env.rail.grid[r][c] > 0:
    #             print("expected_grid_map[{}][{}] = {}".format(r, c, env.rail.grid[r][c]))
    expected_grid_map = np.zeros((50, 50))
    expected_grid_map[7][15] = 16386
    expected_grid_map[7][16] = 1025
    expected_grid_map[7][17] = 4608
    expected_grid_map[7][21] = 16386
    expected_grid_map[7][22] = 1025
    expected_grid_map[7][23] = 1025
    expected_grid_map[7][24] = 1025
    expected_grid_map[7][25] = 1025
    expected_grid_map[7][26] = 17411
    expected_grid_map[7][27] = 1025
    expected_grid_map[7][28] = 5633
    expected_grid_map[7][29] = 1025
    expected_grid_map[7][30] = 1025
    expected_grid_map[7][31] = 1025
    expected_grid_map[7][32] = 1025
    expected_grid_map[7][33] = 1025
    expected_grid_map[7][34] = 4608
    expected_grid_map[8][10] = 16386
    expected_grid_map[8][11] = 1025
    expected_grid_map[8][12] = 1025
    expected_grid_map[8][13] = 5633
    expected_grid_map[8][14] = 17411
    expected_grid_map[8][15] = 3089
    expected_grid_map[8][16] = 1025
    expected_grid_map[8][17] = 1097
    expected_grid_map[8][18] = 5633
    expected_grid_map[8][19] = 17411
    expected_grid_map[8][20] = 1025
    expected_grid_map[8][21] = 3089
    expected_grid_map[8][22] = 1025
    expected_grid_map[8][23] = 1025
    expected_grid_map[8][24] = 5633
    expected_grid_map[8][25] = 1025
    expected_grid_map[8][26] = 3089
    expected_grid_map[8][27] = 1025
    expected_grid_map[8][28] = 1097
    expected_grid_map[8][29] = 1025
    expected_grid_map[8][30] = 17411
    expected_grid_map[8][31] = 1025
    expected_grid_map[8][32] = 1025
    expected_grid_map[8][33] = 1025
    expected_grid_map[8][34] = 38505
    expected_grid_map[8][35] = 4608
    expected_grid_map[9][10] = 32800
    expected_grid_map[9][11] = 16386
    expected_grid_map[9][12] = 1025
    expected_grid_map[9][13] = 1097
    expected_grid_map[9][14] = 3089
    expected_grid_map[9][15] = 5633
    expected_grid_map[9][16] = 1025
    expected_grid_map[9][17] = 17411
    expected_grid_map[9][18] = 1097
    expected_grid_map[9][19] = 3089
    expected_grid_map[9][20] = 1025
    expected_grid_map[9][21] = 1025
    expected_grid_map[9][22] = 17411
    expected_grid_map[9][23] = 1025
    expected_grid_map[9][24] = 1097
    expected_grid_map[9][25] = 1025
    expected_grid_map[9][26] = 5633
    expected_grid_map[9][27] = 1025
    expected_grid_map[9][28] = 17411
    expected_grid_map[9][29] = 1025
    expected_grid_map[9][30] = 3089
    expected_grid_map[9][31] = 1025
    expected_grid_map[9][32] = 5633
    expected_grid_map[9][33] = 4608
    expected_grid_map[9][34] = 72
    expected_grid_map[9][35] = 37408
    expected_grid_map[9][39] = 16386
    expected_grid_map[9][40] = 1025
    expected_grid_map[9][41] = 4608
    expected_grid_map[10][10] = 32800
    expected_grid_map[10][11] = 32800
    expected_grid_map[10][15] = 72
    expected_grid_map[10][16] = 1025
    expected_grid_map[10][17] = 2064
    expected_grid_map[10][22] = 32800
    expected_grid_map[10][26] = 72
    expected_grid_map[10][27] = 1025
    expected_grid_map[10][28] = 2064
    expected_grid_map[10][32] = 72
    expected_grid_map[10][33] = 1097
    expected_grid_map[10][34] = 1025
    expected_grid_map[10][35] = 1097
    expected_grid_map[10][36] = 1025
    expected_grid_map[10][37] = 5633
    expected_grid_map[10][38] = 17411
    expected_grid_map[10][39] = 3089
    expected_grid_map[10][40] = 1025
    expected_grid_map[10][41] = 1097
    expected_grid_map[10][42] = 5633
    expected_grid_map[10][43] = 17411
    expected_grid_map[10][44] = 1025
    expected_grid_map[10][45] = 4608
    expected_grid_map[11][10] = 32800
    expected_grid_map[11][11] = 32800
    expected_grid_map[11][22] = 32800
    expected_grid_map[11][37] = 72
    expected_grid_map[11][38] = 3089
    expected_grid_map[11][39] = 5633
    expected_grid_map[11][40] = 1025
    expected_grid_map[11][41] = 17411
    expected_grid_map[11][42] = 1097
    expected_grid_map[11][43] = 2064
    expected_grid_map[11][45] = 32800
    expected_grid_map[12][10] = 32800
    expected_grid_map[12][11] = 32800
    expected_grid_map[12][22] = 72
    expected_grid_map[12][23] = 5633
    expected_grid_map[12][24] = 5633
    expected_grid_map[12][25] = 4608
    expected_grid_map[12][39] = 72
    expected_grid_map[12][40] = 1025
    expected_grid_map[12][41] = 2064
    expected_grid_map[12][45] = 32800
    expected_grid_map[13][10] = 49186
    expected_grid_map[13][11] = 34864
    expected_grid_map[13][23] = 32800
    expected_grid_map[13][24] = 32800
    expected_grid_map[13][25] = 72
    expected_grid_map[13][26] = 4608
    expected_grid_map[13][45] = 32800
    expected_grid_map[14][10] = 32800
    expected_grid_map[14][11] = 32800
    expected_grid_map[14][23] = 32800
    expected_grid_map[14][24] = 72
    expected_grid_map[14][25] = 4608
    expected_grid_map[14][26] = 32800
    expected_grid_map[14][45] = 32800
    expected_grid_map[15][10] = 32800
    expected_grid_map[15][11] = 32800
    expected_grid_map[15][23] = 72
    expected_grid_map[15][24] = 4608
    expected_grid_map[15][25] = 32800
    expected_grid_map[15][26] = 32800
    expected_grid_map[15][43] = 16386
    expected_grid_map[15][44] = 17411
    expected_grid_map[15][45] = 2064
    expected_grid_map[16][10] = 32800
    expected_grid_map[16][11] = 32800
    expected_grid_map[16][24] = 32800
    expected_grid_map[16][25] = 32800
    expected_grid_map[16][26] = 32800
    expected_grid_map[16][42] = 16386
    expected_grid_map[16][43] = 2064
    expected_grid_map[16][44] = 32800
    expected_grid_map[17][0] = 16386
    expected_grid_map[17][1] = 1025
    expected_grid_map[17][2] = 5633
    expected_grid_map[17][3] = 17411
    expected_grid_map[17][4] = 1025
    expected_grid_map[17][5] = 1025
    expected_grid_map[17][6] = 1025
    expected_grid_map[17][7] = 5633
    expected_grid_map[17][8] = 17411
    expected_grid_map[17][9] = 1025
    expected_grid_map[17][10] = 3089
    expected_grid_map[17][11] = 2064
    expected_grid_map[17][24] = 32800
    expected_grid_map[17][25] = 32872
    expected_grid_map[17][26] = 37408
    expected_grid_map[17][42] = 32800
    expected_grid_map[17][43] = 16386
    expected_grid_map[17][44] = 34864
    expected_grid_map[18][0] = 32800
    expected_grid_map[18][2] = 72
    expected_grid_map[18][3] = 3089
    expected_grid_map[18][4] = 1025
    expected_grid_map[18][5] = 1025
    expected_grid_map[18][6] = 1025
    expected_grid_map[18][7] = 1097
    expected_grid_map[18][8] = 2064
    expected_grid_map[18][24] = 32800
    expected_grid_map[18][25] = 32800
    expected_grid_map[18][26] = 32800
    expected_grid_map[18][41] = 16386
    expected_grid_map[18][42] = 33825
    expected_grid_map[18][43] = 33825
    expected_grid_map[18][44] = 2064
    expected_grid_map[19][0] = 32800
    expected_grid_map[19][24] = 49186
    expected_grid_map[19][25] = 34864
    expected_grid_map[19][26] = 32872
    expected_grid_map[19][27] = 4608
    expected_grid_map[19][41] = 32800
    expected_grid_map[19][42] = 32800
    expected_grid_map[19][43] = 32800
    expected_grid_map[20][0] = 32800
    expected_grid_map[20][24] = 32800
    expected_grid_map[20][25] = 32800
    expected_grid_map[20][26] = 32800
    expected_grid_map[20][27] = 32800
    expected_grid_map[20][41] = 32800
    expected_grid_map[20][42] = 32800
    expected_grid_map[20][43] = 32800
    expected_grid_map[21][0] = 32800
    expected_grid_map[21][24] = 32872
    expected_grid_map[21][25] = 37408
    expected_grid_map[21][26] = 49186
    expected_grid_map[21][27] = 2064
    expected_grid_map[21][41] = 32800
    expected_grid_map[21][42] = 32800
    expected_grid_map[21][43] = 32800
    expected_grid_map[22][0] = 72
    expected_grid_map[22][1] = 5633
    expected_grid_map[22][2] = 4608
    expected_grid_map[22][24] = 32800
    expected_grid_map[22][25] = 32800
    expected_grid_map[22][26] = 32800
    expected_grid_map[22][41] = 32800
    expected_grid_map[22][42] = 49186
    expected_grid_map[22][43] = 2064
    expected_grid_map[23][1] = 32800
    expected_grid_map[23][2] = 32800
    expected_grid_map[23][24] = 32800
    expected_grid_map[23][25] = 49186
    expected_grid_map[23][26] = 34864
    expected_grid_map[23][41] = 32800
    expected_grid_map[23][42] = 32800
    expected_grid_map[24][1] = 32800
    expected_grid_map[24][2] = 32800
    expected_grid_map[24][24] = 32800
    expected_grid_map[24][25] = 32800
    expected_grid_map[24][26] = 32800
    expected_grid_map[24][41] = 32872
    expected_grid_map[24][42] = 37408
    expected_grid_map[25][1] = 32800
    expected_grid_map[25][2] = 32800
    expected_grid_map[25][24] = 72
    expected_grid_map[25][25] = 38505
    expected_grid_map[25][26] = 37408
    expected_grid_map[25][41] = 49186
    expected_grid_map[25][42] = 34864
    expected_grid_map[26][1] = 32800
    expected_grid_map[26][2] = 32800
    expected_grid_map[26][25] = 72
    expected_grid_map[26][26] = 37408
    expected_grid_map[26][40] = 16386
    expected_grid_map[26][41] = 34864
    expected_grid_map[26][42] = 32872
    expected_grid_map[26][43] = 4608
    expected_grid_map[27][1] = 32800
    expected_grid_map[27][2] = 32800
    expected_grid_map[27][26] = 32800
    expected_grid_map[27][40] = 32800
    expected_grid_map[27][41] = 32800
    expected_grid_map[27][42] = 32800
    expected_grid_map[27][43] = 32800
    expected_grid_map[28][1] = 32800
    expected_grid_map[28][2] = 32800
    expected_grid_map[28][26] = 32872
    expected_grid_map[28][27] = 4608
    expected_grid_map[28][40] = 72
    expected_grid_map[28][41] = 37408
    expected_grid_map[28][42] = 49186
    expected_grid_map[28][43] = 2064
    expected_grid_map[29][1] = 32800
    expected_grid_map[29][2] = 32800
    expected_grid_map[29][26] = 49186
    expected_grid_map[29][27] = 34864
    expected_grid_map[29][41] = 32872
    expected_grid_map[29][42] = 37408
    expected_grid_map[30][1] = 32800
    expected_grid_map[30][2] = 32800
    expected_grid_map[30][26] = 32800
    expected_grid_map[30][27] = 32800
    expected_grid_map[30][41] = 49186
    expected_grid_map[30][42] = 34864
    expected_grid_map[31][1] = 32800
    expected_grid_map[31][2] = 32800
    expected_grid_map[31][26] = 32800
    expected_grid_map[31][27] = 32800
    expected_grid_map[31][41] = 32800
    expected_grid_map[31][42] = 32800
    expected_grid_map[32][1] = 32800
    expected_grid_map[32][2] = 32800
    expected_grid_map[32][26] = 32800
    expected_grid_map[32][27] = 32800
    expected_grid_map[32][40] = 16386
    expected_grid_map[32][41] = 2064
    expected_grid_map[32][42] = 32800
    expected_grid_map[33][1] = 32800
    expected_grid_map[33][2] = 32800
    expected_grid_map[33][26] = 32872
    expected_grid_map[33][27] = 37408
    expected_grid_map[33][40] = 32800
    expected_grid_map[33][41] = 16386
    expected_grid_map[33][42] = 2064
    expected_grid_map[34][1] = 32800
    expected_grid_map[34][2] = 72
    expected_grid_map[34][3] = 4608
    expected_grid_map[34][26] = 49186
    expected_grid_map[34][27] = 2064
    expected_grid_map[34][40] = 32800
    expected_grid_map[34][41] = 32800
    expected_grid_map[35][1] = 32800
    expected_grid_map[35][3] = 32800
    expected_grid_map[35][26] = 32800
    expected_grid_map[35][40] = 32800
    expected_grid_map[35][41] = 32800
    expected_grid_map[36][1] = 32800
    expected_grid_map[36][3] = 32800
    expected_grid_map[36][26] = 32872
    expected_grid_map[36][27] = 1025
    expected_grid_map[36][28] = 4608
    expected_grid_map[36][40] = 32800
    expected_grid_map[36][41] = 32800
    expected_grid_map[37][1] = 32800
    expected_grid_map[37][3] = 32800
    expected_grid_map[37][26] = 72
    expected_grid_map[37][27] = 4608
    expected_grid_map[37][28] = 32800
    expected_grid_map[37][40] = 32800
    expected_grid_map[37][41] = 32800
    expected_grid_map[38][1] = 72
    expected_grid_map[38][2] = 1025
    expected_grid_map[38][3] = 1097
    expected_grid_map[38][4] = 1025
    expected_grid_map[38][5] = 5633
    expected_grid_map[38][6] = 17411
    expected_grid_map[38][7] = 1025
    expected_grid_map[38][8] = 1025
    expected_grid_map[38][9] = 1025
    expected_grid_map[38][10] = 5633
    expected_grid_map[38][11] = 17411
    expected_grid_map[38][12] = 1025
    expected_grid_map[38][13] = 5633
    expected_grid_map[38][14] = 5633
    expected_grid_map[38][15] = 4608
    expected_grid_map[38][27] = 32800
    expected_grid_map[38][28] = 72
    expected_grid_map[38][29] = 1025
    expected_grid_map[38][30] = 1025
    expected_grid_map[38][31] = 1025
    expected_grid_map[38][32] = 1025
    expected_grid_map[38][33] = 1025
    expected_grid_map[38][34] = 1025
    expected_grid_map[38][35] = 1025
    expected_grid_map[38][36] = 1025
    expected_grid_map[38][37] = 4608
    expected_grid_map[38][40] = 32800
    expected_grid_map[38][41] = 32800
    expected_grid_map[39][5] = 72
    expected_grid_map[39][6] = 3089
    expected_grid_map[39][7] = 1025
    expected_grid_map[39][8] = 1025
    expected_grid_map[39][9] = 1025
    expected_grid_map[39][10] = 1097
    expected_grid_map[39][11] = 2064
    expected_grid_map[39][13] = 32800
    expected_grid_map[39][14] = 32800
    expected_grid_map[39][15] = 72
    expected_grid_map[39][16] = 4608
    expected_grid_map[39][27] = 32800
    expected_grid_map[39][37] = 72
    expected_grid_map[39][38] = 1025
    expected_grid_map[39][39] = 4608
    expected_grid_map[39][40] = 32800
    expected_grid_map[39][41] = 32800
    expected_grid_map[40][13] = 32800
    expected_grid_map[40][14] = 32800
    expected_grid_map[40][16] = 32800
    expected_grid_map[40][27] = 32800
    expected_grid_map[40][39] = 32800
    expected_grid_map[40][40] = 32800
    expected_grid_map[40][41] = 32800
    expected_grid_map[41][13] = 32800
    expected_grid_map[41][14] = 72
    expected_grid_map[41][15] = 4608
    expected_grid_map[41][16] = 72
    expected_grid_map[41][17] = 4608
    expected_grid_map[41][21] = 16386
    expected_grid_map[41][22] = 1025
    expected_grid_map[41][23] = 4608
    expected_grid_map[41][27] = 32800
    expected_grid_map[41][39] = 32800
    expected_grid_map[41][40] = 32800
    expected_grid_map[41][41] = 32800
    expected_grid_map[42][13] = 72
    expected_grid_map[42][14] = 1025
    expected_grid_map[42][15] = 33825
    expected_grid_map[42][16] = 1025
    expected_grid_map[42][17] = 1097
    expected_grid_map[42][18] = 1025
    expected_grid_map[42][19] = 5633
    expected_grid_map[42][20] = 17411
    expected_grid_map[42][21] = 3089
    expected_grid_map[42][22] = 1025
    expected_grid_map[42][23] = 1097
    expected_grid_map[42][24] = 5633
    expected_grid_map[42][25] = 17411
    expected_grid_map[42][26] = 1025
    expected_grid_map[42][27] = 3089
    expected_grid_map[42][28] = 1025
    expected_grid_map[42][29] = 4608
    expected_grid_map[42][39] = 32800
    expected_grid_map[42][40] = 32800
    expected_grid_map[42][41] = 32800
    expected_grid_map[43][15] = 72
    expected_grid_map[43][16] = 1025
    expected_grid_map[43][17] = 1025
    expected_grid_map[43][18] = 1025
    expected_grid_map[43][19] = 1097
    expected_grid_map[43][20] = 3089
    expected_grid_map[43][21] = 5633
    expected_grid_map[43][22] = 1025
    expected_grid_map[43][23] = 17411
    expected_grid_map[43][24] = 1097
    expected_grid_map[43][25] = 3089
    expected_grid_map[43][26] = 1025
    expected_grid_map[43][27] = 5633
    expected_grid_map[43][28] = 1025
    expected_grid_map[43][29] = 37408
    expected_grid_map[43][39] = 32800
    expected_grid_map[43][40] = 32800
    expected_grid_map[43][41] = 32800
    expected_grid_map[44][21] = 72
    expected_grid_map[44][22] = 1025
    expected_grid_map[44][23] = 2064
    expected_grid_map[44][27] = 72
    expected_grid_map[44][28] = 1025
    expected_grid_map[44][29] = 1097
    expected_grid_map[44][30] = 1025
    expected_grid_map[44][31] = 5633
    expected_grid_map[44][32] = 17411
    expected_grid_map[44][33] = 1025
    expected_grid_map[44][34] = 1025
    expected_grid_map[44][35] = 1025
    expected_grid_map[44][36] = 5633
    expected_grid_map[44][37] = 17411
    expected_grid_map[44][38] = 1025
    expected_grid_map[44][39] = 3089
    expected_grid_map[44][40] = 3089
    expected_grid_map[44][41] = 2064
    expected_grid_map[45][31] = 72
    expected_grid_map[45][32] = 3089
    expected_grid_map[45][33] = 1025
    expected_grid_map[45][34] = 1025
    expected_grid_map[45][35] = 1025
    expected_grid_map[45][36] = 1097
    expected_grid_map[45][37] = 2064


def test_rail_env_malfunction_speed_info():
    env = RailEnv(width=50, height=50, rail_generator=sparse_rail_generator(max_num_cities=10,
                                                                            max_rails_between_cities=3,
                                                                            seed=5,
                                                                            grid_mode=False
                                                                            ),
                  line_generator=sparse_line_generator(), number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())
    env.reset(False, False, random_seed=42)

    env_renderer = RenderTool(env, gl="PILSVG", )
    for step in range(100):
        action_dict = dict()
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = np.random.choice(np.arange(4))
            action_dict.update({a: action})

        obs, rewards, done, info = env.step(action_dict)

        assert 'malfunction' in info
        for a in range(env.get_num_agents()):
            assert info['malfunction'][a] >= 0
            assert info['speed'][a] >= 0 and info['speed'][a] <= 1
            assert info['speed'][a] == env.agents[a].speed_counter.speed

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if done['__all__']:
            break
    env_renderer.close_window()


def test_sparse_generator_with_too_man_cities_does_not_break_down():
    RailEnv(width=50, height=50, rail_generator=sparse_rail_generator(
        max_num_cities=100,
        max_rails_between_cities=3,
        seed=5,
        grid_mode=False
    ), line_generator=sparse_line_generator(), number_of_agents=10, obs_builder_object=GlobalObsForRailEnv())


def test_sparse_generator_with_illegal_params_aborts():
    """
    Test that the constructor aborts if the initial parameters don't allow more than one city to be built.
    """
    with unittest.TestCase.assertRaises(test_sparse_generator_with_illegal_params_aborts, ValueError):
        RailEnv(width=6, height=6, rail_generator=sparse_rail_generator(
            max_num_cities=100,
            max_rails_between_cities=3,
            seed=5,
            grid_mode=False
        ), line_generator=sparse_line_generator(), number_of_agents=10,
                obs_builder_object=GlobalObsForRailEnv()).reset()

    with unittest.TestCase.assertRaises(test_sparse_generator_with_illegal_params_aborts, ValueError):
        RailEnv(width=60, height=60, rail_generator=sparse_rail_generator(
            max_num_cities=1,
            max_rails_between_cities=3,
            seed=5,
            grid_mode=False
        ), line_generator=sparse_line_generator(), number_of_agents=10,
                obs_builder_object=GlobalObsForRailEnv()).reset()


def test_sparse_generator_changes_to_grid_mode():
    """
        Test that grid mode is evoked and two cities are created when env is too small to find random cities.
    We set the limit of the env such that two cities fit in grid mode but unlikely under random mode
    we initiate random seed to be sure that we never create random cities.

    """
    rail_env = RailEnv(width=10, height=20, rail_generator=sparse_rail_generator(
        max_num_cities=100,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=1,
        seed=15,
        grid_mode=False
    ), line_generator=sparse_line_generator(), number_of_agents=10,
                       obs_builder_object=GlobalObsForRailEnv())

    # Catch warnings and check that a warning *IS* raised
    with warnings.catch_warnings(record=True) as w:
        rail_env.reset(True, True, random_seed=15)
        assert any("[WARNING]" in str(m.message) for m in w)


def test_sparse_generator_with_level_free_03():
    """Check that sparse generator generates level-free diamond-crossings."""

    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.,  # Fast freight train
                        1. / 3.: 0.,  # Slow commuter train
                        1. / 4.: 0.}  # Slow freight train

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=sparse_rail_generator(
                      max_num_cities=5,
                      max_rails_between_cities=3,
                      grid_mode=True,
                      p_level_free=0.3
                  ),
                  line_generator=sparse_line_generator(speed_ration_map),
                  number_of_agents=1,
                  random_seed=1)
    env.reset(random_seed=215545)
    assert env.resource_map.level_free_positions == set()


def test_sparse_generator_with_level_free_04():
    """Check that sparse generator generates level-free diamond crossings."""

    generate = sparse_rail_generator(max_num_cities=5, max_rails_between_cities=3, grid_mode=True, p_level_free=0.4)
    _, agent_hints = generate(width=25, height=30, num_agents=1, np_random=RandomState(215545))
    assert agent_hints["level_free_positions"] == {(12, 20)}


def test_sparse_generator_with_level_free_08():
    """Check that sparse generator generates level with diamond crossings."""

    generate = sparse_rail_generator(max_num_cities=5, max_rails_between_cities=3, grid_mode=True, p_level_free=0.8)
    _, agent_hints = generate(width=25, height=30, num_agents=1, np_random=RandomState(215545))
    assert agent_hints["level_free_positions"] == {(12, 20)}


def test_sparse_generator_with_level_free_09():
    """Check that sparse generator generates level with diamond crossings."""

    generate = sparse_rail_generator(max_num_cities=5, max_rails_between_cities=3, grid_mode=True, p_level_free=0.9)
    _, agent_hints = generate(width=25, height=30, num_agents=1, np_random=RandomState(215545))
    assert agent_hints["level_free_positions"] == {(12, 20), (4, 18)}


def test_sparse_generator_with_level_free_10():
    """Check that sparse generator generates all diamond crossings as level-free if p_level_free=1.0."""

    generate = sparse_rail_generator(max_num_cities=5, max_rails_between_cities=3, grid_mode=True, p_level_free=1.0)
    _, agent_hints = generate(width=25, height=30, num_agents=1, np_random=RandomState(215545))
    assert agent_hints["level_free_positions"] == {(12, 20), (4, 18)}
