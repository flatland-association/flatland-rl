import random

import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


def test_sparse_rail_generator():
    np.random.seed(0)
    random.seed(0)
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(max_num_cities=10,
                                                       max_rails_between_cities=3,
                                                       seed=5,
                                                       grid_mode=False
                                                       ),
                  schedule_generator=sparse_schedule_generator(),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv()
                  )
    env.reset(False, False, True)
    # for r in range(env.height):
    #     for c in range (env.width):
    #         if env.rail.grid[r][c] > 0:
    #             print("expected_grid_map[{}][{}] = {}".format(r,c,env.rail.grid[r][c]))
    expected_grid_map = np.zeros((50, 50), dtype=env.rail.transitions.get_type())
    expected_grid_map[4][32] = 16386
    expected_grid_map[4][33] = 1025
    expected_grid_map[4][34] = 1025
    expected_grid_map[4][35] = 1025
    expected_grid_map[4][36] = 1025
    expected_grid_map[4][37] = 1025
    expected_grid_map[4][38] = 1025
    expected_grid_map[4][39] = 1025
    expected_grid_map[4][40] = 1025
    expected_grid_map[4][41] = 1025
    expected_grid_map[4][42] = 1025
    expected_grid_map[4][43] = 5633
    expected_grid_map[4][44] = 5633
    expected_grid_map[4][45] = 4608
    expected_grid_map[5][32] = 32800
    expected_grid_map[5][43] = 32872
    expected_grid_map[5][44] = 38505
    expected_grid_map[5][45] = 37408
    expected_grid_map[6][32] = 32800
    expected_grid_map[6][43] = 32800
    expected_grid_map[6][44] = 32800
    expected_grid_map[6][45] = 32800
    expected_grid_map[7][32] = 32800
    expected_grid_map[7][43] = 32800
    expected_grid_map[7][44] = 32800
    expected_grid_map[7][45] = 32800
    expected_grid_map[8][32] = 32800
    expected_grid_map[8][43] = 32800
    expected_grid_map[8][44] = 32800
    expected_grid_map[9][32] = 32800
    expected_grid_map[9][43] = 32800
    expected_grid_map[9][44] = 32800
    expected_grid_map[9][45] = 32800
    expected_grid_map[10][18] = 16386
    expected_grid_map[10][19] = 1025
    expected_grid_map[10][20] = 1025
    expected_grid_map[10][21] = 1025
    expected_grid_map[10][22] = 1025
    expected_grid_map[10][23] = 17411
    expected_grid_map[10][24] = 1025
    expected_grid_map[10][25] = 1025
    expected_grid_map[10][26] = 1025
    expected_grid_map[10][27] = 1025
    expected_grid_map[10][28] = 1025
    expected_grid_map[10][29] = 1025
    expected_grid_map[10][30] = 1025
    expected_grid_map[10][31] = 5633
    expected_grid_map[10][32] = 1025
    expected_grid_map[10][33] = 1025
    expected_grid_map[10][34] = 1025
    expected_grid_map[10][35] = 1025
    expected_grid_map[10][36] = 1025
    expected_grid_map[10][37] = 1025
    expected_grid_map[10][38] = 4608
    expected_grid_map[10][41] = 32800
    expected_grid_map[10][42] = 32800
    expected_grid_map[10][43] = 32800
    expected_grid_map[10][44] = 32800
    expected_grid_map[10][45] = 32800

    expected_grid_map[11][18] = 49186
    expected_grid_map[11][19] = 1025
    expected_grid_map[11][20] = 1025
    expected_grid_map[11][21] = 1025
    expected_grid_map[11][22] = 17411
    expected_grid_map[11][23] = 1097
    expected_grid_map[11][24] = 1025
    expected_grid_map[11][25] = 1025
    expected_grid_map[11][26] = 1025
    expected_grid_map[11][27] = 1025
    expected_grid_map[11][28] = 1025
    expected_grid_map[11][29] = 1025
    expected_grid_map[11][30] = 1025
    expected_grid_map[11][31] = 3089
    expected_grid_map[11][32] = 1025
    expected_grid_map[11][33] = 1025
    expected_grid_map[11][34] = 1025
    expected_grid_map[11][35] = 1025
    expected_grid_map[11][36] = 1025
    expected_grid_map[11][37] = 1025
    expected_grid_map[11][38] = 37408
    expected_grid_map[11][41] = 32800
    expected_grid_map[11][42] = 32800
    expected_grid_map[11][43] = 32800
    expected_grid_map[11][44] = 32800
    expected_grid_map[12][18] = 32800
    expected_grid_map[12][22] = 32800
    expected_grid_map[12][38] = 32800
    expected_grid_map[12][41] = 32800
    expected_grid_map[12][42] = 32800
    expected_grid_map[12][43] = 32800
    expected_grid_map[12][44] = 32800
    expected_grid_map[13][6] = 16386
    expected_grid_map[13][7] = 1025
    expected_grid_map[13][8] = 1025
    expected_grid_map[13][9] = 17411
    expected_grid_map[13][10] = 1025
    expected_grid_map[13][11] = 1025
    expected_grid_map[13][12] = 1025
    expected_grid_map[13][13] = 1025
    expected_grid_map[13][14] = 1025
    expected_grid_map[13][15] = 1025
    expected_grid_map[13][16] = 1025
    expected_grid_map[13][17] = 17411
    expected_grid_map[13][18] = 3089
    expected_grid_map[13][19] = 1025
    expected_grid_map[13][20] = 1025
    expected_grid_map[13][21] = 1025
    expected_grid_map[13][22] = 34864
    expected_grid_map[13][38] = 32800
    expected_grid_map[13][41] = 72
    expected_grid_map[13][42] = 38505
    expected_grid_map[13][43] = 52275
    expected_grid_map[13][44] = 34864
    expected_grid_map[14][6] = 49186
    expected_grid_map[14][7] = 1025
    expected_grid_map[14][8] = 1025
    expected_grid_map[14][9] = 38505
    expected_grid_map[14][10] = 1025
    expected_grid_map[14][11] = 1025
    expected_grid_map[14][12] = 1025
    expected_grid_map[14][13] = 1025
    expected_grid_map[14][14] = 1025
    expected_grid_map[14][15] = 1025
    expected_grid_map[14][16] = 1025
    expected_grid_map[14][17] = 52275
    expected_grid_map[14][18] = 1025
    expected_grid_map[14][19] = 1025
    expected_grid_map[14][20] = 1025
    expected_grid_map[14][21] = 1025
    expected_grid_map[14][22] = 34864
    expected_grid_map[14][38] = 72
    expected_grid_map[14][39] = 1025
    expected_grid_map[14][40] = 1025
    expected_grid_map[14][41] = 1025
    expected_grid_map[14][42] = 2136
    expected_grid_map[14][43] = 37408
    expected_grid_map[14][44] = 32800
    expected_grid_map[15][6] = 49186
    expected_grid_map[15][7] = 1025
    expected_grid_map[15][8] = 17411
    expected_grid_map[15][9] = 1097
    expected_grid_map[15][10] = 1025
    expected_grid_map[15][11] = 1025
    expected_grid_map[15][12] = 1025
    expected_grid_map[15][13] = 1025
    expected_grid_map[15][14] = 1025
    expected_grid_map[15][15] = 1025
    expected_grid_map[15][16] = 1025
    expected_grid_map[15][17] = 3089
    expected_grid_map[15][18] = 1025
    expected_grid_map[15][19] = 1025
    expected_grid_map[15][20] = 1025
    expected_grid_map[15][21] = 1025
    expected_grid_map[15][22] = 38505
    expected_grid_map[15][23] = 4608
    expected_grid_map[15][43] = 32800
    expected_grid_map[15][44] = 32800
    expected_grid_map[16][6] = 32800
    expected_grid_map[16][8] = 32800
    expected_grid_map[16][22] = 32800
    expected_grid_map[16][23] = 32800
    expected_grid_map[16][43] = 32800
    expected_grid_map[16][44] = 32800
    expected_grid_map[17][6] = 32800
    expected_grid_map[17][8] = 32800
    expected_grid_map[17][22] = 32800
    expected_grid_map[17][23] = 32800
    expected_grid_map[17][43] = 32800
    expected_grid_map[17][44] = 32800
    expected_grid_map[18][6] = 32800
    expected_grid_map[18][8] = 32800
    expected_grid_map[18][22] = 32800
    expected_grid_map[18][23] = 32800
    expected_grid_map[18][43] = 32800
    expected_grid_map[18][44] = 32872
    expected_grid_map[18][45] = 4608
    expected_grid_map[19][6] = 32800
    expected_grid_map[19][8] = 32800
    expected_grid_map[19][22] = 32800
    expected_grid_map[19][23] = 32800
    expected_grid_map[19][42] = 16386
    expected_grid_map[19][43] = 52275
    expected_grid_map[19][44] = 38505
    expected_grid_map[19][45] = 34864
    expected_grid_map[20][4] = 16386
    expected_grid_map[20][5] = 17411
    expected_grid_map[20][6] = 50211
    expected_grid_map[20][7] = 1025
    expected_grid_map[20][8] = 2064
    expected_grid_map[20][22] = 32800
    expected_grid_map[20][23] = 32800
    expected_grid_map[20][42] = 32800
    expected_grid_map[20][43] = 32800
    expected_grid_map[20][44] = 32800
    expected_grid_map[20][45] = 32800
    expected_grid_map[21][3] = 16386
    expected_grid_map[21][4] = 38505
    expected_grid_map[21][5] = 52275
    expected_grid_map[21][6] = 37408
    expected_grid_map[21][22] = 32800
    expected_grid_map[21][23] = 32800
    expected_grid_map[21][42] = 32800
    expected_grid_map[21][43] = 32800
    expected_grid_map[21][44] = 32800
    expected_grid_map[21][45] = 32800
    expected_grid_map[22][3] = 32800
    expected_grid_map[22][4] = 32800
    expected_grid_map[22][5] = 32800
    expected_grid_map[22][6] = 32800
    expected_grid_map[22][21] = 16386
    expected_grid_map[22][22] = 34864
    expected_grid_map[22][23] = 32872
    expected_grid_map[22][24] = 4608
    expected_grid_map[22][42] = 32800
    expected_grid_map[22][43] = 32800
    expected_grid_map[22][44] = 32800
    expected_grid_map[22][45] = 32800
    expected_grid_map[23][3] = 32800
    expected_grid_map[23][4] = 32800
    expected_grid_map[23][5] = 32800
    expected_grid_map[23][6] = 32800
    expected_grid_map[23][21] = 32872
    expected_grid_map[23][22] = 38505
    expected_grid_map[23][23] = 38505
    expected_grid_map[23][24] = 37408
    expected_grid_map[23][42] = 32800
    expected_grid_map[23][43] = 32800
    expected_grid_map[23][44] = 32800
    expected_grid_map[23][45] = 32800
    expected_grid_map[24][3] = 32800
    expected_grid_map[24][4] = 32800
    expected_grid_map[24][5] = 32800
    expected_grid_map[24][6] = 32800
    expected_grid_map[24][21] = 32800
    expected_grid_map[24][22] = 32800
    expected_grid_map[24][23] = 32800
    expected_grid_map[24][24] = 32800
    expected_grid_map[24][42] = 32800
    expected_grid_map[24][43] = 32800
    expected_grid_map[24][44] = 32800
    expected_grid_map[24][45] = 32800
    expected_grid_map[25][3] = 32800
    expected_grid_map[25][4] = 32800
    expected_grid_map[25][5] = 32800
    expected_grid_map[25][6] = 32800
    expected_grid_map[25][21] = 32800
    expected_grid_map[25][22] = 32800
    expected_grid_map[25][23] = 32800
    expected_grid_map[25][24] = 32800
    expected_grid_map[25][42] = 32800
    expected_grid_map[25][43] = 32800
    expected_grid_map[25][44] = 32800
    expected_grid_map[25][45] = 32800
    expected_grid_map[26][3] = 32800
    expected_grid_map[26][4] = 32800
    expected_grid_map[26][5] = 32800
    expected_grid_map[26][6] = 32800
    expected_grid_map[26][21] = 32800
    expected_grid_map[26][22] = 32800
    expected_grid_map[26][23] = 32800
    expected_grid_map[26][24] = 32800
    expected_grid_map[26][42] = 32800
    expected_grid_map[26][43] = 32800
    expected_grid_map[26][44] = 32800
    expected_grid_map[26][45] = 32800
    expected_grid_map[27][3] = 32800
    expected_grid_map[27][4] = 32800
    expected_grid_map[27][5] = 32800
    expected_grid_map[27][6] = 32800
    expected_grid_map[27][21] = 32800
    expected_grid_map[27][22] = 32800
    expected_grid_map[27][23] = 32800
    expected_grid_map[27][24] = 32800
    expected_grid_map[27][42] = 72
    expected_grid_map[27][43] = 52275
    expected_grid_map[27][44] = 38505
    expected_grid_map[27][45] = 34864
    expected_grid_map[28][3] = 32800
    expected_grid_map[28][4] = 32800
    expected_grid_map[28][5] = 32800
    expected_grid_map[28][6] = 32800
    expected_grid_map[28][21] = 32800
    expected_grid_map[28][22] = 32800
    expected_grid_map[28][23] = 32800
    expected_grid_map[28][24] = 32800
    expected_grid_map[28][40] = 16386
    expected_grid_map[28][41] = 1025
    expected_grid_map[28][42] = 1025
    expected_grid_map[28][43] = 2064
    expected_grid_map[28][44] = 72
    expected_grid_map[28][45] = 33897
    expected_grid_map[28][46] = 1025
    expected_grid_map[28][47] = 1025
    expected_grid_map[28][48] = 4608
    expected_grid_map[29][3] = 72
    expected_grid_map[29][4] = 52275
    expected_grid_map[29][5] = 52275
    expected_grid_map[29][6] = 37408
    expected_grid_map[29][21] = 32800
    expected_grid_map[29][22] = 32800
    expected_grid_map[29][23] = 32800
    expected_grid_map[29][24] = 32800
    expected_grid_map[29][40] = 32800
    expected_grid_map[29][45] = 32800
    expected_grid_map[29][48] = 32800
    expected_grid_map[30][4] = 72
    expected_grid_map[30][5] = 1097
    expected_grid_map[30][6] = 37408
    expected_grid_map[30][21] = 32800
    expected_grid_map[30][22] = 32800
    expected_grid_map[30][23] = 32800
    expected_grid_map[30][24] = 32800
    expected_grid_map[30][40] = 32800
    expected_grid_map[30][45] = 32800
    expected_grid_map[30][48] = 32800
    expected_grid_map[31][6] = 32872
    expected_grid_map[31][7] = 5633
    expected_grid_map[31][8] = 4608
    expected_grid_map[31][21] = 49186
    expected_grid_map[31][22] = 52275
    expected_grid_map[31][23] = 38505
    expected_grid_map[31][24] = 34864
    expected_grid_map[31][40] = 32800
    expected_grid_map[31][45] = 32800
    expected_grid_map[31][48] = 32800
    expected_grid_map[32][6] = 32872
    expected_grid_map[32][7] = 52275
    expected_grid_map[32][8] = 37408
    expected_grid_map[32][21] = 72
    expected_grid_map[32][22] = 1097
    expected_grid_map[32][23] = 37408
    expected_grid_map[32][24] = 32800
    expected_grid_map[32][40] = 32800
    expected_grid_map[32][45] = 32800
    expected_grid_map[32][48] = 32800
    expected_grid_map[33][6] = 32800
    expected_grid_map[33][7] = 32800
    expected_grid_map[33][8] = 32800
    expected_grid_map[33][23] = 32800
    expected_grid_map[33][24] = 32800
    expected_grid_map[33][40] = 32800
    expected_grid_map[33][45] = 32800
    expected_grid_map[33][48] = 32800
    expected_grid_map[34][6] = 32800
    expected_grid_map[34][7] = 32800
    expected_grid_map[34][8] = 32800
    expected_grid_map[34][23] = 32800
    expected_grid_map[34][24] = 32800
    expected_grid_map[34][40] = 32800
    expected_grid_map[34][45] = 32800
    expected_grid_map[34][48] = 32800
    expected_grid_map[35][6] = 32800
    expected_grid_map[35][7] = 32800
    expected_grid_map[35][8] = 32800
    expected_grid_map[35][23] = 72
    expected_grid_map[35][24] = 37408
    expected_grid_map[35][40] = 32800
    expected_grid_map[35][45] = 32800
    expected_grid_map[35][48] = 32800
    expected_grid_map[36][6] = 32800
    expected_grid_map[36][7] = 32800
    expected_grid_map[36][8] = 32800
    expected_grid_map[36][24] = 32800
    expected_grid_map[36][39] = 16386
    expected_grid_map[36][40] = 2064
    expected_grid_map[36][45] = 72
    expected_grid_map[36][46] = 1025
    expected_grid_map[36][47] = 1025
    expected_grid_map[36][48] = 1097
    expected_grid_map[36][49] = 4608
    expected_grid_map[37][6] = 32800
    expected_grid_map[37][7] = 32800
    expected_grid_map[37][8] = 32800
    expected_grid_map[37][24] = 32800
    expected_grid_map[37][39] = 32800
    expected_grid_map[37][49] = 32800
    expected_grid_map[38][6] = 32800
    expected_grid_map[38][7] = 32800
    expected_grid_map[38][8] = 32800
    expected_grid_map[38][12] = 16386
    expected_grid_map[38][13] = 1025
    expected_grid_map[38][14] = 1025
    expected_grid_map[38][15] = 5633
    expected_grid_map[38][16] = 1025
    expected_grid_map[38][17] = 1025
    expected_grid_map[38][18] = 1025
    expected_grid_map[38][19] = 1025
    expected_grid_map[38][20] = 1025
    expected_grid_map[38][21] = 1025
    expected_grid_map[38][22] = 1025
    expected_grid_map[38][23] = 17411
    expected_grid_map[38][24] = 3089
    expected_grid_map[38][25] = 1025
    expected_grid_map[38][26] = 1025
    expected_grid_map[38][27] = 4608
    expected_grid_map[38][39] = 32800
    expected_grid_map[38][49] = 32800
    expected_grid_map[39][6] = 32800
    expected_grid_map[39][7] = 32800
    expected_grid_map[39][8] = 32800
    expected_grid_map[39][12] = 49186
    expected_grid_map[39][13] = 1025
    expected_grid_map[39][14] = 1025
    expected_grid_map[39][15] = 52275
    expected_grid_map[39][16] = 1025
    expected_grid_map[39][17] = 1025
    expected_grid_map[39][18] = 1025
    expected_grid_map[39][19] = 1025
    expected_grid_map[39][20] = 1025
    expected_grid_map[39][21] = 1025
    expected_grid_map[39][22] = 1025
    expected_grid_map[39][23] = 52275
    expected_grid_map[39][24] = 1025
    expected_grid_map[39][25] = 1025
    expected_grid_map[39][26] = 1025
    expected_grid_map[39][27] = 37408
    expected_grid_map[39][39] = 32800
    expected_grid_map[39][49] = 32800
    expected_grid_map[40][6] = 32872
    expected_grid_map[40][7] = 38505
    expected_grid_map[40][8] = 37408
    expected_grid_map[40][12] = 49186
    expected_grid_map[40][13] = 1025
    expected_grid_map[40][14] = 17411
    expected_grid_map[40][15] = 52275
    expected_grid_map[40][16] = 1025
    expected_grid_map[40][17] = 1025
    expected_grid_map[40][18] = 1025
    expected_grid_map[40][19] = 1025
    expected_grid_map[40][20] = 1025
    expected_grid_map[40][21] = 1025
    expected_grid_map[40][22] = 1025
    expected_grid_map[40][23] = 38505
    expected_grid_map[40][24] = 1025
    expected_grid_map[40][25] = 1025
    expected_grid_map[40][26] = 1025
    expected_grid_map[40][27] = 1097
    expected_grid_map[40][28] = 17411
    expected_grid_map[40][29] = 1025
    expected_grid_map[40][30] = 1025
    expected_grid_map[40][31] = 1025
    expected_grid_map[40][32] = 1025
    expected_grid_map[40][33] = 1025
    expected_grid_map[40][34] = 1025
    expected_grid_map[40][35] = 1025
    expected_grid_map[40][36] = 17411
    expected_grid_map[40][37] = 1025
    expected_grid_map[40][38] = 1025
    expected_grid_map[40][39] = 1097
    expected_grid_map[40][40] = 5633
    expected_grid_map[40][41] = 1025
    expected_grid_map[40][42] = 1025
    expected_grid_map[40][43] = 1025
    expected_grid_map[40][44] = 1025
    expected_grid_map[40][45] = 1025
    expected_grid_map[40][46] = 1025
    expected_grid_map[40][47] = 1025
    expected_grid_map[40][48] = 5633
    expected_grid_map[40][49] = 34864
    expected_grid_map[41][6] = 72
    expected_grid_map[41][7] = 1097
    expected_grid_map[41][8] = 1097
    expected_grid_map[41][9] = 1025
    expected_grid_map[41][10] = 1025
    expected_grid_map[41][11] = 1025
    expected_grid_map[41][12] = 3089
    expected_grid_map[41][13] = 1025
    expected_grid_map[41][14] = 3089
    expected_grid_map[41][15] = 3089
    expected_grid_map[41][16] = 1025
    expected_grid_map[41][17] = 1025
    expected_grid_map[41][18] = 1025
    expected_grid_map[41][19] = 1025
    expected_grid_map[41][20] = 1025
    expected_grid_map[41][21] = 1025
    expected_grid_map[41][22] = 1025
    expected_grid_map[41][23] = 3089
    expected_grid_map[41][24] = 5633
    expected_grid_map[41][25] = 1025
    expected_grid_map[41][26] = 1025
    expected_grid_map[41][27] = 1025
    expected_grid_map[41][28] = 38505
    expected_grid_map[41][29] = 1025
    expected_grid_map[41][30] = 1025
    expected_grid_map[41][31] = 1025
    expected_grid_map[41][32] = 1025
    expected_grid_map[41][33] = 1025
    expected_grid_map[41][34] = 1025
    expected_grid_map[41][35] = 1025
    expected_grid_map[41][36] = 52275
    expected_grid_map[41][37] = 1025
    expected_grid_map[41][38] = 1025
    expected_grid_map[41][39] = 1025
    expected_grid_map[41][40] = 38505
    expected_grid_map[41][41] = 1025
    expected_grid_map[41][42] = 1025
    expected_grid_map[41][43] = 1025
    expected_grid_map[41][44] = 1025
    expected_grid_map[41][45] = 1025
    expected_grid_map[41][46] = 1025
    expected_grid_map[41][47] = 1025
    expected_grid_map[41][48] = 52275
    expected_grid_map[41][49] = 34864
    expected_grid_map[42][24] = 72
    expected_grid_map[42][25] = 1025
    expected_grid_map[42][26] = 1025
    expected_grid_map[42][27] = 1025
    expected_grid_map[42][28] = 1097
    expected_grid_map[42][29] = 1025
    expected_grid_map[42][30] = 1025
    expected_grid_map[42][31] = 1025
    expected_grid_map[42][32] = 1025
    expected_grid_map[42][33] = 1025
    expected_grid_map[42][34] = 1025
    expected_grid_map[42][35] = 1025
    expected_grid_map[42][36] = 1097
    expected_grid_map[42][37] = 1025
    expected_grid_map[42][38] = 1025
    expected_grid_map[42][39] = 1025
    expected_grid_map[42][40] = 3089
    expected_grid_map[42][41] = 1025
    expected_grid_map[42][42] = 1025
    expected_grid_map[42][43] = 1025
    expected_grid_map[42][44] = 1025
    expected_grid_map[42][45] = 1025
    expected_grid_map[42][46] = 1025
    expected_grid_map[42][47] = 1025
    expected_grid_map[42][48] = 1097
    expected_grid_map[42][49] = 2064

    # Attention, once we have fixed the generator this needs to be changed!!!!
    expected_grid_map = env.rail.grid
    assert np.array_equal(env.rail.grid, expected_grid_map), "actual={}, expected={}".format(env.rail.grid,
                                                                                             expected_grid_map)
    s0 = 0
    s1 = 0
    for a in range(env.get_num_agents()):
        s0 = Vec2d.get_manhattan_distance(env.agents[a].initial_position, (0, 0))
        s1 = Vec2d.get_chebyshev_distance(env.agents[a].initial_position, (0, 0))
    assert s0 == 31, "actual={}".format(s0)
    assert s1 == 24, "actual={}".format(s1)


def test_sparse_rail_generator_deterministic():
    """Check that sparse_rail_generator runs deterministic over different python versions!"""
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
                                                       max_rails_between_cities=3,
                                                       seed=215545,  # Random seed
                                                       grid_mode=True
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    for r in range(env.height):
        for c in range(env.width):
            print("assert env.rail.get_full_transitions({}, {}) == {}, \"[{}][{}]\"".format(r, c,
                                                                                            env.rail.get_full_transitions(
                                                                                                r, c), r, c))
    assert env.rail.get_full_transitions(0, 1) == 0, "[0][1]"
    assert env.rail.get_full_transitions(0, 2) == 0, "[0][2]"
    assert env.rail.get_full_transitions(0, 3) == 0, "[0][3]"
    assert env.rail.get_full_transitions(0, 4) == 0, "[0][4]"
    assert env.rail.get_full_transitions(0, 5) == 0, "[0][5]"
    assert env.rail.get_full_transitions(0, 6) == 0, "[0][6]"
    assert env.rail.get_full_transitions(0, 7) == 0, "[0][7]"
    assert env.rail.get_full_transitions(0, 8) == 0, "[0][8]"
    assert env.rail.get_full_transitions(0, 9) == 0, "[0][9]"
    assert env.rail.get_full_transitions(0, 10) == 0, "[0][10]"
    assert env.rail.get_full_transitions(0, 11) == 0, "[0][11]"
    assert env.rail.get_full_transitions(0, 12) == 0, "[0][12]"
    assert env.rail.get_full_transitions(0, 13) == 0, "[0][13]"
    assert env.rail.get_full_transitions(0, 14) == 0, "[0][14]"
    assert env.rail.get_full_transitions(0, 15) == 0, "[0][15]"
    assert env.rail.get_full_transitions(0, 16) == 0, "[0][16]"
    assert env.rail.get_full_transitions(0, 17) == 0, "[0][17]"
    assert env.rail.get_full_transitions(0, 18) == 0, "[0][18]"
    assert env.rail.get_full_transitions(0, 19) == 0, "[0][19]"
    assert env.rail.get_full_transitions(0, 20) == 0, "[0][20]"
    assert env.rail.get_full_transitions(0, 21) == 0, "[0][21]"
    assert env.rail.get_full_transitions(0, 22) == 0, "[0][22]"
    assert env.rail.get_full_transitions(0, 23) == 0, "[0][23]"
    assert env.rail.get_full_transitions(0, 24) == 0, "[0][24]"
    assert env.rail.get_full_transitions(1, 0) == 0, "[1][0]"
    assert env.rail.get_full_transitions(1, 1) == 0, "[1][1]"
    assert env.rail.get_full_transitions(1, 2) == 0, "[1][2]"
    assert env.rail.get_full_transitions(1, 3) == 0, "[1][3]"
    assert env.rail.get_full_transitions(1, 4) == 0, "[1][4]"
    assert env.rail.get_full_transitions(1, 5) == 0, "[1][5]"
    assert env.rail.get_full_transitions(1, 6) == 0, "[1][6]"
    assert env.rail.get_full_transitions(1, 7) == 0, "[1][7]"
    assert env.rail.get_full_transitions(1, 8) == 0, "[1][8]"
    assert env.rail.get_full_transitions(1, 9) == 0, "[1][9]"
    assert env.rail.get_full_transitions(1, 10) == 0, "[1][10]"
    assert env.rail.get_full_transitions(1, 11) == 0, "[1][11]"
    assert env.rail.get_full_transitions(1, 12) == 0, "[1][12]"
    assert env.rail.get_full_transitions(1, 13) == 0, "[1][13]"
    assert env.rail.get_full_transitions(1, 14) == 0, "[1][14]"
    assert env.rail.get_full_transitions(1, 15) == 0, "[1][15]"
    assert env.rail.get_full_transitions(1, 16) == 0, "[1][16]"
    assert env.rail.get_full_transitions(1, 17) == 0, "[1][17]"
    assert env.rail.get_full_transitions(1, 18) == 0, "[1][18]"
    assert env.rail.get_full_transitions(1, 19) == 0, "[1][19]"
    assert env.rail.get_full_transitions(1, 20) == 0, "[1][20]"
    assert env.rail.get_full_transitions(1, 21) == 0, "[1][21]"
    assert env.rail.get_full_transitions(1, 22) == 0, "[1][22]"
    assert env.rail.get_full_transitions(1, 23) == 0, "[1][23]"
    assert env.rail.get_full_transitions(1, 24) == 0, "[1][24]"
    assert env.rail.get_full_transitions(2, 0) == 0, "[2][0]"
    assert env.rail.get_full_transitions(2, 1) == 0, "[2][1]"
    assert env.rail.get_full_transitions(2, 2) == 0, "[2][2]"
    assert env.rail.get_full_transitions(2, 3) == 0, "[2][3]"
    assert env.rail.get_full_transitions(2, 4) == 0, "[2][4]"
    assert env.rail.get_full_transitions(2, 5) == 0, "[2][5]"
    assert env.rail.get_full_transitions(2, 6) == 0, "[2][6]"
    assert env.rail.get_full_transitions(2, 7) == 0, "[2][7]"
    assert env.rail.get_full_transitions(2, 8) == 0, "[2][8]"
    assert env.rail.get_full_transitions(2, 9) == 0, "[2][9]"
    assert env.rail.get_full_transitions(2, 10) == 0, "[2][10]"
    assert env.rail.get_full_transitions(2, 11) == 0, "[2][11]"
    assert env.rail.get_full_transitions(2, 12) == 0, "[2][12]"
    assert env.rail.get_full_transitions(2, 13) == 0, "[2][13]"
    assert env.rail.get_full_transitions(2, 14) == 0, "[2][14]"
    assert env.rail.get_full_transitions(2, 15) == 0, "[2][15]"
    assert env.rail.get_full_transitions(2, 16) == 0, "[2][16]"
    assert env.rail.get_full_transitions(2, 17) == 0, "[2][17]"
    assert env.rail.get_full_transitions(2, 18) == 0, "[2][18]"
    assert env.rail.get_full_transitions(2, 19) == 0, "[2][19]"
    assert env.rail.get_full_transitions(2, 20) == 0, "[2][20]"
    assert env.rail.get_full_transitions(2, 21) == 0, "[2][21]"
    assert env.rail.get_full_transitions(2, 22) == 0, "[2][22]"
    assert env.rail.get_full_transitions(2, 23) == 0, "[2][23]"
    assert env.rail.get_full_transitions(2, 24) == 0, "[2][24]"
    assert env.rail.get_full_transitions(3, 0) == 0, "[3][0]"
    assert env.rail.get_full_transitions(3, 1) == 0, "[3][1]"
    assert env.rail.get_full_transitions(3, 2) == 0, "[3][2]"
    assert env.rail.get_full_transitions(3, 3) == 0, "[3][3]"
    assert env.rail.get_full_transitions(3, 4) == 0, "[3][4]"
    assert env.rail.get_full_transitions(3, 5) == 0, "[3][5]"
    assert env.rail.get_full_transitions(3, 6) == 0, "[3][6]"
    assert env.rail.get_full_transitions(3, 7) == 0, "[3][7]"
    assert env.rail.get_full_transitions(3, 8) == 0, "[3][8]"
    assert env.rail.get_full_transitions(3, 9) == 0, "[3][9]"
    assert env.rail.get_full_transitions(3, 10) == 0, "[3][10]"
    assert env.rail.get_full_transitions(3, 11) == 0, "[3][11]"
    assert env.rail.get_full_transitions(3, 12) == 0, "[3][12]"
    assert env.rail.get_full_transitions(3, 13) == 0, "[3][13]"
    assert env.rail.get_full_transitions(3, 14) == 0, "[3][14]"
    assert env.rail.get_full_transitions(3, 15) == 0, "[3][15]"
    assert env.rail.get_full_transitions(3, 16) == 0, "[3][16]"
    assert env.rail.get_full_transitions(3, 17) == 0, "[3][17]"
    assert env.rail.get_full_transitions(3, 18) == 0, "[3][18]"
    assert env.rail.get_full_transitions(3, 19) == 0, "[3][19]"
    assert env.rail.get_full_transitions(3, 20) == 0, "[3][20]"
    assert env.rail.get_full_transitions(3, 21) == 0, "[3][21]"
    assert env.rail.get_full_transitions(3, 22) == 0, "[3][22]"
    assert env.rail.get_full_transitions(3, 23) == 0, "[3][23]"
    assert env.rail.get_full_transitions(3, 24) == 0, "[3][24]"
    assert env.rail.get_full_transitions(4, 0) == 0, "[4][0]"
    assert env.rail.get_full_transitions(4, 1) == 0, "[4][1]"
    assert env.rail.get_full_transitions(4, 2) == 0, "[4][2]"
    assert env.rail.get_full_transitions(4, 3) == 0, "[4][3]"
    assert env.rail.get_full_transitions(4, 4) == 0, "[4][4]"
    assert env.rail.get_full_transitions(4, 5) == 0, "[4][5]"
    assert env.rail.get_full_transitions(4, 6) == 0, "[4][6]"
    assert env.rail.get_full_transitions(4, 7) == 0, "[4][7]"
    assert env.rail.get_full_transitions(4, 8) == 0, "[4][8]"
    assert env.rail.get_full_transitions(4, 9) == 0, "[4][9]"
    assert env.rail.get_full_transitions(4, 10) == 0, "[4][10]"
    assert env.rail.get_full_transitions(4, 11) == 0, "[4][11]"
    assert env.rail.get_full_transitions(4, 12) == 0, "[4][12]"
    assert env.rail.get_full_transitions(4, 13) == 0, "[4][13]"
    assert env.rail.get_full_transitions(4, 14) == 0, "[4][14]"
    assert env.rail.get_full_transitions(4, 15) == 0, "[4][15]"
    assert env.rail.get_full_transitions(4, 16) == 0, "[4][16]"
    assert env.rail.get_full_transitions(4, 17) == 0, "[4][17]"
    assert env.rail.get_full_transitions(4, 18) == 0, "[4][18]"
    assert env.rail.get_full_transitions(4, 19) == 0, "[4][19]"
    assert env.rail.get_full_transitions(4, 20) == 0, "[4][20]"
    assert env.rail.get_full_transitions(4, 21) == 0, "[4][21]"
    assert env.rail.get_full_transitions(4, 22) == 0, "[4][22]"
    assert env.rail.get_full_transitions(4, 23) == 0, "[4][23]"
    assert env.rail.get_full_transitions(4, 24) == 0, "[4][24]"
    assert env.rail.get_full_transitions(5, 0) == 0, "[5][0]"
    assert env.rail.get_full_transitions(5, 1) == 0, "[5][1]"
    assert env.rail.get_full_transitions(5, 2) == 0, "[5][2]"
    assert env.rail.get_full_transitions(5, 3) == 0, "[5][3]"
    assert env.rail.get_full_transitions(5, 4) == 0, "[5][4]"
    assert env.rail.get_full_transitions(5, 5) == 0, "[5][5]"
    assert env.rail.get_full_transitions(5, 6) == 0, "[5][6]"
    assert env.rail.get_full_transitions(5, 7) == 0, "[5][7]"
    assert env.rail.get_full_transitions(5, 8) == 0, "[5][8]"
    assert env.rail.get_full_transitions(5, 9) == 16386, "[5][9]"
    assert env.rail.get_full_transitions(5, 10) == 1025, "[5][10]"
    assert env.rail.get_full_transitions(5, 11) == 1025, "[5][11]"
    assert env.rail.get_full_transitions(5, 12) == 1025, "[5][12]"
    assert env.rail.get_full_transitions(5, 13) == 1025, "[5][13]"
    assert env.rail.get_full_transitions(5, 14) == 1025, "[5][14]"
    assert env.rail.get_full_transitions(5, 15) == 1025, "[5][15]"
    assert env.rail.get_full_transitions(5, 16) == 1025, "[5][16]"
    assert env.rail.get_full_transitions(5, 17) == 4608, "[5][17]"
    assert env.rail.get_full_transitions(5, 18) == 0, "[5][18]"
    assert env.rail.get_full_transitions(5, 19) == 0, "[5][19]"
    assert env.rail.get_full_transitions(5, 20) == 0, "[5][20]"
    assert env.rail.get_full_transitions(5, 21) == 0, "[5][21]"
    assert env.rail.get_full_transitions(5, 22) == 0, "[5][22]"
    assert env.rail.get_full_transitions(5, 23) == 0, "[5][23]"
    assert env.rail.get_full_transitions(5, 24) == 0, "[5][24]"
    assert env.rail.get_full_transitions(6, 0) == 16386, "[6][0]"
    assert env.rail.get_full_transitions(6, 1) == 1025, "[6][1]"
    assert env.rail.get_full_transitions(6, 2) == 1025, "[6][2]"
    assert env.rail.get_full_transitions(6, 3) == 5633, "[6][3]"
    assert env.rail.get_full_transitions(6, 4) == 1025, "[6][4]"
    assert env.rail.get_full_transitions(6, 5) == 1025, "[6][5]"
    assert env.rail.get_full_transitions(6, 6) == 1025, "[6][6]"
    assert env.rail.get_full_transitions(6, 7) == 1025, "[6][7]"
    assert env.rail.get_full_transitions(6, 8) == 1025, "[6][8]"
    assert env.rail.get_full_transitions(6, 9) == 52275, "[6][9]"
    assert env.rail.get_full_transitions(6, 10) == 1025, "[6][10]"
    assert env.rail.get_full_transitions(6, 11) == 1025, "[6][11]"
    assert env.rail.get_full_transitions(6, 12) == 1025, "[6][12]"
    assert env.rail.get_full_transitions(6, 13) == 1025, "[6][13]"
    assert env.rail.get_full_transitions(6, 14) == 1025, "[6][14]"
    assert env.rail.get_full_transitions(6, 15) == 1025, "[6][15]"
    assert env.rail.get_full_transitions(6, 16) == 17411, "[6][16]"
    assert env.rail.get_full_transitions(6, 17) == 1097, "[6][17]"
    assert env.rail.get_full_transitions(6, 18) == 1025, "[6][18]"
    assert env.rail.get_full_transitions(6, 19) == 4608, "[6][19]"
    assert env.rail.get_full_transitions(6, 20) == 0, "[6][20]"
    assert env.rail.get_full_transitions(6, 21) == 0, "[6][21]"
    assert env.rail.get_full_transitions(6, 22) == 0, "[6][22]"
    assert env.rail.get_full_transitions(6, 23) == 0, "[6][23]"
    assert env.rail.get_full_transitions(6, 24) == 0, "[6][24]"
    assert env.rail.get_full_transitions(7, 0) == 32800, "[7][0]"
    assert env.rail.get_full_transitions(7, 1) == 0, "[7][1]"
    assert env.rail.get_full_transitions(7, 2) == 0, "[7][2]"
    assert env.rail.get_full_transitions(7, 3) == 72, "[7][3]"
    assert env.rail.get_full_transitions(7, 4) == 1025, "[7][4]"
    assert env.rail.get_full_transitions(7, 5) == 1025, "[7][5]"
    assert env.rail.get_full_transitions(7, 6) == 1025, "[7][6]"
    assert env.rail.get_full_transitions(7, 7) == 1025, "[7][7]"
    assert env.rail.get_full_transitions(7, 8) == 1025, "[7][8]"
    assert env.rail.get_full_transitions(7, 9) == 2064, "[7][9]"
    assert env.rail.get_full_transitions(7, 10) == 4, "[7][10]"
    assert env.rail.get_full_transitions(7, 11) == 5633, "[7][11]"
    assert env.rail.get_full_transitions(7, 12) == 1025, "[7][12]"
    assert env.rail.get_full_transitions(7, 13) == 1025, "[7][13]"
    assert env.rail.get_full_transitions(7, 14) == 1025, "[7][14]"
    assert env.rail.get_full_transitions(7, 15) == 17411, "[7][15]"
    assert env.rail.get_full_transitions(7, 16) == 2064, "[7][16]"
    assert env.rail.get_full_transitions(7, 17) == 0, "[7][17]"
    assert env.rail.get_full_transitions(7, 18) == 0, "[7][18]"
    assert env.rail.get_full_transitions(7, 19) == 32800, "[7][19]"
    assert env.rail.get_full_transitions(7, 20) == 0, "[7][20]"
    assert env.rail.get_full_transitions(7, 21) == 0, "[7][21]"
    assert env.rail.get_full_transitions(7, 22) == 0, "[7][22]"
    assert env.rail.get_full_transitions(7, 23) == 0, "[7][23]"
    assert env.rail.get_full_transitions(7, 24) == 0, "[7][24]"
    assert env.rail.get_full_transitions(8, 0) == 32800, "[8][0]"
    assert env.rail.get_full_transitions(8, 1) == 0, "[8][1]"
    assert env.rail.get_full_transitions(8, 2) == 0, "[8][2]"
    assert env.rail.get_full_transitions(8, 3) == 0, "[8][3]"
    assert env.rail.get_full_transitions(8, 4) == 0, "[8][4]"
    assert env.rail.get_full_transitions(8, 5) == 0, "[8][5]"
    assert env.rail.get_full_transitions(8, 6) == 0, "[8][6]"
    assert env.rail.get_full_transitions(8, 7) == 0, "[8][7]"
    assert env.rail.get_full_transitions(8, 8) == 0, "[8][8]"
    assert env.rail.get_full_transitions(8, 9) == 0, "[8][9]"
    assert env.rail.get_full_transitions(8, 10) == 0, "[8][10]"
    assert env.rail.get_full_transitions(8, 11) == 72, "[8][11]"
    assert env.rail.get_full_transitions(8, 12) == 1025, "[8][12]"
    assert env.rail.get_full_transitions(8, 13) == 1025, "[8][13]"
    assert env.rail.get_full_transitions(8, 14) == 1025, "[8][14]"
    assert env.rail.get_full_transitions(8, 15) == 2064, "[8][15]"
    assert env.rail.get_full_transitions(8, 16) == 0, "[8][16]"
    assert env.rail.get_full_transitions(8, 17) == 0, "[8][17]"
    assert env.rail.get_full_transitions(8, 18) == 0, "[8][18]"
    assert env.rail.get_full_transitions(8, 19) == 32800, "[8][19]"
    assert env.rail.get_full_transitions(8, 20) == 0, "[8][20]"
    assert env.rail.get_full_transitions(8, 21) == 0, "[8][21]"
    assert env.rail.get_full_transitions(8, 22) == 0, "[8][22]"
    assert env.rail.get_full_transitions(8, 23) == 0, "[8][23]"
    assert env.rail.get_full_transitions(8, 24) == 0, "[8][24]"
    assert env.rail.get_full_transitions(9, 0) == 32800, "[9][0]"
    assert env.rail.get_full_transitions(9, 1) == 0, "[9][1]"
    assert env.rail.get_full_transitions(9, 2) == 0, "[9][2]"
    assert env.rail.get_full_transitions(9, 3) == 0, "[9][3]"
    assert env.rail.get_full_transitions(9, 4) == 0, "[9][4]"
    assert env.rail.get_full_transitions(9, 5) == 0, "[9][5]"
    assert env.rail.get_full_transitions(9, 6) == 0, "[9][6]"
    assert env.rail.get_full_transitions(9, 7) == 0, "[9][7]"
    assert env.rail.get_full_transitions(9, 8) == 0, "[9][8]"
    assert env.rail.get_full_transitions(9, 9) == 0, "[9][9]"
    assert env.rail.get_full_transitions(9, 10) == 0, "[9][10]"
    assert env.rail.get_full_transitions(9, 11) == 0, "[9][11]"
    assert env.rail.get_full_transitions(9, 12) == 0, "[9][12]"
    assert env.rail.get_full_transitions(9, 13) == 0, "[9][13]"
    assert env.rail.get_full_transitions(9, 14) == 0, "[9][14]"
    assert env.rail.get_full_transitions(9, 15) == 0, "[9][15]"
    assert env.rail.get_full_transitions(9, 16) == 0, "[9][16]"
    assert env.rail.get_full_transitions(9, 17) == 0, "[9][17]"
    assert env.rail.get_full_transitions(9, 18) == 0, "[9][18]"
    assert env.rail.get_full_transitions(9, 19) == 32800, "[9][19]"
    assert env.rail.get_full_transitions(9, 20) == 0, "[9][20]"
    assert env.rail.get_full_transitions(9, 21) == 0, "[9][21]"
    assert env.rail.get_full_transitions(9, 22) == 0, "[9][22]"
    assert env.rail.get_full_transitions(9, 23) == 0, "[9][23]"
    assert env.rail.get_full_transitions(9, 24) == 0, "[9][24]"
    assert env.rail.get_full_transitions(10, 0) == 32800, "[10][0]"
    assert env.rail.get_full_transitions(10, 1) == 0, "[10][1]"
    assert env.rail.get_full_transitions(10, 2) == 0, "[10][2]"
    assert env.rail.get_full_transitions(10, 3) == 0, "[10][3]"
    assert env.rail.get_full_transitions(10, 4) == 0, "[10][4]"
    assert env.rail.get_full_transitions(10, 5) == 0, "[10][5]"
    assert env.rail.get_full_transitions(10, 6) == 0, "[10][6]"
    assert env.rail.get_full_transitions(10, 7) == 0, "[10][7]"
    assert env.rail.get_full_transitions(10, 8) == 0, "[10][8]"
    assert env.rail.get_full_transitions(10, 9) == 0, "[10][9]"
    assert env.rail.get_full_transitions(10, 10) == 0, "[10][10]"
    assert env.rail.get_full_transitions(10, 11) == 0, "[10][11]"
    assert env.rail.get_full_transitions(10, 12) == 0, "[10][12]"
    assert env.rail.get_full_transitions(10, 13) == 0, "[10][13]"
    assert env.rail.get_full_transitions(10, 14) == 0, "[10][14]"
    assert env.rail.get_full_transitions(10, 15) == 0, "[10][15]"
    assert env.rail.get_full_transitions(10, 16) == 0, "[10][16]"
    assert env.rail.get_full_transitions(10, 17) == 0, "[10][17]"
    assert env.rail.get_full_transitions(10, 18) == 0, "[10][18]"
    assert env.rail.get_full_transitions(10, 19) == 32800, "[10][19]"
    assert env.rail.get_full_transitions(10, 20) == 0, "[10][20]"
    assert env.rail.get_full_transitions(10, 21) == 0, "[10][21]"
    assert env.rail.get_full_transitions(10, 22) == 0, "[10][22]"
    assert env.rail.get_full_transitions(10, 23) == 0, "[10][23]"
    assert env.rail.get_full_transitions(10, 24) == 0, "[10][24]"
    assert env.rail.get_full_transitions(11, 0) == 32800, "[11][0]"
    assert env.rail.get_full_transitions(11, 1) == 0, "[11][1]"
    assert env.rail.get_full_transitions(11, 2) == 0, "[11][2]"
    assert env.rail.get_full_transitions(11, 3) == 0, "[11][3]"
    assert env.rail.get_full_transitions(11, 4) == 0, "[11][4]"
    assert env.rail.get_full_transitions(11, 5) == 0, "[11][5]"
    assert env.rail.get_full_transitions(11, 6) == 0, "[11][6]"
    assert env.rail.get_full_transitions(11, 7) == 0, "[11][7]"
    assert env.rail.get_full_transitions(11, 8) == 0, "[11][8]"
    assert env.rail.get_full_transitions(11, 9) == 0, "[11][9]"
    assert env.rail.get_full_transitions(11, 10) == 0, "[11][10]"
    assert env.rail.get_full_transitions(11, 11) == 0, "[11][11]"
    assert env.rail.get_full_transitions(11, 12) == 0, "[11][12]"
    assert env.rail.get_full_transitions(11, 13) == 0, "[11][13]"
    assert env.rail.get_full_transitions(11, 14) == 0, "[11][14]"
    assert env.rail.get_full_transitions(11, 15) == 0, "[11][15]"
    assert env.rail.get_full_transitions(11, 16) == 0, "[11][16]"
    assert env.rail.get_full_transitions(11, 17) == 0, "[11][17]"
    assert env.rail.get_full_transitions(11, 18) == 0, "[11][18]"
    assert env.rail.get_full_transitions(11, 19) == 32800, "[11][19]"
    assert env.rail.get_full_transitions(11, 20) == 0, "[11][20]"
    assert env.rail.get_full_transitions(11, 21) == 0, "[11][21]"
    assert env.rail.get_full_transitions(11, 22) == 0, "[11][22]"
    assert env.rail.get_full_transitions(11, 23) == 0, "[11][23]"
    assert env.rail.get_full_transitions(11, 24) == 0, "[11][24]"
    assert env.rail.get_full_transitions(12, 0) == 32800, "[12][0]"
    assert env.rail.get_full_transitions(12, 1) == 0, "[12][1]"
    assert env.rail.get_full_transitions(12, 2) == 0, "[12][2]"
    assert env.rail.get_full_transitions(12, 3) == 0, "[12][3]"
    assert env.rail.get_full_transitions(12, 4) == 0, "[12][4]"
    assert env.rail.get_full_transitions(12, 5) == 0, "[12][5]"
    assert env.rail.get_full_transitions(12, 6) == 0, "[12][6]"
    assert env.rail.get_full_transitions(12, 7) == 0, "[12][7]"
    assert env.rail.get_full_transitions(12, 8) == 0, "[12][8]"
    assert env.rail.get_full_transitions(12, 9) == 0, "[12][9]"
    assert env.rail.get_full_transitions(12, 10) == 0, "[12][10]"
    assert env.rail.get_full_transitions(12, 11) == 0, "[12][11]"
    assert env.rail.get_full_transitions(12, 12) == 0, "[12][12]"
    assert env.rail.get_full_transitions(12, 13) == 0, "[12][13]"
    assert env.rail.get_full_transitions(12, 14) == 0, "[12][14]"
    assert env.rail.get_full_transitions(12, 15) == 0, "[12][15]"
    assert env.rail.get_full_transitions(12, 16) == 0, "[12][16]"
    assert env.rail.get_full_transitions(12, 17) == 0, "[12][17]"
    assert env.rail.get_full_transitions(12, 18) == 0, "[12][18]"
    assert env.rail.get_full_transitions(12, 19) == 32800, "[12][19]"
    assert env.rail.get_full_transitions(12, 20) == 0, "[12][20]"
    assert env.rail.get_full_transitions(12, 21) == 0, "[12][21]"
    assert env.rail.get_full_transitions(12, 22) == 0, "[12][22]"
    assert env.rail.get_full_transitions(12, 23) == 0, "[12][23]"
    assert env.rail.get_full_transitions(12, 24) == 0, "[12][24]"
    assert env.rail.get_full_transitions(13, 0) == 32800, "[13][0]"
    assert env.rail.get_full_transitions(13, 1) == 0, "[13][1]"
    assert env.rail.get_full_transitions(13, 2) == 0, "[13][2]"
    assert env.rail.get_full_transitions(13, 3) == 0, "[13][3]"
    assert env.rail.get_full_transitions(13, 4) == 0, "[13][4]"
    assert env.rail.get_full_transitions(13, 5) == 0, "[13][5]"
    assert env.rail.get_full_transitions(13, 6) == 0, "[13][6]"
    assert env.rail.get_full_transitions(13, 7) == 0, "[13][7]"
    assert env.rail.get_full_transitions(13, 8) == 0, "[13][8]"
    assert env.rail.get_full_transitions(13, 9) == 0, "[13][9]"
    assert env.rail.get_full_transitions(13, 10) == 0, "[13][10]"
    assert env.rail.get_full_transitions(13, 11) == 0, "[13][11]"
    assert env.rail.get_full_transitions(13, 12) == 0, "[13][12]"
    assert env.rail.get_full_transitions(13, 13) == 0, "[13][13]"
    assert env.rail.get_full_transitions(13, 14) == 0, "[13][14]"
    assert env.rail.get_full_transitions(13, 15) == 0, "[13][15]"
    assert env.rail.get_full_transitions(13, 16) == 0, "[13][16]"
    assert env.rail.get_full_transitions(13, 17) == 0, "[13][17]"
    assert env.rail.get_full_transitions(13, 18) == 0, "[13][18]"
    assert env.rail.get_full_transitions(13, 19) == 32800, "[13][19]"
    assert env.rail.get_full_transitions(13, 20) == 0, "[13][20]"
    assert env.rail.get_full_transitions(13, 21) == 0, "[13][21]"
    assert env.rail.get_full_transitions(13, 22) == 0, "[13][22]"
    assert env.rail.get_full_transitions(13, 23) == 0, "[13][23]"
    assert env.rail.get_full_transitions(13, 24) == 0, "[13][24]"
    assert env.rail.get_full_transitions(14, 0) == 32800, "[14][0]"
    assert env.rail.get_full_transitions(14, 1) == 0, "[14][1]"
    assert env.rail.get_full_transitions(14, 2) == 0, "[14][2]"
    assert env.rail.get_full_transitions(14, 3) == 0, "[14][3]"
    assert env.rail.get_full_transitions(14, 4) == 0, "[14][4]"
    assert env.rail.get_full_transitions(14, 5) == 0, "[14][5]"
    assert env.rail.get_full_transitions(14, 6) == 0, "[14][6]"
    assert env.rail.get_full_transitions(14, 7) == 0, "[14][7]"
    assert env.rail.get_full_transitions(14, 8) == 0, "[14][8]"
    assert env.rail.get_full_transitions(14, 9) == 0, "[14][9]"
    assert env.rail.get_full_transitions(14, 10) == 0, "[14][10]"
    assert env.rail.get_full_transitions(14, 11) == 0, "[14][11]"
    assert env.rail.get_full_transitions(14, 12) == 0, "[14][12]"
    assert env.rail.get_full_transitions(14, 13) == 0, "[14][13]"
    assert env.rail.get_full_transitions(14, 14) == 0, "[14][14]"
    assert env.rail.get_full_transitions(14, 15) == 0, "[14][15]"
    assert env.rail.get_full_transitions(14, 16) == 0, "[14][16]"
    assert env.rail.get_full_transitions(14, 17) == 0, "[14][17]"
    assert env.rail.get_full_transitions(14, 18) == 0, "[14][18]"
    assert env.rail.get_full_transitions(14, 19) == 32800, "[14][19]"
    assert env.rail.get_full_transitions(14, 20) == 0, "[14][20]"
    assert env.rail.get_full_transitions(14, 21) == 0, "[14][21]"
    assert env.rail.get_full_transitions(14, 22) == 0, "[14][22]"
    assert env.rail.get_full_transitions(14, 23) == 0, "[14][23]"
    assert env.rail.get_full_transitions(14, 24) == 0, "[14][24]"
    assert env.rail.get_full_transitions(15, 0) == 32800, "[15][0]"
    assert env.rail.get_full_transitions(15, 1) == 0, "[15][1]"
    assert env.rail.get_full_transitions(15, 2) == 0, "[15][2]"
    assert env.rail.get_full_transitions(15, 3) == 0, "[15][3]"
    assert env.rail.get_full_transitions(15, 4) == 0, "[15][4]"
    assert env.rail.get_full_transitions(15, 5) == 0, "[15][5]"
    assert env.rail.get_full_transitions(15, 6) == 0, "[15][6]"
    assert env.rail.get_full_transitions(15, 7) == 0, "[15][7]"
    assert env.rail.get_full_transitions(15, 8) == 0, "[15][8]"
    assert env.rail.get_full_transitions(15, 9) == 0, "[15][9]"
    assert env.rail.get_full_transitions(15, 10) == 0, "[15][10]"
    assert env.rail.get_full_transitions(15, 11) == 0, "[15][11]"
    assert env.rail.get_full_transitions(15, 12) == 0, "[15][12]"
    assert env.rail.get_full_transitions(15, 13) == 0, "[15][13]"
    assert env.rail.get_full_transitions(15, 14) == 0, "[15][14]"
    assert env.rail.get_full_transitions(15, 15) == 0, "[15][15]"
    assert env.rail.get_full_transitions(15, 16) == 0, "[15][16]"
    assert env.rail.get_full_transitions(15, 17) == 0, "[15][17]"
    assert env.rail.get_full_transitions(15, 18) == 0, "[15][18]"
    assert env.rail.get_full_transitions(15, 19) == 32800, "[15][19]"
    assert env.rail.get_full_transitions(15, 20) == 0, "[15][20]"
    assert env.rail.get_full_transitions(15, 21) == 0, "[15][21]"
    assert env.rail.get_full_transitions(15, 22) == 0, "[15][22]"
    assert env.rail.get_full_transitions(15, 23) == 0, "[15][23]"
    assert env.rail.get_full_transitions(15, 24) == 0, "[15][24]"
    assert env.rail.get_full_transitions(16, 0) == 32800, "[16][0]"
    assert env.rail.get_full_transitions(16, 1) == 0, "[16][1]"
    assert env.rail.get_full_transitions(16, 2) == 0, "[16][2]"
    assert env.rail.get_full_transitions(16, 3) == 0, "[16][3]"
    assert env.rail.get_full_transitions(16, 4) == 0, "[16][4]"
    assert env.rail.get_full_transitions(16, 5) == 0, "[16][5]"
    assert env.rail.get_full_transitions(16, 6) == 0, "[16][6]"
    assert env.rail.get_full_transitions(16, 7) == 0, "[16][7]"
    assert env.rail.get_full_transitions(16, 8) == 0, "[16][8]"
    assert env.rail.get_full_transitions(16, 9) == 0, "[16][9]"
    assert env.rail.get_full_transitions(16, 10) == 0, "[16][10]"
    assert env.rail.get_full_transitions(16, 11) == 0, "[16][11]"
    assert env.rail.get_full_transitions(16, 12) == 0, "[16][12]"
    assert env.rail.get_full_transitions(16, 13) == 0, "[16][13]"
    assert env.rail.get_full_transitions(16, 14) == 0, "[16][14]"
    assert env.rail.get_full_transitions(16, 15) == 0, "[16][15]"
    assert env.rail.get_full_transitions(16, 16) == 0, "[16][16]"
    assert env.rail.get_full_transitions(16, 17) == 0, "[16][17]"
    assert env.rail.get_full_transitions(16, 18) == 0, "[16][18]"
    assert env.rail.get_full_transitions(16, 19) == 32800, "[16][19]"
    assert env.rail.get_full_transitions(16, 20) == 0, "[16][20]"
    assert env.rail.get_full_transitions(16, 21) == 0, "[16][21]"
    assert env.rail.get_full_transitions(16, 22) == 0, "[16][22]"
    assert env.rail.get_full_transitions(16, 23) == 0, "[16][23]"
    assert env.rail.get_full_transitions(16, 24) == 0, "[16][24]"
    assert env.rail.get_full_transitions(17, 0) == 32800, "[17][0]"
    assert env.rail.get_full_transitions(17, 1) == 0, "[17][1]"
    assert env.rail.get_full_transitions(17, 2) == 0, "[17][2]"
    assert env.rail.get_full_transitions(17, 3) == 0, "[17][3]"
    assert env.rail.get_full_transitions(17, 4) == 0, "[17][4]"
    assert env.rail.get_full_transitions(17, 5) == 0, "[17][5]"
    assert env.rail.get_full_transitions(17, 6) == 0, "[17][6]"
    assert env.rail.get_full_transitions(17, 7) == 0, "[17][7]"
    assert env.rail.get_full_transitions(17, 8) == 0, "[17][8]"
    assert env.rail.get_full_transitions(17, 9) == 0, "[17][9]"
    assert env.rail.get_full_transitions(17, 10) == 0, "[17][10]"
    assert env.rail.get_full_transitions(17, 11) == 0, "[17][11]"
    assert env.rail.get_full_transitions(17, 12) == 0, "[17][12]"
    assert env.rail.get_full_transitions(17, 13) == 0, "[17][13]"
    assert env.rail.get_full_transitions(17, 14) == 0, "[17][14]"
    assert env.rail.get_full_transitions(17, 15) == 0, "[17][15]"
    assert env.rail.get_full_transitions(17, 16) == 0, "[17][16]"
    assert env.rail.get_full_transitions(17, 17) == 0, "[17][17]"
    assert env.rail.get_full_transitions(17, 18) == 0, "[17][18]"
    assert env.rail.get_full_transitions(17, 19) == 32800, "[17][19]"
    assert env.rail.get_full_transitions(17, 20) == 0, "[17][20]"
    assert env.rail.get_full_transitions(17, 21) == 0, "[17][21]"
    assert env.rail.get_full_transitions(17, 22) == 0, "[17][22]"
    assert env.rail.get_full_transitions(17, 23) == 0, "[17][23]"
    assert env.rail.get_full_transitions(17, 24) == 0, "[17][24]"
    assert env.rail.get_full_transitions(18, 0) == 72, "[18][0]"
    assert env.rail.get_full_transitions(18, 1) == 1025, "[18][1]"
    assert env.rail.get_full_transitions(18, 2) == 1025, "[18][2]"
    assert env.rail.get_full_transitions(18, 3) == 5633, "[18][3]"
    assert env.rail.get_full_transitions(18, 4) == 1025, "[18][4]"
    assert env.rail.get_full_transitions(18, 5) == 1025, "[18][5]"
    assert env.rail.get_full_transitions(18, 6) == 1025, "[18][6]"
    assert env.rail.get_full_transitions(18, 7) == 1025, "[18][7]"
    assert env.rail.get_full_transitions(18, 8) == 1025, "[18][8]"
    assert env.rail.get_full_transitions(18, 9) == 5633, "[18][9]"
    assert env.rail.get_full_transitions(18, 10) == 1025, "[18][10]"
    assert env.rail.get_full_transitions(18, 11) == 1025, "[18][11]"
    assert env.rail.get_full_transitions(18, 12) == 1025, "[18][12]"
    assert env.rail.get_full_transitions(18, 13) == 1025, "[18][13]"
    assert env.rail.get_full_transitions(18, 14) == 1025, "[18][14]"
    assert env.rail.get_full_transitions(18, 15) == 1025, "[18][15]"
    assert env.rail.get_full_transitions(18, 16) == 17411, "[18][16]"
    assert env.rail.get_full_transitions(18, 17) == 1025, "[18][17]"
    assert env.rail.get_full_transitions(18, 18) == 1025, "[18][18]"
    assert env.rail.get_full_transitions(18, 19) == 34864, "[18][19]"
    assert env.rail.get_full_transitions(18, 20) == 0, "[18][20]"
    assert env.rail.get_full_transitions(18, 21) == 0, "[18][21]"
    assert env.rail.get_full_transitions(18, 22) == 0, "[18][22]"
    assert env.rail.get_full_transitions(18, 23) == 0, "[18][23]"
    assert env.rail.get_full_transitions(18, 24) == 0, "[18][24]"
    assert env.rail.get_full_transitions(19, 0) == 0, "[19][0]"
    assert env.rail.get_full_transitions(19, 1) == 0, "[19][1]"
    assert env.rail.get_full_transitions(19, 2) == 0, "[19][2]"
    assert env.rail.get_full_transitions(19, 3) == 72, "[19][3]"
    assert env.rail.get_full_transitions(19, 4) == 1025, "[19][4]"
    assert env.rail.get_full_transitions(19, 5) == 1025, "[19][5]"
    assert env.rail.get_full_transitions(19, 6) == 1025, "[19][6]"
    assert env.rail.get_full_transitions(19, 7) == 1025, "[19][7]"
    assert env.rail.get_full_transitions(19, 8) == 1025, "[19][8]"
    assert env.rail.get_full_transitions(19, 9) == 3089, "[19][9]"
    assert env.rail.get_full_transitions(19, 10) == 1025, "[19][10]"
    assert env.rail.get_full_transitions(19, 11) == 1025, "[19][11]"
    assert env.rail.get_full_transitions(19, 12) == 1025, "[19][12]"
    assert env.rail.get_full_transitions(19, 13) == 1025, "[19][13]"
    assert env.rail.get_full_transitions(19, 14) == 1025, "[19][14]"
    assert env.rail.get_full_transitions(19, 15) == 1025, "[19][15]"
    assert env.rail.get_full_transitions(19, 16) == 3089, "[19][16]"
    assert env.rail.get_full_transitions(19, 17) == 1025, "[19][17]"
    assert env.rail.get_full_transitions(19, 18) == 1025, "[19][18]"
    assert env.rail.get_full_transitions(19, 19) == 2064, "[19][19]"
    assert env.rail.get_full_transitions(19, 20) == 0, "[19][20]"
    assert env.rail.get_full_transitions(19, 21) == 0, "[19][21]"
    assert env.rail.get_full_transitions(19, 22) == 0, "[19][22]"
    assert env.rail.get_full_transitions(19, 23) == 0, "[19][23]"
    assert env.rail.get_full_transitions(19, 24) == 0, "[19][24]"
    assert env.rail.get_full_transitions(20, 0) == 0, "[20][0]"
    assert env.rail.get_full_transitions(20, 1) == 0, "[20][1]"
    assert env.rail.get_full_transitions(20, 2) == 0, "[20][2]"
    assert env.rail.get_full_transitions(20, 3) == 0, "[20][3]"
    assert env.rail.get_full_transitions(20, 4) == 0, "[20][4]"
    assert env.rail.get_full_transitions(20, 5) == 0, "[20][5]"
    assert env.rail.get_full_transitions(20, 6) == 0, "[20][6]"
    assert env.rail.get_full_transitions(20, 7) == 0, "[20][7]"
    assert env.rail.get_full_transitions(20, 8) == 0, "[20][8]"
    assert env.rail.get_full_transitions(20, 9) == 0, "[20][9]"
    assert env.rail.get_full_transitions(20, 10) == 0, "[20][10]"
    assert env.rail.get_full_transitions(20, 11) == 0, "[20][11]"
    assert env.rail.get_full_transitions(20, 12) == 0, "[20][12]"
    assert env.rail.get_full_transitions(20, 13) == 0, "[20][13]"
    assert env.rail.get_full_transitions(20, 14) == 0, "[20][14]"
    assert env.rail.get_full_transitions(20, 15) == 0, "[20][15]"
    assert env.rail.get_full_transitions(20, 16) == 0, "[20][16]"
    assert env.rail.get_full_transitions(20, 17) == 0, "[20][17]"
    assert env.rail.get_full_transitions(20, 18) == 0, "[20][18]"
    assert env.rail.get_full_transitions(20, 19) == 0, "[20][19]"
    assert env.rail.get_full_transitions(20, 20) == 0, "[20][20]"
    assert env.rail.get_full_transitions(20, 21) == 0, "[20][21]"
    assert env.rail.get_full_transitions(20, 22) == 0, "[20][22]"
    assert env.rail.get_full_transitions(20, 23) == 0, "[20][23]"
    assert env.rail.get_full_transitions(20, 24) == 0, "[20][24]"
    assert env.rail.get_full_transitions(21, 0) == 0, "[21][0]"
    assert env.rail.get_full_transitions(21, 1) == 0, "[21][1]"
    assert env.rail.get_full_transitions(21, 2) == 0, "[21][2]"
    assert env.rail.get_full_transitions(21, 3) == 0, "[21][3]"
    assert env.rail.get_full_transitions(21, 4) == 0, "[21][4]"
    assert env.rail.get_full_transitions(21, 5) == 0, "[21][5]"
    assert env.rail.get_full_transitions(21, 6) == 0, "[21][6]"
    assert env.rail.get_full_transitions(21, 7) == 0, "[21][7]"
    assert env.rail.get_full_transitions(21, 8) == 0, "[21][8]"
    assert env.rail.get_full_transitions(21, 9) == 0, "[21][9]"
    assert env.rail.get_full_transitions(21, 10) == 0, "[21][10]"
    assert env.rail.get_full_transitions(21, 11) == 0, "[21][11]"
    assert env.rail.get_full_transitions(21, 12) == 0, "[21][12]"
    assert env.rail.get_full_transitions(21, 13) == 0, "[21][13]"
    assert env.rail.get_full_transitions(21, 14) == 0, "[21][14]"
    assert env.rail.get_full_transitions(21, 15) == 0, "[21][15]"
    assert env.rail.get_full_transitions(21, 16) == 0, "[21][16]"
    assert env.rail.get_full_transitions(21, 17) == 0, "[21][17]"
    assert env.rail.get_full_transitions(21, 18) == 0, "[21][18]"
    assert env.rail.get_full_transitions(21, 19) == 0, "[21][19]"
    assert env.rail.get_full_transitions(21, 20) == 0, "[21][20]"
    assert env.rail.get_full_transitions(21, 21) == 0, "[21][21]"
    assert env.rail.get_full_transitions(21, 22) == 0, "[21][22]"
    assert env.rail.get_full_transitions(21, 23) == 0, "[21][23]"
    assert env.rail.get_full_transitions(21, 24) == 0, "[21][24]"
    assert env.rail.get_full_transitions(22, 0) == 0, "[22][0]"
    assert env.rail.get_full_transitions(22, 1) == 0, "[22][1]"
    assert env.rail.get_full_transitions(22, 2) == 0, "[22][2]"
    assert env.rail.get_full_transitions(22, 3) == 0, "[22][3]"
    assert env.rail.get_full_transitions(22, 4) == 0, "[22][4]"
    assert env.rail.get_full_transitions(22, 5) == 0, "[22][5]"
    assert env.rail.get_full_transitions(22, 6) == 0, "[22][6]"
    assert env.rail.get_full_transitions(22, 7) == 0, "[22][7]"
    assert env.rail.get_full_transitions(22, 8) == 0, "[22][8]"
    assert env.rail.get_full_transitions(22, 9) == 0, "[22][9]"
    assert env.rail.get_full_transitions(22, 10) == 0, "[22][10]"
    assert env.rail.get_full_transitions(22, 11) == 0, "[22][11]"
    assert env.rail.get_full_transitions(22, 12) == 0, "[22][12]"
    assert env.rail.get_full_transitions(22, 13) == 0, "[22][13]"
    assert env.rail.get_full_transitions(22, 14) == 0, "[22][14]"
    assert env.rail.get_full_transitions(22, 15) == 0, "[22][15]"
    assert env.rail.get_full_transitions(22, 16) == 0, "[22][16]"
    assert env.rail.get_full_transitions(22, 17) == 0, "[22][17]"
    assert env.rail.get_full_transitions(22, 18) == 0, "[22][18]"
    assert env.rail.get_full_transitions(22, 19) == 0, "[22][19]"
    assert env.rail.get_full_transitions(22, 20) == 0, "[22][20]"
    assert env.rail.get_full_transitions(22, 21) == 0, "[22][21]"
    assert env.rail.get_full_transitions(22, 22) == 0, "[22][22]"
    assert env.rail.get_full_transitions(22, 23) == 0, "[22][23]"
    assert env.rail.get_full_transitions(22, 24) == 0, "[22][24]"
    assert env.rail.get_full_transitions(23, 0) == 0, "[23][0]"
    assert env.rail.get_full_transitions(23, 1) == 0, "[23][1]"
    assert env.rail.get_full_transitions(23, 2) == 0, "[23][2]"
    assert env.rail.get_full_transitions(23, 3) == 0, "[23][3]"
    assert env.rail.get_full_transitions(23, 4) == 0, "[23][4]"
    assert env.rail.get_full_transitions(23, 5) == 0, "[23][5]"
    assert env.rail.get_full_transitions(23, 6) == 0, "[23][6]"
    assert env.rail.get_full_transitions(23, 7) == 0, "[23][7]"
    assert env.rail.get_full_transitions(23, 8) == 0, "[23][8]"
    assert env.rail.get_full_transitions(23, 9) == 0, "[23][9]"
    assert env.rail.get_full_transitions(23, 10) == 0, "[23][10]"
    assert env.rail.get_full_transitions(23, 11) == 0, "[23][11]"
    assert env.rail.get_full_transitions(23, 12) == 0, "[23][12]"
    assert env.rail.get_full_transitions(23, 13) == 0, "[23][13]"
    assert env.rail.get_full_transitions(23, 14) == 0, "[23][14]"
    assert env.rail.get_full_transitions(23, 15) == 0, "[23][15]"
    assert env.rail.get_full_transitions(23, 16) == 0, "[23][16]"
    assert env.rail.get_full_transitions(23, 17) == 0, "[23][17]"
    assert env.rail.get_full_transitions(23, 18) == 0, "[23][18]"
    assert env.rail.get_full_transitions(23, 19) == 0, "[23][19]"
    assert env.rail.get_full_transitions(23, 20) == 0, "[23][20]"
    assert env.rail.get_full_transitions(23, 21) == 0, "[23][21]"
    assert env.rail.get_full_transitions(23, 22) == 0, "[23][22]"
    assert env.rail.get_full_transitions(23, 23) == 0, "[23][23]"
    assert env.rail.get_full_transitions(23, 24) == 0, "[23][24]"
    assert env.rail.get_full_transitions(24, 0) == 0, "[24][0]"
    assert env.rail.get_full_transitions(24, 1) == 0, "[24][1]"
    assert env.rail.get_full_transitions(24, 2) == 0, "[24][2]"
    assert env.rail.get_full_transitions(24, 3) == 0, "[24][3]"
    assert env.rail.get_full_transitions(24, 4) == 0, "[24][4]"
    assert env.rail.get_full_transitions(24, 5) == 0, "[24][5]"
    assert env.rail.get_full_transitions(24, 6) == 0, "[24][6]"
    assert env.rail.get_full_transitions(24, 7) == 0, "[24][7]"
    assert env.rail.get_full_transitions(24, 8) == 0, "[24][8]"
    assert env.rail.get_full_transitions(24, 9) == 0, "[24][9]"
    assert env.rail.get_full_transitions(24, 10) == 0, "[24][10]"
    assert env.rail.get_full_transitions(24, 11) == 0, "[24][11]"
    assert env.rail.get_full_transitions(24, 12) == 0, "[24][12]"
    assert env.rail.get_full_transitions(24, 13) == 0, "[24][13]"
    assert env.rail.get_full_transitions(24, 14) == 0, "[24][14]"
    assert env.rail.get_full_transitions(24, 15) == 0, "[24][15]"
    assert env.rail.get_full_transitions(24, 16) == 0, "[24][16]"
    assert env.rail.get_full_transitions(24, 17) == 0, "[24][17]"
    assert env.rail.get_full_transitions(24, 18) == 0, "[24][18]"
    assert env.rail.get_full_transitions(24, 19) == 0, "[24][19]"
    assert env.rail.get_full_transitions(24, 20) == 0, "[24][20]"
    assert env.rail.get_full_transitions(24, 21) == 0, "[24][21]"
    assert env.rail.get_full_transitions(24, 22) == 0, "[24][22]"
    assert env.rail.get_full_transitions(24, 23) == 0, "[24][23]"
    assert env.rail.get_full_transitions(24, 24) == 0, "[24][24]"
    assert env.rail.get_full_transitions(25, 0) == 0, "[25][0]"
    assert env.rail.get_full_transitions(25, 1) == 0, "[25][1]"
    assert env.rail.get_full_transitions(25, 2) == 0, "[25][2]"
    assert env.rail.get_full_transitions(25, 3) == 0, "[25][3]"
    assert env.rail.get_full_transitions(25, 4) == 0, "[25][4]"
    assert env.rail.get_full_transitions(25, 5) == 0, "[25][5]"
    assert env.rail.get_full_transitions(25, 6) == 0, "[25][6]"
    assert env.rail.get_full_transitions(25, 7) == 0, "[25][7]"
    assert env.rail.get_full_transitions(25, 8) == 0, "[25][8]"
    assert env.rail.get_full_transitions(25, 9) == 0, "[25][9]"
    assert env.rail.get_full_transitions(25, 10) == 0, "[25][10]"
    assert env.rail.get_full_transitions(25, 11) == 0, "[25][11]"
    assert env.rail.get_full_transitions(25, 12) == 0, "[25][12]"
    assert env.rail.get_full_transitions(25, 13) == 0, "[25][13]"
    assert env.rail.get_full_transitions(25, 14) == 0, "[25][14]"
    assert env.rail.get_full_transitions(25, 15) == 0, "[25][15]"
    assert env.rail.get_full_transitions(25, 16) == 0, "[25][16]"
    assert env.rail.get_full_transitions(25, 17) == 0, "[25][17]"
    assert env.rail.get_full_transitions(25, 18) == 0, "[25][18]"
    assert env.rail.get_full_transitions(25, 19) == 0, "[25][19]"
    assert env.rail.get_full_transitions(25, 20) == 0, "[25][20]"
    assert env.rail.get_full_transitions(25, 21) == 0, "[25][21]"
    assert env.rail.get_full_transitions(25, 22) == 0, "[25][22]"
    assert env.rail.get_full_transitions(25, 23) == 0, "[25][23]"
    assert env.rail.get_full_transitions(25, 24) == 0, "[25][24]"
    assert env.rail.get_full_transitions(26, 0) == 0, "[26][0]"
    assert env.rail.get_full_transitions(26, 1) == 0, "[26][1]"
    assert env.rail.get_full_transitions(26, 2) == 0, "[26][2]"
    assert env.rail.get_full_transitions(26, 3) == 0, "[26][3]"
    assert env.rail.get_full_transitions(26, 4) == 0, "[26][4]"
    assert env.rail.get_full_transitions(26, 5) == 0, "[26][5]"
    assert env.rail.get_full_transitions(26, 6) == 0, "[26][6]"
    assert env.rail.get_full_transitions(26, 7) == 0, "[26][7]"
    assert env.rail.get_full_transitions(26, 8) == 0, "[26][8]"
    assert env.rail.get_full_transitions(26, 9) == 0, "[26][9]"
    assert env.rail.get_full_transitions(26, 10) == 0, "[26][10]"
    assert env.rail.get_full_transitions(26, 11) == 0, "[26][11]"
    assert env.rail.get_full_transitions(26, 12) == 0, "[26][12]"
    assert env.rail.get_full_transitions(26, 13) == 0, "[26][13]"
    assert env.rail.get_full_transitions(26, 14) == 0, "[26][14]"
    assert env.rail.get_full_transitions(26, 15) == 0, "[26][15]"
    assert env.rail.get_full_transitions(26, 16) == 0, "[26][16]"
    assert env.rail.get_full_transitions(26, 17) == 0, "[26][17]"
    assert env.rail.get_full_transitions(26, 18) == 0, "[26][18]"
    assert env.rail.get_full_transitions(26, 19) == 0, "[26][19]"
    assert env.rail.get_full_transitions(26, 20) == 0, "[26][20]"
    assert env.rail.get_full_transitions(26, 21) == 0, "[26][21]"
    assert env.rail.get_full_transitions(26, 22) == 0, "[26][22]"
    assert env.rail.get_full_transitions(26, 23) == 0, "[26][23]"
    assert env.rail.get_full_transitions(26, 24) == 0, "[26][24]"
    assert env.rail.get_full_transitions(27, 0) == 0, "[27][0]"
    assert env.rail.get_full_transitions(27, 1) == 0, "[27][1]"
    assert env.rail.get_full_transitions(27, 2) == 0, "[27][2]"
    assert env.rail.get_full_transitions(27, 3) == 0, "[27][3]"
    assert env.rail.get_full_transitions(27, 4) == 0, "[27][4]"
    assert env.rail.get_full_transitions(27, 5) == 0, "[27][5]"
    assert env.rail.get_full_transitions(27, 6) == 0, "[27][6]"
    assert env.rail.get_full_transitions(27, 7) == 0, "[27][7]"
    assert env.rail.get_full_transitions(27, 8) == 0, "[27][8]"
    assert env.rail.get_full_transitions(27, 9) == 0, "[27][9]"
    assert env.rail.get_full_transitions(27, 10) == 0, "[27][10]"
    assert env.rail.get_full_transitions(27, 11) == 0, "[27][11]"
    assert env.rail.get_full_transitions(27, 12) == 0, "[27][12]"
    assert env.rail.get_full_transitions(27, 13) == 0, "[27][13]"
    assert env.rail.get_full_transitions(27, 14) == 0, "[27][14]"
    assert env.rail.get_full_transitions(27, 15) == 0, "[27][15]"
    assert env.rail.get_full_transitions(27, 16) == 0, "[27][16]"
    assert env.rail.get_full_transitions(27, 17) == 0, "[27][17]"
    assert env.rail.get_full_transitions(27, 18) == 0, "[27][18]"
    assert env.rail.get_full_transitions(27, 19) == 0, "[27][19]"
    assert env.rail.get_full_transitions(27, 20) == 0, "[27][20]"
    assert env.rail.get_full_transitions(27, 21) == 0, "[27][21]"
    assert env.rail.get_full_transitions(27, 22) == 0, "[27][22]"
    assert env.rail.get_full_transitions(27, 23) == 0, "[27][23]"
    assert env.rail.get_full_transitions(27, 24) == 0, "[27][24]"
    assert env.rail.get_full_transitions(28, 0) == 0, "[28][0]"
    assert env.rail.get_full_transitions(28, 1) == 0, "[28][1]"
    assert env.rail.get_full_transitions(28, 2) == 0, "[28][2]"
    assert env.rail.get_full_transitions(28, 3) == 0, "[28][3]"
    assert env.rail.get_full_transitions(28, 4) == 0, "[28][4]"
    assert env.rail.get_full_transitions(28, 5) == 0, "[28][5]"
    assert env.rail.get_full_transitions(28, 6) == 0, "[28][6]"
    assert env.rail.get_full_transitions(28, 7) == 0, "[28][7]"
    assert env.rail.get_full_transitions(28, 8) == 0, "[28][8]"
    assert env.rail.get_full_transitions(28, 9) == 0, "[28][9]"
    assert env.rail.get_full_transitions(28, 10) == 0, "[28][10]"
    assert env.rail.get_full_transitions(28, 11) == 0, "[28][11]"
    assert env.rail.get_full_transitions(28, 12) == 0, "[28][12]"
    assert env.rail.get_full_transitions(28, 13) == 0, "[28][13]"
    assert env.rail.get_full_transitions(28, 14) == 0, "[28][14]"
    assert env.rail.get_full_transitions(28, 15) == 0, "[28][15]"
    assert env.rail.get_full_transitions(28, 16) == 0, "[28][16]"
    assert env.rail.get_full_transitions(28, 17) == 0, "[28][17]"
    assert env.rail.get_full_transitions(28, 18) == 0, "[28][18]"
    assert env.rail.get_full_transitions(28, 19) == 0, "[28][19]"
    assert env.rail.get_full_transitions(28, 20) == 0, "[28][20]"
    assert env.rail.get_full_transitions(28, 21) == 0, "[28][21]"
    assert env.rail.get_full_transitions(28, 22) == 0, "[28][22]"
    assert env.rail.get_full_transitions(28, 23) == 0, "[28][23]"
    assert env.rail.get_full_transitions(28, 24) == 0, "[28][24]"
    assert env.rail.get_full_transitions(29, 0) == 0, "[29][0]"
    assert env.rail.get_full_transitions(29, 1) == 0, "[29][1]"
    assert env.rail.get_full_transitions(29, 2) == 0, "[29][2]"
    assert env.rail.get_full_transitions(29, 3) == 0, "[29][3]"
    assert env.rail.get_full_transitions(29, 4) == 0, "[29][4]"
    assert env.rail.get_full_transitions(29, 5) == 0, "[29][5]"
    assert env.rail.get_full_transitions(29, 6) == 0, "[29][6]"
    assert env.rail.get_full_transitions(29, 7) == 0, "[29][7]"
    assert env.rail.get_full_transitions(29, 8) == 0, "[29][8]"
    assert env.rail.get_full_transitions(29, 9) == 0, "[29][9]"
    assert env.rail.get_full_transitions(29, 10) == 0, "[29][10]"
    assert env.rail.get_full_transitions(29, 11) == 0, "[29][11]"
    assert env.rail.get_full_transitions(29, 12) == 0, "[29][12]"
    assert env.rail.get_full_transitions(29, 13) == 0, "[29][13]"
    assert env.rail.get_full_transitions(29, 14) == 0, "[29][14]"
    assert env.rail.get_full_transitions(29, 15) == 0, "[29][15]"
    assert env.rail.get_full_transitions(29, 16) == 0, "[29][16]"
    assert env.rail.get_full_transitions(29, 17) == 0, "[29][17]"
    assert env.rail.get_full_transitions(29, 18) == 0, "[29][18]"
    assert env.rail.get_full_transitions(29, 19) == 0, "[29][19]"
    assert env.rail.get_full_transitions(29, 20) == 0, "[29][20]"
    assert env.rail.get_full_transitions(29, 21) == 0, "[29][21]"
    assert env.rail.get_full_transitions(29, 22) == 0, "[29][22]"
    assert env.rail.get_full_transitions(29, 23) == 0, "[29][23]"
    assert env.rail.get_full_transitions(29, 24) == 0, "[29][24]"

def test_rail_env_action_required_info():
    np.random.seed(0)
    random.seed(0)
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train
    env_always_action = RailEnv(width=50,
                                height=50,
                                rail_generator=sparse_rail_generator(
                                    max_num_cities=10,
                                    max_rails_between_cities=3,
                                    seed=5,  # Random seed
                                    grid_mode=False  # Ordered distribution of nodes
                                ),
                                schedule_generator=sparse_schedule_generator(speed_ration_map),
                                number_of_agents=10,
                                obs_builder_object=GlobalObsForRailEnv())
    np.random.seed(0)
    random.seed(0)
    env_only_if_action_required = RailEnv(width=50,
                                          height=50,
                                          rail_generator=sparse_rail_generator(
                                              max_num_cities=10,
                                              max_rails_between_cities=3,
                                              seed=5,  # Random seed
                                              grid_mode=False
                                              # Ordered distribution of nodes
                                          ),
                                          schedule_generator=sparse_schedule_generator(speed_ration_map),
                                          number_of_agents=10,
                                          obs_builder_object=GlobalObsForRailEnv())
    env_renderer = RenderTool(env_always_action, gl="PILSVG", )

    env_always_action.reset(False, False, True)
    env_only_if_action_required.reset(False, False, True)

    for step in range(100):
        print("step {}".format(step))

        action_dict_always_action = dict()
        action_dict_only_if_action_required = dict()
        # Chose an action for each agent in the environment
        for a in range(env_always_action.get_num_agents()):
            action = np.random.choice(np.arange(4))
            action_dict_always_action.update({a: action})
            if step == 0 or info_only_if_action_required['action_required'][a]:
                action_dict_only_if_action_required.update({a: action})
            else:
                print("[{}] not action_required {}, speed_data={}".format(step, a,
                                                                          env_always_action.agents[a].speed_data))

        obs_always_action, rewards_always_action, done_always_action, info_always_action = env_always_action.step(
            action_dict_always_action)
        obs_only_if_action_required, rewards_only_if_action_required, done_only_if_action_required, info_only_if_action_required = env_only_if_action_required.step(
            action_dict_only_if_action_required)

        for a in range(env_always_action.get_num_agents()):
            assert len(obs_always_action[a]) == len(obs_only_if_action_required[a])
            for i in range(len(obs_always_action[a])):
                assert len(obs_always_action[a][i]) == len(obs_only_if_action_required[a][i])
                equal = np.array_equal(obs_always_action[a][i], obs_only_if_action_required[a][i])
                if not equal:
                    for r in range(50):
                        for c in range(50):
                            assert np.array_equal(obs_always_action[a][i][(r, c)], obs_only_if_action_required[a][i][
                                (r, c)]), "[{}]  a={},i={},{}\n{}\n\nvs.\n\n{}".format(step, a, i, (r, c),
                                                                                       obs_always_action[a][i][(r, c)],
                                                                                       obs_only_if_action_required[a][
                                                                                           i][(r, c)])
                assert equal, \
                    "[{}]   [{}][{}] {} vs. {}".format(step, a, i, obs_always_action[a][i],
                                                       obs_only_if_action_required[a][i])
            assert np.array_equal(rewards_always_action[a], rewards_only_if_action_required[a])
            assert np.array_equal(done_always_action[a], done_only_if_action_required[a])
            assert info_always_action['action_required'][a] == info_only_if_action_required['action_required'][a]

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if done_always_action['__all__']:
            break
    env_renderer.close_window()


def test_rail_env_malfunction_speed_info():
    np.random.seed(0)
    stochastic_data = {'prop_malfunction': 0.5,  # Percentage of defective agents
                       'malfunction_rate': 30,  # Rate of malfunction occurence
                       'min_duration': 3,  # Minimal duration of malfunction
                       'max_duration': 10  # Max duration of malfunction
                       }
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(max_num_cities=10,
                                                       max_rails_between_cities=3,
                                                       seed=5,
                                                       grid_mode=False
                                                       ),
                  schedule_generator=sparse_schedule_generator(),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv(),
                  stochastic_data=stochastic_data)
    env.reset(False, False, True)

    env_renderer = RenderTool(env, gl="PILSVG", )
    for step in range(100):
        action_dict = dict()
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = np.random.choice(np.arange(4))
            action_dict.update({a: action})

        obs, rewards, done, info = env.step(
            action_dict)

        assert 'malfunction' in info
        for a in range(env.get_num_agents()):
            assert info['malfunction'][a] >= 0
            assert info['speed'][a] >= 0 and info['speed'][a] <= 1
            assert info['speed'][a] == env.agents[a].speed_data['speed']

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if done['__all__']:
            break
    env_renderer.close_window()


def test_sparse_generator_with_too_man_cities_does_not_break_down():
    np.random.seed(0)

    RailEnv(width=50,
            height=50,
            rail_generator=sparse_rail_generator(
                max_num_cities=100,
                max_rails_between_cities=3,
                seed=5,
                grid_mode=False
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=10,
            obs_builder_object=GlobalObsForRailEnv())
