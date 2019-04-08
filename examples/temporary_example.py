import random
import numpy as np
import matplotlib.pyplot as plt

from flatland.core.env import RailEnv
from flatland.utils.rail_env_generator import *

random.seed(100)
np.random.seed(100)


def pyplot_draw_square(center, size, color):
    x0 = center[0] - size/2
    x1 = center[0] + size/2
    y0 = center[1] - size/2
    y1 = center[1] + size/2
    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color)


def pyplot_render_env(env):
    cell_size = 10

    plt.figure()

    # Draw cells grid
    grid_color = [0.95, 0.95, 0.95]
    for r in range(env.height+1):
        plt.plot([0, (env.width+1)*cell_size],
                 [-r*cell_size, -r*cell_size], color=grid_color)
    for c in range(env.width+1):
        plt.plot([c*cell_size, c*cell_size],
                 [0, -(env.height+1)*cell_size], color=grid_color)

    # Draw each cell independently
    for r in range(env.height):
        for c in range(env.width):
            trans_ = env.rail[r][c]

            x0 = c*cell_size
            x1 = (c+1)*cell_size
            y0 = -r*cell_size
            y1 = -(r+1)*cell_size

            coords = [((x0+x1) / 2.0, y0), (x1, (y0+y1) / 2.0),
                      ((x0+x1) / 2.0, y1), (x0, (y0+y1) / 2.0)]

            for orientation in range(4):
                from_ori = (orientation + 2) % 4
                from_ = coords[from_ori]

                # Special Case 7, with a single bit; terminate at center
                nbits = 0
                tmp = trans_

                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1

                if nbits == 1:
                    from_ = ((x0+x1) / 2.0, (y0+y1) / 2.0)

                moves = env.t_utils.get_transitions_from_orientation(
                         env.rail[r][c], orientation)
                for moves_i in range(4):
                    if moves[moves_i]:
                        to = coords[moves_i]
                        plt.plot([from_[0], to[0]], [from_[1], to[1]], 'k')

    # Draw each agent + its orientation + its target
    cmap = plt.get_cmap('hsv', lut=env.number_of_agents+1)
    for i in range(env.number_of_agents):
        pyplot_draw_square((env.agents_position[i][1] * cell_size+cell_size/2,
                           -env.agents_position[i][0] * cell_size-cell_size/2),
                           cell_size / 8, cmap(i))
    for i in range(env.number_of_agents):
        pyplot_draw_square((env.agents_target[i][1] * cell_size+cell_size/2,
                           -env.agents_target[i][0] * cell_size-cell_size/2),
                           cell_size / 3, [c for c in cmap(i)])

        # orientation is a line connecting the center of the cell to the side
        # of the square of the agent
        new_position = env._new_position(env.agents_position[i],
                                         env.agents_direction[i])
        new_position = ((new_position[0]+env.agents_position[i][0])/2 *
                        cell_size,
                        (new_position[1]+env.agents_position[i][1])/2 *
                        cell_size)

        plt.plot([env.agents_position[i][1] * cell_size + cell_size/2,
                 new_position[1] + cell_size/2],
                 [-env.agents_position[i][0] * cell_size-cell_size/2,
                 -new_position[0] - cell_size/2], color=cmap(i), linewidth=2.0)

    plt.xlim([0, env.width * cell_size])
    plt.ylim([-env.height * cell_size, 0])
    plt.show()


# Example generate a random rail
rail = generate_random_rail(20, 20)

env = RailEnv(rail, number_of_agents=10)
env.reset()

pyplot_render_env(env)


# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)]]

rail = generate_rail_from_manual_specifications(specs)
env = RailEnv(rail, number_of_agents=1)

handle = env.get_agent_handles()

env.reset()

env.agents_position = [[1, 4]]
env.agents_target = [[1, 1]]
env.agents_direction = [1]

pyplot_render_env(env)


print("Manual control: s=perform step, q=quit, [agent id] [1-2-3 action] \
       (turnleft+move, move to front, turnright+move)")
for step in range(100):
    cmd = input(">> ")
    cmds = cmd.split(" ")

    action_dict = {}

    i = 0
    while i < len(cmds):
        if cmds[i] == 'q':
            import sys
            sys.exit()
        elif cmds[i] == 's':
            obs, all_rewards, done, _ = env.step(action_dict)
            action_dict = {}
            print("Rewards: ", all_rewards, "  [done=", done, "]")
        else:
            agent_id = int(cmds[i])
            action = int(cmds[i+1])
            action_dict[agent_id] = action
            i = i+1
        i += 1

    pyplot_render_env(env)
