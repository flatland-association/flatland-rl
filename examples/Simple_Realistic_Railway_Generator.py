import os
import time
import warnings

import numpy as np

from flatland.core.grid.grid4_utils import mirror
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_from_nodes, connect_nodes, connect_rail
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import RailGenerator, RailGeneratorProduct
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


class Vec2dOperations:
    def subtract_pos(nodeA, nodeB):
        """
        vector operation : nodeA - nodeB

        :param nodeA: tuple with coordinate (x,y) or 2d vector
        :param nodeB: tuple with coordinate (x,y) or 2d vector
        :return:
            -------
        tuple with coordinate (x,y) or 2d vector
        """
        return (nodeA[0] - nodeB[0], nodeA[1] - nodeB[1])

    def add_pos(nodeA, nodeB):
        """
        vector operation : nodeA + nodeB

        :param nodeA: tuple with coordinate (x,y) or 2d vector
        :param nodeB: tuple with coordinate (x,y) or 2d vector
        :return:
            -------
        tuple with coordinate (x,y) or 2d vector
        """
        return (nodeA[0] + nodeB[0], nodeA[1] + nodeB[1])

    def make_orthogonal_pos(node):
        """
        vector operation : rotates the 2D vector +90°

        :param node: tuple with coordinate (x,y) or 2d vector
        :return:
            -------
        tuple with coordinate (x,y) or 2d vector
        """
        return (node[1], -node[0])

    def get_norm_pos(node):
        """
        calculates the euclidean norm of the 2d vector

        :param node: tuple with coordinate (x,y) or 2d vector
        :return:
            -------
        tuple with coordinate (x,y) or 2d vector
        """
        return np.sqrt(node[0] * node[0] + node[1] * node[1])

    def normalize_pos(node):
        """
        normalize the 2d vector = v/|v|

        :param node: tuple with coordinate (x,y) or 2d vector
        :return:
            -------
        tuple with coordinate (x,y) or 2d vector
        """
        n = Vec2dOperations.get_norm_pos(node)
        if n > 0.0:
            n = 1 / n
        return Vec2dOperations.scale_pos(node, n)

    def scale_pos(node, scalar):
        """
         scales the 2d vector = node * scale

         :param node: tuple with coordinate (x,y) or 2d vector
         :param scale: scalar to scale
         :return:
             -------
         tuple with coordinate (x,y) or 2d vector
         """
        return (node[0] * scalar, node[1] * scalar)

    def round_pos(node):
        """
         rounds the x and y coordinate and convert them to an integer values

         :param node: tuple with coordinate (x,y) or 2d vector
         :return:
             -------
         tuple with coordinate (x,y) or 2d vector
         """
        return (int(np.round(node[0])), int(np.round(node[1])))

    def ceil_pos(node):
        """
         ceiling the x and y coordinate and convert them to an integer values

         :param node: tuple with coordinate (x,y) or 2d vector
         :return:
             -------
         tuple with coordinate (x,y) or 2d vector
         """
        return (int(np.ceil(node[0])), int(np.ceil(node[1])))

    def bound_pos(node, min_value, max_value):
        """
         force the values x and y to be between min_value and max_value

         :param node: tuple with coordinate (x,y) or 2d vector
         :param min_value: scalar value
         :param max_value: scalar value
         :return:
             -------
         tuple with coordinate (x,y) or 2d vector
         """
        return (max(min_value, min(max_value, node[0])), max(min_value, min(max_value, node[1])))

    def rotate_pos(node, rot_in_degree):
        """
         rotate the 2d vector with given angle in degree

         :param node: tuple with coordinate (x,y) or 2d vector
         :param rot_in_degree:  angle in degree
         :return:
             -------
         tuple with coordinate (x,y) or 2d vector
         """
        alpha = rot_in_degree / 180.0 * np.pi
        x0 = node[0]
        y0 = node[1]
        x1 = x0 * np.cos(alpha) - y0 * np.sin(alpha)
        y1 = x0 * np.sin(alpha) + y0 * np.cos(alpha)
        return (x1, y1)


def min_max_cut(min_v, max_v, v):
    return max(min_v, min(max_v, v))


def add_rail(width, height, grid_map, pt_from, pt_via, pt_to, bAddRemove=True):
    gRCTrans = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC

    lrcStroke = [[min_max_cut(0, height - 1, pt_from[0]),
                  min_max_cut(0, width - 1, pt_from[1])],
                 [min_max_cut(0, height - 1, pt_via[0]),
                  min_max_cut(0, width - 1, pt_via[1])],
                 [min_max_cut(0, height - 1, pt_to[0]),
                  min_max_cut(0, width - 1, pt_to[1])]]

    rc3Cells = np.array(lrcStroke[:3])  # the 3 cells
    rcMiddle = rc3Cells[1]  # the middle cell which we will update
    bDeadend = np.all(lrcStroke[0] == lrcStroke[2])  # deadend means cell 0 == cell 2

    # get the 2 row, col deltas between the 3 cells, eg [[-1,0],[0,1]] = North, East
    rc2Trans = np.diff(rc3Cells, axis=0)

    # get the direction index for the 2 transitions
    liTrans = []
    for rcTrans in rc2Trans:
        # gRCTrans - rcTrans gives an array of vector differences between our rcTrans
        # and the 4 directions stored in gRCTrans.
        # Where the vector difference is zero, we have a match...
        # np.all detects where the whole row,col vector is zero.
        # argwhere gives the index of the zero vector, ie the direction index
        iTrans = np.argwhere(np.all(gRCTrans - rcTrans == 0, axis=1))
        if len(iTrans) > 0:
            iTrans = iTrans[0][0]
            liTrans.append(iTrans)

    # check that we have two transitions
    if len(liTrans) == 2:
        # Set the transition
        # Set the transition
        # If this transition spans 3 cells, it is not a deadend, so remove any deadends.
        # The user will need to resolve any conflicts.
        grid_map.set_transition((*rcMiddle, liTrans[0]),
                                liTrans[1],
                                bAddRemove,
                                remove_deadends=not bDeadend)

        # Also set the reverse transition
        # use the reversed outbound transition for inbound
        # and the reversed inbound transition for outbound
        grid_map.set_transition((*rcMiddle, mirror(liTrans[1])),
                                mirror(liTrans[0]), bAddRemove, remove_deadends=not bDeadend)


def realistic_rail_generator(num_cities=5,
                             city_size=10,
                             allowed_rotation_angles=[0, 90],
                             max_number_of_station_tracks=4,
                             nbr_of_switches_per_station_track=2,
                             max_number_of_connecting_tracks=4,
                             do_random_connect_stations=False,
                             seed=0,
                             print_out_info=True) -> RailGenerator:
    """
    This is a level generator which generates a realistic rail configurations

    :param num_cities: Number of city node
    :param city_size: Length of city measure in cells
    :param allowed_rotation_angles: Rotate the city (around center)
    :param max_number_of_station_tracks: max number of tracks per station
    :param nbr_of_switches_per_station_track: number of switches per track (max)
    :param max_number_of_connecting_tracks: max number of connecting track between stations
    :param do_random_connect_stations : if false connect the stations along the grid (top,left -> down,right), else rand
    :param seed: Random Seed
    :print_out_info : print debug info
    :return:
        -------
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """

    def do_generate_city_locations(width, height, intern_city_size, intern_max_number_of_station_tracks):

        X = int(np.floor(max(1, height - 2 * intern_max_number_of_station_tracks - 1) / intern_city_size))
        Y = int(np.floor(max(1, width - 2 * intern_max_number_of_station_tracks - 1) / intern_city_size))

        max_num_cities = min(num_cities, X * Y)

        cities_at = np.random.choice(X * Y, max_num_cities, False)
        cities_at = np.sort(cities_at)
        if print_out_info:
            print("max nbr of cities with given configuration is:", max_num_cities)

        x = np.floor(cities_at / Y)
        y = cities_at - x * Y
        xs = (x * intern_city_size + intern_max_number_of_station_tracks) + intern_city_size / 2
        ys = (y * intern_city_size + intern_max_number_of_station_tracks) + intern_city_size / 2

        generate_city_locations = [[(int(xs[i]), int(ys[i])), (int(xs[i]), int(ys[i]))] for i in range(len(xs))]
        return generate_city_locations, max_num_cities

    def do_orient_cities(generate_city_locations, intern_city_size, allowed_rotation_angles):
        for i in range(len(generate_city_locations)):
            # station main orientation  (horizontal or vertical
            rot_angle = np.random.choice(allowed_rotation_angles)
            add_pos_val = Vec2dOperations.scale_pos(Vec2dOperations.rotate_pos((1, 0), rot_angle),
                                                    (max(1, (intern_city_size - 3) / 2)))
            generate_city_locations[i][0] = Vec2dOperations.add_pos(generate_city_locations[i][1], add_pos_val)
            add_pos_val = Vec2dOperations.scale_pos(Vec2dOperations.rotate_pos((1, 0), 180 + rot_angle),
                                                    (max(1, (intern_city_size - 3) / 2)))
            generate_city_locations[i][1] = Vec2dOperations.add_pos(generate_city_locations[i][1], add_pos_val)
        return generate_city_locations

    def create_stations_from_city_locations(rail_trans, rail_array, generate_city_locations,
                                            intern_max_number_of_station_tracks):
        nodes_added = []
        start_nodes_added = [[] for i in range(len(generate_city_locations))]
        end_nodes_added = [[] for i in range(len(generate_city_locations))]
        station_slots = [[] for i in range(len(generate_city_locations))]
        station_tracks = [[[] for j in range(intern_max_number_of_station_tracks)] for i in range(len(
            generate_city_locations))]

        station_slots_cnt = 0

        for city_loop in range(len(generate_city_locations)):
            # Connect train station to the correct node
            number_of_connecting_tracks = np.random.choice(max(0, intern_max_number_of_station_tracks)) + 1
            for ct in range(number_of_connecting_tracks):
                org_start_node = generate_city_locations[city_loop][0]
                org_end_node = generate_city_locations[city_loop][1]

                ortho_trans = Vec2dOperations.make_orthogonal_pos(
                    Vec2dOperations.normalize_pos(Vec2dOperations.subtract_pos(org_start_node, org_end_node)))
                s = (ct - number_of_connecting_tracks / 2.0)
                start_node = Vec2dOperations.ceil_pos(
                    Vec2dOperations.add_pos(org_start_node, Vec2dOperations.scale_pos(ortho_trans, s)))
                end_node = Vec2dOperations.ceil_pos(
                    Vec2dOperations.add_pos(org_end_node, Vec2dOperations.scale_pos(ortho_trans, s)))

                connection = connect_from_nodes(rail_trans, rail_array, start_node, end_node)
                if len(connection) > 0:
                    nodes_added.append(start_node)
                    nodes_added.append(end_node)

                    start_nodes_added[city_loop].append(start_node)
                    end_nodes_added[city_loop].append(end_node)

                    # place in the center of path a station slot
                    station_slots[city_loop].append(connection[int(np.floor(len(connection) / 2))])
                    station_slots_cnt += 1

                    station_tracks[city_loop][ct] = connection

        if print_out_info:
            print("max nbr of station slots with given configuration is:", station_slots_cnt)

        return nodes_added, station_slots, start_nodes_added, end_nodes_added, station_tracks

    def create_switches_at_stations(width, height, grid_map, station_tracks, nodes_added,
                                    intern_nbr_of_switches_per_station_track):
        # generate switch based on switch slot list and connect them
        for city_loop in range(len(station_tracks)):
            datas = station_tracks[city_loop]
            for data_loop in range(len(datas) - 1):
                data = datas[data_loop]
                data1 = datas[data_loop + 1]
                if len(data) > 2 and len(data1) > 2:
                    for i in np.random.choice(min(len(data1), len(data)) - 2,
                                              intern_nbr_of_switches_per_station_track):
                        add_rail(width, height, grid_map, data[i + 1], data1[i + 1], data1[i + 2], True)
                        nodes_added.append(data[i + 1])
                        nodes_added.append(data1[i + 1])
                        nodes_added.append(data1[i + 2])

        return nodes_added

    def connect_stations(rail_trans, rail_array, org_s_nodes, org_e_nodes, nodes_added,
                         inter_max_number_of_connecting_tracks):

        s_nodes = org_s_nodes.copy()
        e_nodes = org_e_nodes.copy()


        for city_loop in range(len(s_nodes)):
            for k in range(inter_max_number_of_connecting_tracks):
                sns = s_nodes[city_loop]
                old_cl = -1
                min_distance = np.inf
                end_node = None
                start_node = None
                for city_loop_find_shortest in range(len(e_nodes)):
                    if city_loop_find_shortest == city_loop:
                        continue
                    ens = e_nodes[city_loop_find_shortest]
                    for en in ens:
                        for sn in sns:
                            d = Vec2dOperations.get_norm_pos(Vec2dOperations.subtract_pos(en, sn))
                            if d < min_distance and old_cl != city_loop_find_shortest:
                                min_distance = d
                                end_node = en
                                start_node = sn
                                cl = city_loop_find_shortest

                if end_node is not None:
                    old_cl = cl
                    tmp_trans_sn = rail_array[start_node]
                    tmp_trans_en = rail_array[end_node]
                    rail_array[start_node] = 0
                    rail_array[end_node] = 0
                    connection = connect_rail(rail_trans, rail_array, start_node, end_node)
                    if len(connection) > 0:
                        s_nodes[city_loop].remove(start_node)
                        e_nodes[cl].remove(end_node)
                        nodes_added.append(start_node)
                        nodes_added.append(end_node)
                    else:
                        rail_array[start_node] = tmp_trans_sn
                        rail_array[end_node] = tmp_trans_en

    def connect_random_stations(rail_trans, rail_array, start_nodes_added, end_nodes_added, nodes_added,
                                inter_max_number_of_connecting_tracks):
        x = np.arange(len(start_nodes_added))
        random_city_idx = np.random.choice(x, len(x), False)

        # cyclic connection
        random_city_idx = np.append(random_city_idx, random_city_idx[0])

        for city_loop in range(len(random_city_idx) - 1):
            idx_a = random_city_idx[city_loop + 1]
            idx_b = random_city_idx[city_loop]
            s_nodes = start_nodes_added[idx_a]
            e_nodes = end_nodes_added[idx_b]

            max_input_output = max(len(s_nodes), len(e_nodes))
            max_input_output = min(inter_max_number_of_connecting_tracks, max_input_output)

            if do_random_connect_stations:
                idx_s_nodes = np.random.choice(np.arange(len(s_nodes)), len(s_nodes), False)
                idx_e_nodes = np.random.choice(np.arange(len(e_nodes)), len(e_nodes), False)
            else:
                idx_s_nodes = np.arange(len(s_nodes))
                idx_e_nodes = np.arange(len(e_nodes))

            if len(idx_s_nodes) < max_input_output:
                idx_s_nodes = np.append(idx_s_nodes, np.random.choice(np.arange(len(s_nodes)), max_input_output - len(
                    idx_s_nodes)))
            if len(idx_e_nodes) < max_input_output:
                idx_e_nodes = np.append(idx_e_nodes,
                                        np.random.choice(np.arange(len(idx_e_nodes)), max_input_output - len(
                                            idx_e_nodes)))

            if len(idx_s_nodes) > inter_max_number_of_connecting_tracks:
                idx_s_nodes = np.random.choice(idx_s_nodes, inter_max_number_of_connecting_tracks, False)
            if len(idx_e_nodes) > inter_max_number_of_connecting_tracks:
                idx_e_nodes = np.random.choice(idx_e_nodes, inter_max_number_of_connecting_tracks, False)

            for i in range(max_input_output):
                start_node = s_nodes[idx_s_nodes[i]]
                end_node = e_nodes[idx_e_nodes[i]]
                new_trans = rail_array[start_node] = 0
                new_trans = rail_array[end_node] = 0
                connection = connect_nodes(rail_trans, rail_array, start_node, end_node)
                if len(connection) > 0:
                    nodes_added.append(start_node)
                    nodes_added.append(end_node)

    def generator(width, height, num_agents, num_resets=0) -> RailGeneratorProduct:
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)
        np.random.seed(seed + num_resets)

        intern_city_size = city_size
        if city_size < 3:
            warnings.warn("min city_size requried to be > 3!")
            intern_city_size = 3
        if print_out_info:
            print("intern_city_size:", intern_city_size)

        intern_max_number_of_station_tracks = max_number_of_station_tracks
        if max_number_of_station_tracks < 1:
            warnings.warn("min max_number_of_station_tracks requried to be > 1!")
            intern_max_number_of_station_tracks = 1
        if print_out_info:
            print("intern_max_number_of_station_tracks:", intern_max_number_of_station_tracks)

        intern_nbr_of_switches_per_station_track = nbr_of_switches_per_station_track
        if nbr_of_switches_per_station_track < 1:
            warnings.warn("min intern_nbr_of_switches_per_station_track requried to be > 2!")
            intern_nbr_of_switches_per_station_track = 2
        if print_out_info:
            print("intern_nbr_of_switches_per_station_track:", intern_nbr_of_switches_per_station_track)

        inter_max_number_of_connecting_tracks = max_number_of_connecting_tracks
        if max_number_of_connecting_tracks < 1:
            warnings.warn("min inter_max_number_of_connecting_tracks requried to be > 1!")
            inter_max_number_of_connecting_tracks = 1
        if print_out_info:
            print("inter_max_number_of_connecting_tracks:", inter_max_number_of_connecting_tracks)

        agent_start_targets_nodes = []

        # ----------------------------------------------------------------------------------
        # generate city locations
        generate_city_locations, max_num_cities = do_generate_city_locations(width, height, intern_city_size,
                                                                             intern_max_number_of_station_tracks)

        # ----------------------------------------------------------------------------------
        # apply orientation to cities (horizontal, vertical)
        generate_city_locations = do_orient_cities(generate_city_locations, intern_city_size, allowed_rotation_angles)

        # ----------------------------------------------------------------------------------
        # generate city topology
        nodes_added, train_stations, s_nodes, e_nodes, station_tracks = \
            create_stations_from_city_locations(rail_trans, rail_array,
                                                generate_city_locations,
                                                intern_max_number_of_station_tracks)
        # build switches
        create_switches_at_stations(width, height, grid_map, station_tracks, nodes_added,
                                    intern_nbr_of_switches_per_station_track)

        # ----------------------------------------------------------------------------------
        # connect stations
        if True:
            if do_random_connect_stations:
                connect_random_stations(rail_trans, rail_array, s_nodes, e_nodes, nodes_added,
                                        inter_max_number_of_connecting_tracks)
            else:
                connect_stations(rail_trans, rail_array, s_nodes, e_nodes, nodes_added,
                                 inter_max_number_of_connecting_tracks)

        # ----------------------------------------------------------------------------------
        # fix all transition at starting / ending points (mostly add a dead end, if missing)
        for i in range(len(nodes_added)):
            grid_map.fix_transitions(nodes_added[i])

        # ----------------------------------------------------------------------------------
        # Slot availability in node
        node_available_start = []
        node_available_target = []
        for node_idx in range(max_num_cities):
            node_available_start.append(len(train_stations[node_idx]))
            node_available_target.append(len(train_stations[node_idx]))

        # Assign agents to slots
        for agent_idx in range(num_agents):
            avail_start_nodes = [idx for idx, val in enumerate(node_available_start) if val > 0]
            avail_target_nodes = [idx for idx, val in enumerate(node_available_target) if val > 0]
            if len(avail_target_nodes) == 0:
                num_agents -= 1
                continue
            start_node = np.random.choice(avail_start_nodes)
            target_node = np.random.choice(avail_target_nodes)
            tries = 0
            found_agent_pair = True
            while target_node == start_node:
                target_node = np.random.choice(avail_target_nodes)
                tries += 1
                # Test again with new start node if no pair is found (This code needs to be improved)
                if (tries + 1) % 10 == 0:
                    start_node = np.random.choice(avail_start_nodes)
                if tries > 100:
                    warnings.warn("Could not set trainstations, removing agent!")
                    found_agent_pair = False
                    break
            if found_agent_pair:
                node_available_start[start_node] -= 1
                node_available_target[target_node] -= 1
                agent_start_targets_nodes.append((start_node, target_node))
            else:
                num_agents -= 1

        return grid_map, {'agents_hints': {
            'num_agents': num_agents,
            'agent_start_targets_nodes': agent_start_targets_nodes,
            'train_stations': train_stations
        }}

    return generator


for itrials in range(100):
    print(itrials, "generate new city")
    np.random.seed(int(time.time()))
    env = RailEnv(width=100,  # 20+np.random.choice(100),
                  height=100,  # 20+np.random.choice(100),
                  rail_generator=realistic_rail_generator(num_cities=np.random.choice(10)+2,
                                                          city_size=np.random.choice(10)+10,
                                                          allowed_rotation_angles=np.arange(0,360,45),
                                                          max_number_of_station_tracks=4,
                                                          nbr_of_switches_per_station_track=2,
                                                          max_number_of_connecting_tracks=4,
                                                          do_random_connect_stations=False,
                                                          # Number of cities in map
                                                          seed=int(time.time())  # Random seed
                                                          ),
                  schedule_generator=sparse_schedule_generator(),
                  number_of_agents=1000,
                  obs_builder_object=GlobalObsForRailEnv())

    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", screen_width=1400, screen_height=1000)
    cnt = 0
    while cnt < 10:
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        cnt += 1

    env_renderer.gl.save_image(
        os.path.join(
            "./../render_output/",
            "flatland_frame_{:04d}_{:04d}.png".format(itrials, 0)
        ))

    env_renderer.close_window()
