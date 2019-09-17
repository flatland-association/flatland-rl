import copy
import os
import warnings

import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d, IntVector2DArrayType
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_from_nodes, connect_nodes, connect_rail
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import RailGenerator, RailGeneratorProduct
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

FloatArrayType = []


def realistic_rail_generator(num_cities=5,
                             city_size=10,
                             allowed_rotation_angles=None,
                             max_number_of_station_tracks=4,
                             nbr_of_switches_per_station_track=2,
                             connect_max_nbr_of_shortes_city=4,
                             do_random_connect_stations=False,
                             seed=0,
                             print_out_info=True) -> RailGenerator:
    """
    This is a level generator which generates a realistic rail configurations

    :param print_out_info:
    :param num_cities: Number of city node
    :param city_size: Length of city measure in cells
    :param allowed_rotation_angles: Rotate the city (around center)
    :param max_number_of_station_tracks: max number of tracks per station
    :param nbr_of_switches_per_station_track: number of switches per track (max)
    :param connect_max_nbr_of_shortes_city: max number of connecting track between stations
    :param do_random_connect_stations : if false connect the stations along the grid (top,left -> down,right), else rand
    :param seed: Random Seed
    :print_out_info : print debug info
    :return:
        -------
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """

    def do_generate_city_locations(width: int,
                                   height: int,
                                   intern_city_size: int,
                                   intern_max_number_of_station_tracks: int) -> (IntVector2DArrayType, int):

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

    def do_orient_cities(generate_city_locations: IntVector2DArrayType, intern_city_size: int,
                         rotation_angles_set: FloatArrayType):
        for i in range(len(generate_city_locations)):
            # station main orientation  (horizontal or vertical
            rot_angle = np.random.choice(rotation_angles_set)
            add_pos_val = Vec2d.scale(Vec2d.rotate((1, 0), rot_angle),
                                      int(max(1.0, (intern_city_size - 3) / 2)))
            generate_city_locations[i][0] = Vec2d.add(generate_city_locations[i][1], add_pos_val)
            add_pos_val = Vec2d.scale(Vec2d.rotate((1, 0), 180 + rot_angle),
                                      int(max(1.0, (intern_city_size - 3) / 2)))
            generate_city_locations[i][1] = Vec2d.add(generate_city_locations[i][1], add_pos_val)
        return generate_city_locations

    def create_stations_from_city_locations(rail_trans: RailEnvTransitions,
                                            grid_map: GridTransitionMap,
                                            generate_city_locations: IntVector2DArrayType,
                                            intern_max_number_of_station_tracks: int) -> (IntVector2DArrayType,
                                                                                          IntVector2DArrayType,
                                                                                          IntVector2DArrayType,
                                                                                          IntVector2DArrayType,
                                                                                          IntVector2DArrayType):

        nodes_added = []
        start_nodes_added = [[] for _ in range(len(generate_city_locations))]
        end_nodes_added = [[] for _ in range(len(generate_city_locations))]
        station_slots = [[] for _ in range(len(generate_city_locations))]
        station_tracks = [[[] for _ in range(intern_max_number_of_station_tracks)] for _ in range(len(
            generate_city_locations))]

        station_slots_cnt = 0

        for city_loop in range(len(generate_city_locations)):
            # Connect train station to the correct node
            number_of_connecting_tracks = np.random.choice(max(0, intern_max_number_of_station_tracks)) + 1
            track_id = 0
            for ct in range(number_of_connecting_tracks):
                org_start_node = generate_city_locations[city_loop][0]
                org_end_node = generate_city_locations[city_loop][1]

                ortho_trans = Vec2d.make_orthogonal(
                    Vec2d.normalize(Vec2d.subtract(org_start_node, org_end_node)))
                s = (ct - number_of_connecting_tracks / 2.0)
                start_node = Vec2d.ceil(
                    Vec2d.add(org_start_node, Vec2d.scale(ortho_trans, s)))
                end_node = Vec2d.ceil(
                    Vec2d.add(org_end_node, Vec2d.scale(ortho_trans, s)))

                connection = connect_from_nodes(rail_trans, grid_map, start_node, end_node)
                if len(connection) > 0:
                    nodes_added.append(start_node)
                    nodes_added.append(end_node)

                    start_nodes_added[city_loop].append(start_node)
                    end_nodes_added[city_loop].append(end_node)

                    # place in the center of path a station slot
                    station_slots[city_loop].append(connection[int(np.floor(len(connection) / 2))])
                    station_slots_cnt += 1

                    station_tracks[city_loop][track_id] = connection
                    track_id += 1
                else:
                    if print_out_info:
                        print("create_stations_from_city_locations : connect_from_nodes -> no path found")

        if print_out_info:
            print("max nbr of station slots with given configuration is:", station_slots_cnt)

        return nodes_added, station_slots, start_nodes_added, end_nodes_added, station_tracks

    def create_switches_at_stations(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                                    station_tracks: IntVector2DArrayType,
                                    nodes_added: IntVector2DArrayType,
                                    intern_nbr_of_switches_per_station_track: int) -> IntVector2DArrayType:

        for k_loop in range(intern_nbr_of_switches_per_station_track):
            for city_loop in range(len(station_tracks)):
                k = k_loop + city_loop
                datas = station_tracks[city_loop]
                if len(datas) > 1:
                    track = datas[0]
                    if len(track) > 0:
                        if k % 2 == 0:
                            x = int(np.random.choice(int(len(track)/2))+1)
                        else:
                            x = len(track) - int(np.random.choice(int(len(track)/2))+1)
                        start_node = track[x]
                        for i in np.arange(1, len(datas)):
                            track = datas[i]
                            if len(track) > 1:
                                if k % 2 == 0:
                                    x = x + 2
                                    if len(track) <= x:
                                        x = 1
                                else:
                                    x = x - 2
                                    if x < 2:
                                        x = len(track) - 1
                                end_node = track[x]
                                connection = connect_rail(rail_trans, grid_map, start_node, end_node)
                                print(start_node, end_node, "-->", connection)
                                if len(connection) == 0:
                                    if print_out_info:
                                        print("create_switches_at_stations : connect_rail -> no path found")
                                    start_node = datas[i][0]
                                    end_node = datas[i - 1][0]
                                    connection = connect_rail(rail_trans, grid_map, start_node, end_node)

                                nodes_added.append(start_node)
                                nodes_added.append(end_node)

                                if k % 2 == 0:
                                    x = x + 2
                                    if len(track) <= x:
                                        x = 1
                                else:
                                    x = x - 2
                                    if x < 2:
                                        x = len(track) - 2
                                start_node = track[x]

        return nodes_added

    def create_graph_edge(from_city_index: int, to_city_index: int) -> (int, int, int):
        return from_city_index, to_city_index, np.inf

    def calc_nbr_of_graphs(graph: []) -> ([], []):
        for i in range(len(graph)):
            for j in range(len(graph)):
                a = graph[i]
                b = graph[j]
                connected = False
                if a[0] == b[0] or a[1] == b[0]:
                    connected = True
                if a[0] == b[1] or a[1] == b[1]:
                    connected = True

                if connected:
                    a = [graph[i][0], graph[i][1], graph[i][2]]
                    b = [graph[j][0], graph[j][1], graph[j][2]]
                    graph[i] = (graph[i][0], graph[i][1], min(np.min(a), np.min(b)))
                    graph[j] = (graph[j][0], graph[j][1], min(np.min(a), np.min(b)))
                else:
                    a = [graph[i][0], graph[i][1], graph[i][2]]
                    graph[i] = (graph[i][0], graph[i][1], np.min(a))
                    b = [graph[j][0], graph[j][1], graph[j][2]]
                    graph[j] = (graph[j][0], graph[j][1], np.min(b))

        graph_ids = []
        for i in range(len(graph)):
            graph_ids.append(graph[i][2])
        if print_out_info:
            print("************* NBR of graphs:", len(np.unique(graph_ids)))
        return graph, np.unique(graph_ids).astype(int)

    def connect_sub_graphs(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                           org_s_nodes: IntVector2DArrayType,
                           org_e_nodes: IntVector2DArrayType,
                           city_edges: IntVector2DArrayType,
                           nodes_added: IntVector2DArrayType):
        _, graphids = calc_nbr_of_graphs(city_edges)
        if len(graphids) > 0:
            for i in range(len(graphids) - 1):
                connection = []
                iteration_counter = 0
                while len(connection) == 0 and iteration_counter < 100:
                    s_nodes = copy.deepcopy(org_s_nodes)
                    e_nodes = copy.deepcopy(org_e_nodes)
                    start_nodes = s_nodes[graphids[i]]
                    end_nodes = e_nodes[graphids[i + 1]]
                    start_node = start_nodes[np.random.choice(len(start_nodes))]
                    end_node = end_nodes[np.random.choice(len(end_nodes))]
                    # TODO : removing, what the hell is going on, why we have to set rail_array -> transition to zero
                    # TODO : before we can call connect_rail. If we don't reset the transistion to zero -> no rail
                    # TODO : will be generated.
                    grid_map.grid[start_node] = 0
                    grid_map.grid[end_node] = 0
                    connection = connect_rail(rail_trans, grid_map, start_node, end_node)
                    if len(connection) > 0:
                        nodes_added.append(start_node)
                        nodes_added.append(end_node)
                    else:
                        if print_out_info:
                            print("connect_sub_graphs : connect_rail -> no path found")

                    iteration_counter += 1

    def connect_stations(rail_trans: RailEnvTransitions,
                         grid_map: GridTransitionMap,
                         org_s_nodes: IntVector2DArrayType,
                         org_e_nodes: IntVector2DArrayType,
                         nodes_added: IntVector2DArrayType,
                         intern_connect_max_nbr_of_shortes_city: int):
        city_edges = []

        s_nodes = copy.deepcopy(org_s_nodes)
        e_nodes = copy.deepcopy(org_e_nodes)

        for nbr_connected in range(intern_connect_max_nbr_of_shortes_city):
            for city_loop in range(len(s_nodes)):
                sns = s_nodes[city_loop]
                for start_node in sns:
                    min_distance = np.inf
                    end_node = None
                    cl = 0
                    for city_loop_find_shortest in range(len(e_nodes)):
                        if city_loop_find_shortest == city_loop:
                            continue
                        ens = e_nodes[city_loop_find_shortest]
                        for en in ens:
                            d = Vec2d.get_euclidean_distance(start_node, en)
                            if d < min_distance:
                                min_distance = d
                                end_node = en
                                cl = city_loop_find_shortest

                    if end_node is not None:
                        tmp_trans_sn = grid_map.grid[start_node]
                        tmp_trans_en = grid_map.grid[end_node]
                        grid_map.grid[start_node] = 0
                        grid_map.grid[end_node] = 0
                        connection = connect_rail(rail_trans, grid_map, start_node, end_node)
                        if len(connection) > 0:
                            s_nodes[city_loop].remove(start_node)
                            e_nodes[cl].remove(end_node)

                            edge = create_graph_edge(city_loop, cl)
                            if city_loop > cl:
                                edge = create_graph_edge(cl, city_loop)
                            if not (edge in city_edges):
                                city_edges.append(edge)
                            nodes_added.append(start_node)
                            nodes_added.append(end_node)
                        else:
                            if print_out_info:
                                print("connect_stations : connect_rail -> no path found")

                            grid_map.grid[start_node] = tmp_trans_sn
                            grid_map.grid[end_node] = tmp_trans_en

        connect_sub_graphs(rail_trans, grid_map, org_s_nodes, org_e_nodes, city_edges, nodes_added)

    def connect_random_stations(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                                start_nodes_added: IntVector2DArrayType,
                                end_nodes_added: IntVector2DArrayType,
                                nodes_added: IntVector2DArrayType,
                                intern_connect_max_nbr_of_shortes_city: int):
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
            max_input_output = min(intern_connect_max_nbr_of_shortes_city, max_input_output)

            idx_s_nodes = np.random.choice(np.arange(len(s_nodes)), len(s_nodes), False)
            idx_e_nodes = np.random.choice(np.arange(len(e_nodes)), len(e_nodes), False)

            if len(idx_s_nodes) < max_input_output:
                idx_s_nodes = np.append(idx_s_nodes, np.random.choice(np.arange(len(s_nodes)), max_input_output - len(
                    idx_s_nodes)))
            if len(idx_e_nodes) < max_input_output:
                idx_e_nodes = np.append(idx_e_nodes,
                                        np.random.choice(np.arange(len(idx_e_nodes)), max_input_output - len(
                                            idx_e_nodes)))

            if len(idx_s_nodes) > intern_connect_max_nbr_of_shortes_city:
                idx_s_nodes = np.random.choice(idx_s_nodes, intern_connect_max_nbr_of_shortes_city, False)
            if len(idx_e_nodes) > intern_connect_max_nbr_of_shortes_city:
                idx_e_nodes = np.random.choice(idx_e_nodes, intern_connect_max_nbr_of_shortes_city, False)

            for i in range(max_input_output):
                start_node = s_nodes[idx_s_nodes[i]]
                end_node = e_nodes[idx_e_nodes[i]]
                grid_map.grid[start_node] = 0
                grid_map.grid[end_node] = 0
                connection = connect_nodes(rail_trans, grid_map, start_node, end_node)
                if len(connection) > 0:
                    nodes_added.append(start_node)
                    nodes_added.append(end_node)
                else:
                    if print_out_info:
                        print("connect_random_stations : connect_nodes -> no path found")

    def generator(width: int, height: int, num_agents: int, num_resets: int = 0) -> RailGeneratorProduct:
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        grid_map.grid.fill(0)
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

        intern_connect_max_nbr_of_shortes_city = connect_max_nbr_of_shortes_city
        if connect_max_nbr_of_shortes_city < 1:
            warnings.warn("min intern_connect_max_nbr_of_shortes_city requried to be > 1!")
            intern_connect_max_nbr_of_shortes_city = 1
        if print_out_info:
            print("intern_connect_max_nbr_of_shortes_city:", intern_connect_max_nbr_of_shortes_city)

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
            create_stations_from_city_locations(rail_trans, grid_map,
                                                generate_city_locations,
                                                intern_max_number_of_station_tracks)
        # build switches
        # TODO remove true/false block
        if True:
            create_switches_at_stations(rail_trans, grid_map, station_tracks, nodes_added,
                                        intern_nbr_of_switches_per_station_track)

        # ----------------------------------------------------------------------------------
        # connect stations
        # TODO remove true/false block
        if True:
            if do_random_connect_stations:
                connect_random_stations(rail_trans, grid_map, s_nodes, e_nodes, nodes_added,
                                        intern_connect_max_nbr_of_shortes_city)
            else:
                connect_stations(rail_trans, grid_map, s_nodes, e_nodes, nodes_added,
                                 intern_connect_max_nbr_of_shortes_city)

        # ----------------------------------------------------------------------------------
        # fix all transition at starting / ending points (mostly add a dead end, if missing)
        # TODO i would like to remove the fixing stuff.
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


if os.path.exists("./../render_output/"):
    for itrials in range(1000):
        print(itrials, "generate new city")
        np.random.seed(itrials)
        env = RailEnv(width=40 + np.random.choice(100),
                      height=40 + np.random.choice(100),
                      rail_generator=realistic_rail_generator(num_cities=1000,
                                                              city_size=10 + np.random.choice(10),
                                                              allowed_rotation_angles=np.arange(-180, 180, 90),
                                                              max_number_of_station_tracks=1 + np.random.choice(4),
                                                              nbr_of_switches_per_station_track=2,
                                                              connect_max_nbr_of_shortes_city=2 + np.random.choice(4),
                                                              do_random_connect_stations=False,
                                                              # Number of cities in map
                                                              seed=itrials,  # Random seed
                                                              print_out_info=True
                                                              ),
                      schedule_generator=sparse_schedule_generator(),
                      number_of_agents=0,
                      obs_builder_object=GlobalObsForRailEnv())

        # reset to initialize agents_static
        env_renderer = RenderTool(env, gl="PILSVG", screen_width=1400, screen_height=1000,
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX)
        cnt = 0
        while cnt < 10:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            cnt += 1

        if os.path.exists("./../render_output/"):
            env_renderer.gl.save_image(
                os.path.join(
                    "./../render_output/",
                    "flatland_frame_{:04d}_{:04d}.png".format(itrials, 0)
                ))

        env_renderer.close_window()
