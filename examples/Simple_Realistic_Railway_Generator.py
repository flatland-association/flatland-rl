import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_from_nodes
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import RailGenerator, RailGeneratorProduct
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


def realistic_rail_generator(num_cities=5, city_size=10, max_number_of_station_tracks=4,
                             max_number_of_connecting_tracks=4,
                             seed=0, print_out_info=True) -> RailGenerator:
    """
    This is a level generator which generates a realistic rail configurations

    :param num_cities: Number of city node (can hold trainstations)
    :param seed: Random Seed
    :return:
        -------
    numpy.ndarray of type numpy.uint16
        The matrix with the correct 16-bit bitmaps for each cell.
    """

    def subtract_pos(nodeA, nodeB):
        return (nodeA[0] - nodeB[0], nodeA[1] - nodeB[1])

    def add_pos(nodeA, nodeB):
        return (nodeA[0] + nodeB[0], nodeA[1] + nodeB[1])

    def make_orthogonal_pos(node):
        return (node[1], -node[0])

    def get_norm_pos(node):
        return np.sqrt(node[0] * node[0] + node[1] * node[1])

    def normalize_pos(node):
        n = get_norm_pos(node)
        if n > 0.0:
            n = 1 / n
        return scale_pos(node, n)

    def scale_pos(node, scalar):
        return (node[0] * scalar, node[1] * scalar)

    def round_pos(node):
        return (int(np.round(node[0])), int(np.round(node[1])))

    def ceil_pos(node):
        return (int(np.ceil(node[0])), int(np.ceil(node[1])))

    def bound_pos(node, min_value, max_value):
        return (max(min_value, min(max_value, node[0])), max(min_value, min(max_value, node[1])))

    def do_generate_city_locations(width, height):

        X = int(np.floor(max(1, width - 2 * max_number_of_connecting_tracks - 1) / city_size))
        Y = int(np.floor(max(1, height - 2 * max_number_of_connecting_tracks - 1) / city_size))

        max_num_cities = min(num_cities, X * Y)

        cities_at = np.random.choice(X * Y, max_num_cities, False)
        cities_at = np.sort(cities_at)
        if print_out_info:
            print("max. nbr of cities with given configuration is:", max_num_cities)

        x = np.floor(cities_at / Y)
        y = cities_at - x * Y
        xs = (x * city_size + max_number_of_connecting_tracks)
        ys = (y * city_size + max_number_of_connecting_tracks)

        generate_city_locations = [[(int(xs[i]), int(ys[i])), (int(xs[i]), int(ys[i]))] for i in range(len(xs))]
        return generate_city_locations, max_num_cities

    def do_orient_cities(generate_city_locations):
        for i in range(len(generate_city_locations)):
            # station main orientation  (horizontal or vertical
            add_pos_val = (city_size, 0)
            if np.random.choice(2) == 0:
                add_pos_val = (0, city_size)
            generate_city_locations[i][1] = add_pos(generate_city_locations[i][1], add_pos_val)
        return generate_city_locations

    def do_tracks_between_start_end_points(rail_trans, rail_array, generate_city_locations):
        nodes_to_added = []
        station_slots = [[] for i in range(len(generate_city_locations))]

        for city_loop in range(len(generate_city_locations)):
            # Connect train station to the correct node
            number_of_connecting_tracks = np.random.choice(max(0, max_number_of_connecting_tracks)) + 1

            for ct in range(number_of_connecting_tracks):
                for kLoop in range(2):
                    org_start_node = generate_city_locations[city_loop][kLoop]

                    a = generate_city_locations[city_loop][0]
                    b = generate_city_locations[city_loop][1]
                    org_end_node = scale_pos(add_pos(a, b), 0.5)

                    ortho_trans = make_orthogonal_pos(normalize_pos(subtract_pos(a, b)))
                    s = (ct - number_of_connecting_tracks / 2.0)
                    start_node = ceil_pos(add_pos(org_start_node, scale_pos(ortho_trans, s)))
                    end_node = ceil_pos(org_end_node)
                    end_node = ceil_pos(add_pos(org_end_node, scale_pos(ortho_trans, s)))

                    connection = connect_from_nodes(rail_trans, rail_array, start_node, end_node)
                    if len(connection) > 0:
                        nodes_to_added.append(start_node)
                        nodes_to_added.append(end_node)
                        # place in the center of path a station slot
                        station_slots[city_loop].append(connection[int(np.floor(len(connection)/2))])

        return nodes_to_added, station_slots,

    def generator(width, height, num_agents, num_resets=0) -> RailGeneratorProduct:
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)
        np.random.seed(seed + num_resets)

        agent_start_targets_nodes = []
        # generate city locations
        generate_city_locations, max_num_cities = do_generate_city_locations(width, height)
        # apply orientation to cities (horizontal, vertical)
        generate_city_locations = do_orient_cities(generate_city_locations)
        # generate city topology
        nodes_to_added, station_slots = do_tracks_between_start_end_points(rail_trans,
                                                                                    rail_array,
                                                                                    generate_city_locations)

        train_stations = [[] for i in range(max_num_cities)]
        for i in range(max_num_cities):
            for j in range(len(station_slots[i])):
                train_stations[i].append(station_slots[i][j])

        # ----------------------------------------------------------------------------------
        # fix all transition at starting / ending points (mostly add a dead end, if missing)
        for i in range(len(nodes_to_added)):
            grid_map.fix_transitions(nodes_to_added[i])

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


env = RailEnv(width=70,
              height=70,
              rail_generator=realistic_rail_generator(num_cities=100,  # Number of cities in map
                                                      seed=0  # Random seed
                                                      ),
              schedule_generator=sparse_schedule_generator(),
              number_of_agents=5,
              obs_builder_object=GlobalObsForRailEnv())

# reset to initialize agents_static
env_renderer = RenderTool(env, gl="PILSVG", screen_width=1400, screen_height=1000)
while True:
    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

env_renderer.close_window()
