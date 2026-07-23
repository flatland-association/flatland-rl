import math

import numpy as np

from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


def _make_simple_env():
    # _ _ _

    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

    rail_map = np.array(
        [[dead_end_from_east] + [horizontal_straight] + [dead_end_from_west]], dtype=np.uint16)
    rail = RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map

    city_positions = [(0, 2), (0, 1)]
    train_stations = [
        [((0, 1), 0)],
        [((0, 2), 0)],
    ]
    city_orientations = [1, 0]
    agents_hints = {
        'num_agents': 1,
        'city_positions': city_positions,
        'train_stations': train_stations,
        'city_orientations': city_orientations
    }
    optionals = {'agents_hints': agents_hints}

    return RailEnv(width=rail_map.shape[1],
                   height=rail_map.shape[0],
                   rail_generator=rail_from_grid_transition_map(rail, optionals),
                   line_generator=sparse_line_generator(),
                   number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2,
                                                        predictor=ShortestPathPredictorForRailEnv(max_depth=10)),
                   )


def test_walker():
    env = _make_simple_env()
    env.reset()

    # set initial position and direction for testing...
    env.agents[0].position = (0, 1)
    env.agents[0].direction = 1
    env.agents[0].target = (0, 0)
    # reset to set agents from agents_static
    # env.reset(False, False)
    env.distance_map._compute(env.agents, env.rail)

    print(env.distance_map.get()[(0, *[0, 1], 1)])
    assert env.distance_map.get()[(0, *[0, 1], 1)] == 3
    print(env.distance_map.get()[(0, *[0, 2], 3)])
    assert env.distance_map.get()[(0, *[0, 2], 1)] == 2


def test_distances_cache_cleared_on_reset():
    """
    Regression test: `ConfigurationDistanceMap.reset()` must clear `self.distances` (the internal
    source->target BFS scratch cache). Without this, a stale distance value left over from a previous
    episode survives into the next episode's BFS and can get picked up by the `min()` comparison in
    `DistanceMapWalker._get_and_update_neighbors`, silently corrupting the newly computed distance for
    a rail topology that is regenerated on every `env.reset()` (the normal RL training loop pattern).
    """
    env = _make_simple_env()
    env.reset(random_seed=1)

    env.distance_map.get()  # trigger _compute(), populating self.distances
    assert len(env.distance_map.distances) > 0

    poisoned_key = next(iter(env.distance_map.distances))
    correct_value = env.distance_map.distances[poisoned_key]
    env.distance_map.distances[poisoned_key] = -999

    env.reset(random_seed=1)
    # reset() must clear the cache immediately - the poisoned entry must be gone, not just overwritten later
    assert env.distance_map.distances.get(poisoned_key, math.inf) != -999

    env.distance_map.get()  # recompute on the identical (same seed) rail/target assignment
    assert env.distance_map.distances[poisoned_key] == correct_value


def test_loaded_distance_map_is_used_for_shortest_paths():
    """
    Regression test: a distance map populated via `.set(...)` (as `RailEnv`/`RailEnvPersister` do when
    loading a precomputed distance map from a saved rail/env file, bypassing `_compute()`/the BFS walk)
    must still be usable by `get_shortest_paths()`/`get_agent_distance()`. Reading only the internal BFS
    scratch cache (which stays empty in this scenario) instead of the actual populated per-agent storage
    silently produced `None` paths for every agent despite a perfectly valid loaded distance map.
    """
    env = _make_simple_env()
    env.reset(random_seed=1)  # seed known to place agent 0 with a valid (non-None) shortest path

    precomputed = env.distance_map.get()
    expected_paths = env.distance_map.get_shortest_paths()
    assert expected_paths[0] is not None

    fresh_dm = DistanceMap(agents=env.agents, env_height=env.height, env_width=env.width)
    fresh_dm.set(precomputed)
    fresh_dm.reset(env.agents, env.rail)

    actual_paths = fresh_dm.get_shortest_paths()
    assert actual_paths[0] is not None
    assert actual_paths[0] == expected_paths[0]


def test_agent_with_no_valid_targets_does_not_crash():
    """
    Regression test: `get_agent_distance()`'s minimum over an agent's target configurations must not
    raise `ValueError: min() arg is an empty sequence` for an agent whose `targets` set has become empty
    (e.g. via `RailEnvPersister.set_full_state`'s post-load filtering to only rail-valid configurations,
    which can filter out every target). The old pre-refactor code had no such min() and never crashed
    on this; `get_shortest_paths()` should keep returning `None` (no path found) for that agent instead.
    """
    env = RailEnv(width=25, height=25,
                  rail_generator=sparse_rail_generator(max_num_cities=3, seed=1),
                  line_generator=sparse_line_generator(),
                  number_of_agents=2)
    env.reset(random_seed=1)
    env.agents[0].targets = set()

    paths = env.distance_map.get_shortest_paths()
    assert paths[0] is None
