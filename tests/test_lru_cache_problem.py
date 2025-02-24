from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

maxsize = 1000000
env_42_hits = 53
env_42_cache_size = 1137
env_43_hits = 60
env_43_cache_size = 1108
grid_size = 30 * 30
hits_42_900_43_900_42_900_43_900 = env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits + grid_size + env_43_hits + grid_size
cache_size_42_43_42_43 = env_42_cache_size + env_43_cache_size + env_42_cache_size + env_43_cache_size
cache_size_42_43_42 = env_42_cache_size + env_43_cache_size + env_42_cache_size
cache_size_42_43 = env_42_cache_size + env_43_cache_size


def test_lru_load():
    # avoid side effects from other tests
    _clear_all_lru_caches()

    # (1) new env with seed 42
    env_42 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=1),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=42)
    assert _info_lru_cache() == (0, 0, maxsize, 0)
    env_42.reset(random_seed=42)
    assert _info_lru_cache() == (env_42_hits, env_42_cache_size, maxsize, env_42_cache_size)
    transitions_42 = {}
    for r in range(30):
        for c in range(30):
            transitions_42[(r, c)] = env_42.rail.get_full_transitions(r, c)
    assert _info_lru_cache() == (env_42_hits + grid_size, env_42_cache_size, maxsize, env_42_cache_size)

    # (1b) save env with seed 42
    RailEnvPersister.save(env_42, "env_42.pkl")
    assert _info_lru_cache() == (env_42_hits + grid_size, env_42_cache_size, maxsize, env_42_cache_size)

    # (2) new env with seed 43
    env_43 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=2),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=43)
    assert _info_lru_cache() == (env_42_hits + grid_size, env_42_cache_size, maxsize, env_42_cache_size)
    env_43.reset(random_seed=43)
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits, 2245, maxsize, 2245)

    transitions_43 = {}
    for r in range(30):
        for c in range(30):
            transitions_43[(r, c)] = env_43.rail.get_full_transitions(r, c)
    # reset clears the cache, so the transitions are indeed different
    assert set(transitions_42.items()) != set(transitions_43.items())
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits + grid_size, cache_size_42_43, maxsize, cache_size_42_43)

    # (3) second new env with seed 42
    env_42_bis = RailEnv(width=30, height=30,
                         rail_generator=sparse_rail_generator(seed=1),
                         line_generator=sparse_line_generator(), number_of_agents=2, random_seed=42)
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits + grid_size, cache_size_42_43, maxsize, cache_size_42_43)
    env_42_bis.reset(random_seed=42)
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits, cache_size_42_43_42, maxsize,
                                 cache_size_42_43_42)

    transitions_42_bis = {}
    for r in range(30):
        for c in range(30):
            transitions_42_bis[(r, c)] = env_42.rail.get_full_transitions(r, c)
    # sanity check: same seed gives same transitions
    assert set(transitions_42.items()) == set(transitions_42_bis.items())
    assert _info_lru_cache() == (
        env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits + grid_size, cache_size_42_43_42, maxsize,
        cache_size_42_43_42)

    # (4) populate cache with infrastructure from seed 43
    env_43 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=2),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=43)
    env_43.reset(random_seed=43)

    assert _info_lru_cache() == (
        env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits + grid_size + env_43_hits,
        cache_size_42_43_42_43, maxsize,
        cache_size_42_43_42_43)

    transitions_43 = {}
    for r in range(30):
        for c in range(30):
            transitions_43[(r, c)] = env_43.rail.get_full_transitions(r, c)
    # reset clears the cache, so the transitions are indeed different
    assert set(transitions_42.items()) != set(transitions_43.items())
    assert _info_lru_cache() == (
        hits_42_900_43_900_42_900_43_900,
        cache_size_42_43_42_43, maxsize,
        cache_size_42_43_42_43)

    # (5) load env_42 from file
    RailEnvPersister.load(env_43, "env_42.pkl")
    # load does no reset -> no additional caching
    assert _info_lru_cache() == (hits_42_900_43_900_42_900_43_900, cache_size_42_43_42_43, maxsize, cache_size_42_43_42_43)
    env_42_tri = env_43
    transitions_42_tri = {}
    for r in range(30):
        for c in range(30):
            transitions_42_tri[(r, c)] = env_42_tri.rail.get_full_transitions(r, c)
    # load() now invalidates cache correctly
    assert set(transitions_42.items()) == set(transitions_42_tri.items())
    # 30*30 additional misses are cached:
    assert _info_lru_cache() == (hits_42_900_43_900_42_900_43_900, cache_size_42_43_42_43 + grid_size, maxsize, cache_size_42_43_42_43 + grid_size)


def test_lru_load_new():
    # avoid side effects from other tests
    _clear_all_lru_caches()

    # (1) new env with seed 42
    env_42 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=1),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=42)
    assert _info_lru_cache() == (0, 0, maxsize, 0)
    env_42.reset(random_seed=42)
    assert _info_lru_cache() == (env_42_hits, env_42_cache_size, maxsize, env_42_cache_size)
    transitions_42 = {}
    for r in range(30):
        for c in range(30):
            transitions_42[(r, c)] = env_42.rail.get_full_transitions(r, c)
    assert _info_lru_cache() == (env_42_hits + grid_size, env_42_cache_size, maxsize, env_42_cache_size)

    # (1b) save env with seed 42
    RailEnvPersister.save(env_42, "env_42.pkl")
    assert _info_lru_cache() == (env_42_hits + grid_size, env_42_cache_size, maxsize, env_42_cache_size)

    # (2) new env with seed 43
    env_43 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=2),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=43)
    assert _info_lru_cache() == (env_42_hits + grid_size, env_42_cache_size, maxsize, env_42_cache_size)
    env_43.reset(random_seed=43)
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits, cache_size_42_43, maxsize, cache_size_42_43)
    transitions_43 = {}
    for r in range(30):
        for c in range(30):
            transitions_43[(r, c)] = env_43.rail.get_full_transitions(r, c)
    # reset clears the cache, so the transitions are indeed different
    assert set(transitions_42.items()) != set(transitions_43.items())
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits + grid_size, cache_size_42_43, maxsize, cache_size_42_43)

    # (3) second new env with seed 42
    env_42_bis = RailEnv(width=30, height=30,
                         rail_generator=sparse_rail_generator(seed=1),
                         line_generator=sparse_line_generator(), number_of_agents=2, random_seed=42)

    env_42_bis.reset(random_seed=42)
    assert _info_lru_cache() == (
        env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits, cache_size_42_43_42, maxsize, cache_size_42_43_42)
    transitions_42_bis = {}
    for r in range(30):
        for c in range(30):
            transitions_42_bis[(r, c)] = env_42.rail.get_full_transitions(r, c)
    # sanity check: same seed gives same transitions
    assert set(transitions_42.items()) == set(transitions_42_bis.items())
    assert _info_lru_cache() == (env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits + grid_size, cache_size_42_43_42, maxsize, cache_size_42_43_42)

    # (4) populate cache with infrastructure from seed 43
    env_43 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=2),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=43)
    env_43.reset(random_seed=43)

    assert _info_lru_cache() == (
        env_42_hits + grid_size + env_43_hits + grid_size + env_42_hits + grid_size + env_43_hits, cache_size_42_43_42_43, maxsize, cache_size_42_43_42_43)

    transitions_43 = {}
    for r in range(30):
        for c in range(30):
            transitions_43[(r, c)] = env_43.rail.get_full_transitions(r, c)
    # reset clears the cache, so the transitions are indeed different
    assert set(transitions_42.items()) != set(transitions_43.items())
    assert _info_lru_cache() == (hits_42_900_43_900_42_900_43_900, cache_size_42_43_42_43, maxsize, cache_size_42_43_42_43)

    # (5) load_new() env_42 from file
    # N.B.line `env.rail = GridTransitionMap(1, 1)` in `load_new` has side effect of clearing infrastructure cache.
    env_42_tri, _ = RailEnvPersister.load_new("env_42.pkl")
    # load does no reset -> no additional caching
    assert _info_lru_cache() == (hits_42_900_43_900_42_900_43_900, cache_size_42_43_42_43, maxsize, cache_size_42_43_42_43)
    transitions_42_tri = {}
    for r in range(30):
        for c in range(30):
            transitions_42_tri[(r, c)] = env_42_tri.rail.get_full_transitions(r, c)
    # load_new() invalidates cache (so env_43 transitions are cleared)
    assert set(transitions_42.items()) == set(transitions_42_tri.items())
    # 900 additional misses are cached:
    assert _info_lru_cache() == (hits_42_900_43_900_42_900_43_900, cache_size_42_43_42_43 + grid_size, maxsize, cache_size_42_43_42_43 + grid_size)


def _info_lru_cache():
    import functools
    import gc

    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)]
    # print(wrappers)
    for wrapper in wrappers:
        if wrapper.__name__ == "get_full_transitions":
            print(f"{wrapper.__name__} {wrapper.cache_info()}")
            return wrapper.cache_info()


# https://stackoverflow.com/questions/40273767/clear-all-lru-cache-in-python
def _clear_all_lru_caches():
    import functools
    import gc

    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)]

    for wrapper in wrappers:
        wrapper.cache_clear()
