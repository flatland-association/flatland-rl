from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


def test_lru_failing():
    # seed 42
    env_42 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=1),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=42)

    env_42.reset(random_seed=42)
    transitions_42 = {}
    for r in range(30):
        for c in range(30):
            transitions_42[(r, c)] = env_42.rail.get_full_transitions(r, c)

    RailEnvPersister.save(env_42, "env_42.pkl")

    # seed 43
    env_43 = RailEnv(width=30, height=30,
                     rail_generator=sparse_rail_generator(seed=2),
                     line_generator=sparse_line_generator(), number_of_agents=2, random_seed=43)

    env_43.reset(random_seed=43)
    transitions_43 = {}
    for r in range(30):
        for c in range(30):
            transitions_43[(r, c)] = env_43.rail.get_full_transitions(r, c)
    # reset clears the cache, so the transitions
    assert set(transitions_42.items()) != set(transitions_43.items())

    # seed 42 bis
    env_42_bis = RailEnv(width=30, height=30,
                         rail_generator=sparse_rail_generator(seed=1),
                         line_generator=sparse_line_generator(), number_of_agents=2, random_seed=42)

    env_42_bis.reset(random_seed=42)
    transitions_42_bis = {}
    for r in range(30):
        for c in range(30):
            transitions_42_bis[(r, c)] = env_42.rail.get_full_transitions(r, c)
    # sanity check: same seed gives same transitions
    assert set(transitions_42.items()) == set(transitions_42_bis.items())

    # populate cache with infrastructure from seed 43
    RailEnv(width=30, height=30,
            rail_generator=sparse_rail_generator(seed=2),
            line_generator=sparse_line_generator(), number_of_agents=2, random_seed=44).reset()

    # load env_42 from file
    env_42_tri, _ = RailEnvPersister.load_new("env_42.pkl")

    transitions_42_tri = {}
    for r in range(30):
        for c in range(30):
            transitions_42_tri[(r, c)] = env_42_tri.rail.get_full_transitions(r, c)
    # this fails! the two sets should be different, but are not as cache kicks in.
    assert set(transitions_42.items()) != set(transitions_42_tri.items())
