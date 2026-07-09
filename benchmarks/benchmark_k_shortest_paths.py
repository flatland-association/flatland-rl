import itertools
import os
import pickle
from pathlib import Path

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths

DOWNLOAD_INSTRUCTIONS = "Download from https://github.com/flatland-association/ecml2026-starterkit/raw/refs/heads/main/reinforcement_learning/sampling/level_0_scenario_1.pkl and set ECML2026_PKL env var to the downloaded file."


def test_k_shortest_paths_between_all_cell_pairs():
    """
    Benchmark `get_k_shortest_paths` by computing the shortest path between every pair of rail cells (for every valid entry direction at the source) in a real ECML 2026 scenario.
    """
    pkl = os.getenv("ECML2026_PKL")
    assert pkl is not None, (DOWNLOAD_INSTRUCTIONS, pkl)
    assert os.path.exists(pkl), (DOWNLOAD_INSTRUCTIONS, pkl)

    env, _ = RailEnvPersister.load_new(pkl)
    with (Path(pkl).parent / "stations.pkl").open('rb') as f:
        data = pickle.load(f)

    train_stations = data["train_stations"]
    # take one track per station only for the benchmark
    train_stations_flat = [station[i % len(station)] for i, station in enumerate(train_stations)]
    for (source_position, source_direction), (target_position, _) in itertools.product(train_stations_flat, train_stations_flat):
        get_k_shortest_paths(
            env=env,
            source_position=source_position,
            source_direction=source_direction,
            target_position=target_position,
        )
