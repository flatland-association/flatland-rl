import os
from pathlib import Path

import numpy as np

from benchmarks.benchmark_episodes import DOWNLOAD_INSTRUCTIONS
from flatland.evaluators.trajectory_analysis import data_frame_for_trajectories


def test_data_frame_for_trajectories():
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = Path(_dir) / "malfunction_deadlock_avoidance_heuristics"
    _, _, all_trains_arrived, _, _, _ = data_frame_for_trajectories(data_dir)
    assert len(all_trains_arrived) == 5 * 10  # 5 Tests (Test_00, ..., Test_04) x 10 scenarios.
    mean_success_rate_ = all_trains_arrived["success_rate"].mean()
    assert np.isclose(mean_success_rate_, 0.6635285714285714)
