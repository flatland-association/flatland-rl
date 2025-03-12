import os
import shutil
import tempfile

import numpy as np
import pytest

from benchmarks.benchmark_episodes import run_episode, DOWNLOAD_INSTRUCTIONS


# run a subset of episodes for regression
@pytest.mark.parametrize("data_sub_dir,ep_id", [
    ("30x30 map/10_trains", "1649ef98-e3a8-4dd3-a289-bbfff12876ce"),
    ("30x30 map/10_trains", "4affa89b-72f6-4305-aeca-e5182efbe467"),

    ("30x30 map/15_trains", "a61843e8-b550-407b-9348-5029686cc967"),
    ("30x30 map/15_trains", "9845da2f-2366-44f6-8b25-beca522495b4"),

    ("30x30 map/20_trains", "57e1ebc5-947c-4314-83c7-0d6fd76b2bd3"),
    ("30x30 map/20_trains", "56a78985-588b-42d0-a972-7f8f2514c665"),

    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_8", "Test_00_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_3", "Test_01_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_6", "Test_02_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2"),
])
def test_episode(data_sub_dir: str, ep_id: str):
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = os.path.join(_dir, data_sub_dir)

    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copytree(data_dir, tmpdirname, dirs_exist_ok=True)

        # run with snapshots
        run_episode(data_dir, ep_id, snapshot_interval=1)
        # start episode from a snapshot to ensure snapshot contains full state!
        run_episode(data_dir, ep_id, start_step=np.random.randint(0,50))
