import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark_episodes import run_episode, DOWNLOAD_INSTRUCTIONS
from flatland.trajectories.trajectories import EVENT_LOGS_SUBDIR, OUTPUTS_SUBDIR


# run a subset of episodes for regression
@pytest.mark.parametrize("data_sub_dir,ep_id,run_from_intermediate", [
    ("30x30 map/10_trains", "1649ef98-e3a8-4dd3-a289-bbfff12876ce", True),
    ("30x30 map/10_trains", "4affa89b-72f6-4305-aeca-e5182efbe467", True),

    ("30x30 map/15_trains", "a61843e8-b550-407b-9348-5029686cc967", True),
    ("30x30 map/15_trains", "9845da2f-2366-44f6-8b25-beca522495b4", True),

    ("30x30 map/20_trains", "57e1ebc5-947c-4314-83c7-0d6fd76b2bd3", True),
    ("30x30 map/20_trains", "56a78985-588b-42d0-a972-7f8f2514c665", True),

    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_8", "Test_00_Level_8", True),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_3", "Test_01_Level_3", True),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_6", "Test_02_Level_6", True),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_8", "Test_02_Level_8", False),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_1", "Test_03_Level_1", False),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2", False),
])
def test_episode(data_sub_dir: str, ep_id: str, run_from_intermediate: bool):
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = Path(os.path.join(_dir, data_sub_dir))

    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copytree(data_dir, tmpdirname, dirs_exist_ok=True)

        # run with snapshots to outputs/serialised_state directory
        run_episode(Path(tmpdirname), ep_id, snapshot_interval=1 if run_from_intermediate else 0)

        if run_from_intermediate:
            # copy actions etc. to outputs subfolder, so outputs subfolder becomes a proper trajectory data dir.
            shutil.copytree(os.path.join(data_dir, EVENT_LOGS_SUBDIR), os.path.join(tmpdirname, OUTPUTS_SUBDIR, EVENT_LOGS_SUBDIR), dirs_exist_ok=True)

            # start episode from a snapshot to ensure snapshot contains full state!
            run_episode(Path(tmpdirname) / OUTPUTS_SUBDIR, ep_id, start_step=np.random.randint(0, 50))
