import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_episodes import run_episode, DOWNLOAD_INSTRUCTIONS
from flatland.env_generation.env_generator import env_generator
from flatland.trajectories.trajectories import EVENT_LOGS_SUBDIR, OUTPUTS_SUBDIR, Trajectory


# run a subset of episodes for regression
@pytest.mark.parametrize("data_sub_dir,ep_id,run_from_intermediate,skip_rewards_dones_infos", [
    ("30x30 map/10_trains", "1649ef98-e3a8-4dd3-a289-bbfff12876ce", True, True),
    ("30x30 map/10_trains", "4affa89b-72f6-4305-aeca-e5182efbe467", True, True),

    ("30x30 map/15_trains", "a61843e8-b550-407b-9348-5029686cc967", True, True),
    ("30x30 map/15_trains", "9845da2f-2366-44f6-8b25-beca522495b4", True, True),

    ("30x30 map/20_trains", "57e1ebc5-947c-4314-83c7-0d6fd76b2bd3", True, True),
    ("30x30 map/20_trains", "56a78985-588b-42d0-a972-7f8f2514c665", True, True),

    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_8", "Test_00_Level_8", True, False),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_3", "Test_01_Level_3", True, False),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_6", "Test_02_Level_6", True, False),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_8", "Test_02_Level_8", False, False),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_1", "Test_03_Level_1", False, False),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2", False, False),
])
def test_episode(data_sub_dir: str, ep_id: str, run_from_intermediate: bool, skip_rewards_dones_infos: bool):
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = Path(os.path.join(_dir, data_sub_dir))

    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copytree(data_dir, tmpdirname, dirs_exist_ok=True)

        # run with snapshots to outputs/serialised_state directory
        run_episode(Path(tmpdirname), ep_id, snapshot_interval=1 if run_from_intermediate else 0, skip_rewards_dones_infos=skip_rewards_dones_infos)

        if run_from_intermediate:
            # copy actions etc. to outputs subfolder, so outputs subfolder becomes a proper trajectory data dir.
            shutil.copytree(os.path.join(data_dir, EVENT_LOGS_SUBDIR), os.path.join(tmpdirname, OUTPUTS_SUBDIR, EVENT_LOGS_SUBDIR), dirs_exist_ok=True)

            # start episode from a snapshot to ensure snapshot contains full state!
            run_episode(Path(tmpdirname) / OUTPUTS_SUBDIR, ep_id, start_step=np.random.randint(0, 50), skip_rewards_dones_infos=skip_rewards_dones_infos)


def test_restore_episode():
    """
    Test that refactorings in env generation does not introduce changes in behaviour with the default parameters.

    See <a href="https://github.com/flatland-association/flatland-scenarios/tree/main?tab=readme-ov-file#changelog-2">changelog</a>.
    """
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)

    metadata_csv = Path(f"{_dir}/malfunction_deadlock_avoidance_heuristics/metadata.csv").resolve()
    metadata = pd.read_csv(metadata_csv)
    for i, (k, v) in enumerate(metadata.iterrows()):
        ep_id = f'{v["test_id"]}_{v["env_id"]}'
        print(ep_id)
        if i >= 40:
            break
        env_regen, _, _ = env_generator(
            n_agents=v["n_agents"],
            x_dim=v["x_dim"],
            y_dim=v["y_dim"],
            n_cities=v["n_cities"],
            max_rail_pairs_in_city=v["max_rail_pairs_in_city"],
            grid_mode=v["grid_mode"],
            max_rails_between_cities=v["max_rails_between_cities"],
            malfunction_duration_min=v["malfunction_duration_min"],
            malfunction_duration_max=v["malfunction_duration_max"],
            malfunction_interval=v["malfunction_interval"],
            speed_ratios={1.0: 0.25,
                          0.5: 0.25,
                          0.33: 0.25,
                          0.25: 0.25},
            seed=v["seed"],
        )

        data_sub_dir = f'malfunction_deadlock_avoidance_heuristics/{v["test_id"]}/{v["env_id"]}'

        data_dir = Path(os.path.join(_dir, data_sub_dir))

        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.copytree(data_dir, tmpdirname, dirs_exist_ok=True)

            t = Trajectory(data_dir=Path(tmpdirname), ep_id=ep_id)
            env_restored = t.restore_episode()

            # TODO poor man's state comparison for now
            assert [a.position for a in env_regen.agents] == [a.position for a in env_restored.agents]
