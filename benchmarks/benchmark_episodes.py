import os
from pathlib import Path

import pytest

from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator

DOWNLOAD_INSTRUCTIONS = "Download from https://github.com/flatland-association/flatland-scenarios/raw/refs/heads/main/trajectories/FLATLAND_BENCHMARK_EPISODES_FOLDER_v3.zip and set BENCHMARK_EPISODES_FOLDER env var to extracted folder."
# zip -r FLATLAND_BENCHMARK_EPISODES_FOLDER_v3.zip 30x30\ map -x "*.DS_Store"; zip -r FLATLAND_BENCHMARK_EPISODES_FOLDER_v3.zip malfunction_deadlock_avoidance_heuristics -x "*.DS_Store"
from flatland.trajectories.trajectories import Trajectory


@pytest.mark.parametrize("data_sub_dir,ep_id", [
    ("30x30 map/10_trains", "1649ef98-e3a8-4dd3-a289-bbfff12876ce"),
    ("30x30 map/10_trains", "4affa89b-72f6-4305-aeca-e5182efbe467"),
    ("30x30 map/10_trains", "6274be74-e859-4445-9d77-97a5966704d8"),
    ("30x30 map/10_trains", "e3e789f4-56ae-495c-b448-5a165f70d9ed"),
    ("30x30 map/10_trains", "b13338c0-b1fc-418e-9bbc-5e5204adebc0"),
    ("30x30 map/10_trains", "7510b7d9-a206-42cb-801f-c72dff7093e7"),
    ("30x30 map/10_trains", "50c92e48-ec61-461b-a6c0-b88515c837a9"),
    ("30x30 map/10_trains", "570c7caf-b7ad-4ffe-89e6-5f870e48be7b"),
    ("30x30 map/10_trains", "bd360f90-0040-4bad-8479-e553411daef1"),
    ("30x30 map/10_trains", "5fe46151-9ff0-4598-b377-a8deafb48da6"),
    ("30x30 map/10_trains", "100a517b-364c-42cf-a54f-4533be408412"),
    ("30x30 map/10_trains", "0f501bec-ec75-4d2a-b7c2-5fdd6e7aad27"),
    ("30x30 map/10_trains", "df78ad59-d541-41b5-8028-e11f8528c296"),
    ("30x30 map/10_trains", "c6181b5d-ce94-4c42-87ff-cd86f32854e0"),
    ("30x30 map/10_trains", "bab6984f-34ef-4f58-9af8-4d156b5f9ae4"),
    ("30x30 map/10_trains", "bbf48024-2c39-4923-b285-227ecac6a679"),
    ("30x30 map/10_trains", "dad1d6d9-83c1-4fde-8073-b4c8fd87455d"),
    ("30x30 map/10_trains", "7dd06d03-d082-4b98-80be-5f194b7a05fd"),
    ("30x30 map/10_trains", "37c9c341-4769-4dd2-9015-730219c359df"),
    ("30x30 map/10_trains", "6e7e8453-a311-43ae-91a4-f2604951a69e"),

    ("30x30 map/15_trains", "a61843e8-b550-407b-9348-5029686cc967"),
    ("30x30 map/15_trains", "9845da2f-2366-44f6-8b25-beca522495b4"),
    ("30x30 map/15_trains", "e5a7061c-31ac-45f1-9f8a-06d58db26945"),
    ("30x30 map/15_trains", "25795962-78f4-4b63-98be-e990004f3a70"),
    ("30x30 map/15_trains", "35075eca-edac-4d14-a1a6-ea1bb0219542"),
    ("30x30 map/15_trains", "ff115fd1-02da-4be9-ad48-4facaa281d34"),
    ("30x30 map/15_trains", "3a6ecfb3-c26b-45d7-ba4f-e9ae0e46fee7"),
    ("30x30 map/15_trains", "9432f8ac-6d70-431d-9694-d9656dec98c3"),
    ("30x30 map/15_trains", "f799b6dc-f9d1-47aa-9df1-2f4833769af2"),
    ("30x30 map/15_trains", "acf4c413-5a9e-4bf5-b415-771681d5ffbd"),
    ("30x30 map/15_trains", "b35df1bc-bbfa-4c46-94f0-ee0faabc13a5"),
    ("30x30 map/15_trains", "93a569fa-9312-4bea-99f4-fc636ad5d411"),
    ("30x30 map/15_trains", "5f4448ef-eda3-47d7-a8c0-e6c001bf445d"),
    ("30x30 map/15_trains", "ab25b524-82e6-4b9a-91d6-09d9d877ac94"),
    ("30x30 map/15_trains", "f79c94fa-c3bf-44c6-a015-8cb1c525bec8"),
    ("30x30 map/15_trains", "0ee9efba-a032-4c61-9e3a-d215ef7d3cf8"),
    ("30x30 map/15_trains", "9620526c-e553-4f21-8496-7876e82a90c4"),
    ("30x30 map/15_trains", "c2f489e1-358e-4a26-8687-fd50803a653a"),
    ("30x30 map/15_trains", "17e9ab98-f869-48a7-a296-c00a6d0e98e0"),
    ("30x30 map/15_trains", "174e24d9-7db6-427a-8431-f3b23a6e2705"),

    ("30x30 map/20_trains", "57e1ebc5-947c-4314-83c7-0d6fd76b2bd3"),
    ("30x30 map/20_trains", "56a78985-588b-42d0-a972-7f8f2514c665"),
    ("30x30 map/20_trains", "83f43bff-3d3e-4be1-9051-93d546d59df1"),
    ("30x30 map/20_trains", "f1ac2bd0-843f-4d5f-b0ae-a0886f0a7a56"),
    ("30x30 map/20_trains", "650cda78-abc1-4dbf-9ed2-ecb72d7797fe"),
    ("30x30 map/20_trains", "a59c3280-84ec-4222-b72d-86af075be7d1"),
    ("30x30 map/20_trains", "26633ccf-19e5-4644-8be2-c0ae2275912e"),
    ("30x30 map/20_trains", "3b5ade57-824a-4b5b-af88-37a32134a706"),
    ("30x30 map/20_trains", "2c37706d-0702-47cf-99b5-195d31435bb4"),
    ("30x30 map/20_trains", "4974ecd7-dadb-4937-b6e8-e3f70e6a8919"),
    ("30x30 map/20_trains", "575ec1c8-2e31-48b0-ba49-083dfa1a71a2"),
    ("30x30 map/20_trains", "91b058d2-7b91-4edf-9cc0-0bcd8ff02311"),
    ("30x30 map/20_trains", "755aa5c5-5f05-44e6-87aa-3bd49abbf15a"),
    ("30x30 map/20_trains", "0f4757a1-1e92-4083-8c44-e759f9ac87e9"),
    ("30x30 map/20_trains", "c751c1c3-6d0e-46ba-a0c5-7846868e7b16"),
    ("30x30 map/20_trains", "cc2f35ca-d82b-437e-809f-ed6222b0a097"),
    ("30x30 map/20_trains", "7953da6b-3521-4cc1-a2e6-0ced898c86cc"),

    ("30x30 map/50_trains", "521a9180-4202-4656-84a4-603cbed8d435"),
    ("30x30 map/50_trains", "5df7e753-4701-48d7-8cc6-adf874876841"),
    ("30x30 map/50_trains", "fa9766cf-e681-45c3-8181-909fd8e1028e"),
    ("30x30 map/50_trains", "14e38e48-53b3-400a-94ba-a06b1fd2aa18"),
    ("30x30 map/50_trains", "2765dda3-4382-4a06-891d-14156ca894b3"),
    ("30x30 map/50_trains", "47ca1212-42db-4981-81df-0e44626bfdfb"),
    ("30x30 map/50_trains", "8a4a8e3e-36d7-4e55-83bf-fc8cd4b9f9d8"),
    ("30x30 map/50_trains", "980f7ba5-02ae-4f40-9c4a-7be392c24953"),
    ("30x30 map/50_trains", "5cdf6c1d-c02e-409c-b591-2a71efadb4d7"),
    ("30x30 map/50_trains", "73e275ed-9c94-400d-8aca-219a035db8e1"),
    ("30x30 map/50_trains", "2f548df3-b9d2-47ff-a3e7-3fa662b0e00e"),
    ("30x30 map/50_trains", "f53c1a08-502a-453f-bfa2-eec7e3a1f4af"),
    ("30x30 map/50_trains", "23de4f61-451b-49b9-ad4a-8e6ef36651e5"),
    ("30x30 map/50_trains", "c1d1c4b3-f7ea-4cf5-9985-79113f60037c"),
    ("30x30 map/50_trains", "9ae034d6-2d63-45da-903b-0d86afaa02f9"),
    ("30x30 map/50_trains", "9c036584-9ba2-4a55-a192-54c241d06187"),
    ("30x30 map/50_trains", "ba5b1cca-448e-47e4-946d-7f63ef426cbc"),
    ("30x30 map/50_trains", "968ec1c3-2e37-4937-b6f1-d7bb188a0bb6"),
    ("30x30 map/50_trains", "ec716f73-b22a-4fae-833a-7f58eed3968d"),
    ("30x30 map/50_trains", "f70202cc-2dec-4080-bfef-7884ab47b1b4"),

    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_0", "Test_00_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_1", "Test_00_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_2", "Test_00_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_3", "Test_00_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_4", "Test_00_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_5", "Test_00_Level_5"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_6", "Test_00_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_7", "Test_00_Level_7"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_8", "Test_00_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_9", "Test_00_Level_9"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_0", "Test_01_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_1", "Test_01_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_2", "Test_01_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_3", "Test_01_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_4", "Test_01_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_5", "Test_01_Level_5"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_6", "Test_01_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_7", "Test_01_Level_7"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_8", "Test_01_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_9", "Test_01_Level_9"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_0", "Test_02_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_1", "Test_02_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_2", "Test_02_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_3", "Test_02_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_4", "Test_02_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_5", "Test_02_Level_5"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_6", "Test_02_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_7", "Test_02_Level_7"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_8", "Test_02_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_9", "Test_02_Level_9"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_0", "Test_03_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_1", "Test_03_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_3", "Test_03_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_4", "Test_03_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_5", "Test_03_Level_5"),
])
def test_episode(data_sub_dir: str, ep_id: str):
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = Path(os.path.join(_dir, data_sub_dir))
    run_episode(data_dir, ep_id)


def run_episode(data_dir: str, ep_id: str, rendering=False, snapshot_interval=0, start_step=None):
    """
    The data is structured as follows:
        -30x30 map
            Contains the data to replay the episodes.
            - <n>_trains                                 -- for n in 10,15,20,50
                - event_logs
                    ActionEvents.discrete_action 		 -- holds set of action to be replayed for the related episodes.
                    TrainMovementEvents.trains_arrived 	 -- holds success rate for the related episodes.
                    TrainMovementEvents.trains_positions -- holds the positions for the related episodes.
                - serialised_state
                    <ep_id>.pkl                          -- Holds the pickled environment version for the episode.

    All these episodes are with constant speed of 1 and malfunctions free.

    Parameters
    ----------
    data_dir: str
        data dir with trajectory
    ep_id : str
        the episode ID
    start_step : int
        start evaluation from intermediate step (requires snapshot to be present)
    rendering : bool
        render while evaluating
    snapshot_interval : int
        interval to write pkl snapshots. 1 means at every step. 0 means never.
    """
    TrajectoryEvaluator(Trajectory(data_dir=data_dir, ep_id=ep_id)).evaluate(start_step=start_step, snapshot_interval=snapshot_interval)
