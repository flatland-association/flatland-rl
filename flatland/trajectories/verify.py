import ast
import warnings
from pathlib import Path

import pandas as pd
import tqdm

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.trajectories.trajectories import Trajectory
# TODO cleanup heuristic baseline
from src.observation.full_state_observation import FullStateObservationBuilder
from src.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


def verify():
    for p in tqdm.tqdm(sorted(Path("../../episodes/malfunction_deadlock_avoidance_heuristics/").rglob("**/TrainMovementEvents.trains_positions.tsv"))):

        df = pd.read_csv(str(p), delimiter="\t")
        df["direction"] = df['position'].apply(lambda x: ast.literal_eval(x)[1])
        df["position"] = df['position'].apply(lambda x: ast.literal_eval(x)[0])
        # TODO surely, there is a more elegant way
        df["position"] = df['position'].apply(lambda x: x if x is not None else "None")
        position_counts = df[df["position"] != "None"].groupby("env_time").agg({"position": ['nunique', 'count']}).reset_index()
        non_exclusives = position_counts[position_counts[("position", "nunique")] != position_counts[("position", "count")]]
        level = p.parent.parent.name
        test = p.parent.parent.parent.name
        data_dir = p.resolve().parent.parent
        if len(non_exclusives) > 0:
            warnings.warn(str(p))
            print(non_exclusives)

            ep_id = f"{test}_{level}"

            try_fix(p.parent.parent, ep_id)


def try_fix(data_dir, ep_id):
    data_dir = (data_dir / "serialised_state" / f"{ep_id}.pkl").resolve()
    output_dir = Path("/tmp/blup")
    output_dir.mkdir(exist_ok=True)
    policy_pkg = ""
    policy_cls = "DeadLockAvoidancePolicy"
    env, _ = RailEnvPersister.load_new(str(data_dir))
    env: RailEnv = env
    env.obs_builder = FullStateObservationBuilder()
    # TODO improve
    env.obs_builder.env = env
    Trajectory.create_from_policy(
        policy=DeadLockAvoidancePolicy(),
        data_dir=output_dir,
        env=env
    )


if __name__ == '__main__':
    # verify()


    # 52%|█████▎    | 21/40 [00:00<00:00, 21.42it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_02/Level_1/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 75%|███████▌  | 30/40 [00:02<00:01,  7.41it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_0/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 78%|███████▊  | 31/40 [00:03<00:01,  5.05it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_1/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 80%|████████  | 32/40 [00:03<00:02,  3.95it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_2/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 82%|████████▎ | 33/40 [00:03<00:01,  3.81it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_3/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 85%|████████▌ | 34/40 [00:04<00:01,  3.64it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_4/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 90%|█████████ | 36/40 [00:04<00:01,  3.00it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_6/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 92%|█████████▎| 37/40 [00:05<00:01,  2.51it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_7/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 95%|█████████▌| 38/40 [00:05<00:00,  2.70it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_8/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 98%|█████████▊| 39/40 [00:06<00:00,  2.61it/s]/Users/che/workspaces/flatland-rl-2/flatland/trajectories/verify.py:19: UserWarning: ../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_9/event_logs/TrainMovementEvents.trains_positions.tsv
    # warnings.warn(str(p))
    # 100%|██████████| 40/40 [00:06<00:00,  5.96it/s]


    failed = [
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_02/Level_1", "Test_02_Level_1"),
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_0", "Test_03_Level_0"),
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_1", "Test_03_Level_1"),
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2"),
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_3", "Test_03_Level_3"),
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_4", "Test_03_Level_4"),
        ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_6", "Test_03_Level_6"),
        # ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_7", "Test_03_Level_7"),
        # ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_8", "Test_03_Level_8"),
        # ("../../episodes/malfunction_deadlock_avoidance_heuristics/Test_03/Level_9", "Test_03_Level_9"),
    ]
    for data_dir, ep_id in failed:
        try_fix(
            data_dir=Path(data_dir).resolve(),
            ep_id=ep_id
        )
