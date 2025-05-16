"""
Helpers to regenerate existing benchmark/regression episodes after fix to stepping function.

To generate new trajectories from policy, use `Trajectory.generate_trajectories_from_metadata()`.
"""
import ast
import shutil
import warnings
from pathlib import Path

import pandas as pd
import tqdm

from flatland.envs.persistence import RailEnvPersister
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.trajectories import Trajectory
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


def verify_trajectories(base_path: Path):
    for p in tqdm.tqdm(sorted(base_path.rglob("**/TrainMovementEvents.trains_positions.tsv"))):

        df = pd.read_csv(str(p), delimiter="\t")
        df["direction"] = df['position'].apply(lambda x: ast.literal_eval(x)[1])
        df["position"] = df['position'].apply(lambda x: ast.literal_eval(x)[0])
        # TODO surely, there is a more elegant way
        df["position"] = df['position'].apply(lambda x: x if x is not None else "None")
        position_counts = df[df["position"] != "None"].groupby("env_time").agg({"position": ['nunique', 'count']}).reset_index()
        non_exclusives = position_counts[position_counts[("position", "nunique")] != position_counts[("position", "count")]]
        level = p.parent.parent.name
        test = p.parent.parent.parent.name
        if len(non_exclusives) > 0:
            warnings.warn(str(p))
            print(non_exclusives)

            ep_id = f"{test}_{level}"

            regen_trajectories(p.parent.parent, ep_id)


def regen_trajectories(data_dir, ep_id, output_dir):
    data_dir = (data_dir / "serialised_state" / f"{ep_id}.pkl").resolve()

    output_dir.mkdir(exist_ok=True)
    shutil.rmtree(output_dir / "event_logs", ignore_errors=True)
    env, _ = RailEnvPersister.load_new(str(data_dir))
    return Trajectory.create_from_policy(
        policy=DeadLockAvoidancePolicy(),
        data_dir=output_dir,
        env=env,
        snapshot_interval=0,
        ep_id=ep_id
    )


if __name__ == '__main__':
    base_path = Path("../../episodes/malfunction_deadlock_avoidance_heuristics/")
    verify_trajectories(base_path=base_path)
    l = [
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
    ]
    for data_dir, ep_id in l:
        trajectory = regen_trajectories(
            data_dir=Path("/Users/che/workspaces/flatland-scenarios/trajectories/" + data_dir).resolve(),
            ep_id=ep_id,
            output_dir=Path(
                f"/Users/che/workspaces/flatland-scenarios/trajectories/malfunction_deadlock_avoidance_heuristics/{ep_id.replace('_Level', '/Level')}")
        )
        TrajectoryEvaluator(trajectory).evaluate()
