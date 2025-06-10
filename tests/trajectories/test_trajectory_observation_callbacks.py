import tempfile
from pathlib import Path

from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.trajectories.trajectory_observation_callbacks import TrajectoryObservationCallbacks
from tests.trajectories.test_policy_runner import RandomPolicy


def test_observation_callbacks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=0)
        assert len(list(trajectory.data_dir.rglob("**/*step*.pkl"))) == 0
        assert len(list(trajectory.data_dir.rglob("**/*obs*.pkl"))) == 0

        data_dir_override = data_dir_override = data_dir / "override"
        TrajectoryEvaluator(trajectory, TrajectoryObservationCallbacks(trajectory, data_dir_override)).evaluate()
        assert len(list(data_dir.rglob("**/*step*.pkl"))) == 0
        assert len(list(data_dir.rglob("**/*obs*.pkl"))) == 472
        assert len(list((data_dir / "outputs" / "serialised_state").rglob("*obs*.pkl"))) == 0
        assert len(list((data_dir_override / "serialised_state").rglob("*obs*.pkl"))) == 472

        TrajectoryEvaluator(trajectory, TrajectoryObservationCallbacks(trajectory)).evaluate(snapshot_interval=1)
        assert len(list(data_dir.rglob("**/*step*.pkl"))) == 472
        assert len(list(data_dir.rglob("**/*obs*.pkl"))) == 2 * 472
        assert len(list((data_dir / "outputs" / "serialised_state").rglob("*step*.pkl"))) == 472
        assert len(list((data_dir / "outputs" / "serialised_state").rglob("*obs*.pkl"))) == 472
        assert len(list((data_dir_override / "serialised_state").rglob("*obs*.pkl"))) == 472
