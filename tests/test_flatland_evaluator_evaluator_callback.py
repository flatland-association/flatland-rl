import tempfile
import time
from pathlib import Path
from typing import Any

from flatland.core.policy import Policy
from flatland.evaluators.evaluator_callback import FlatlandEvaluatorCallbacks
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.trajectories import Trajectory
from tests.trajectories.test_trajectories import RandomPolicy


class DelayPolicy(Policy):
    def __init__(self, initial_planning_delay: int = None, per_step_delay: int = None):
        self._elapsed_steps = -1
        self._initial_planning_delay = initial_planning_delay
        self._per_step_delay = per_step_delay

    def act(self, handle: int, observation: Any, **kwargs) -> Any:
        if handle == 0:
            self._elapsed_steps += 1
            if self._elapsed_steps == 0 and self._initial_planning_delay is not None:
                time.sleep(self._initial_planning_delay)
            elif self._per_step_delay is not None:
                time.sleep(self._per_step_delay)
        return {}


def test_evaluator_callbacks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = Trajectory.create_from_policy(policy=RandomPolicy(), data_dir=data_dir)
        cb = FlatlandEvaluatorCallbacks()
        TrajectoryEvaluator(trajectory, cb).evaluate()
        assert cb.get_evaluation() == {'normalized_reward': -0.5417045799211404,
                                       'percentage_complete': 0.0,
                                       'reward': -1786,
                                       'termination_cause': None}
