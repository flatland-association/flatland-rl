import tempfile
import time
from pathlib import Path
from typing import Any, List, Dict

from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.RailEnvPolicy import RailEnvPolicy
from flatland.envs.rail_env_action import RailEnvActions
from flatland.evaluators.evaluator_callback import FlatlandEvaluatorCallbacks
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy


class DelayPolicy(RailEnvPolicy):
    def __init__(self, initial_planning_delay: int = None, per_step_delay: int = None):
        self._elapsed_steps = -1
        self._initial_planning_delay = initial_planning_delay
        self._per_step_delay = per_step_delay

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        self._elapsed_steps += 1
        if self._elapsed_steps == 0 and self._initial_planning_delay is not None:
            time.sleep(self._initial_planning_delay)
        elif self._per_step_delay is not None:
            time.sleep(self._per_step_delay)
        return super().act_many(handles, observations)

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions.DO_NOTHING


def test_evaluator_callbacks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(env=env_generator_legacy(seed=42, )[0], policy=RandomPolicy(), data_dir=data_dir,
                                                     tqdm_kwargs={"disable": True})
        print(trajectory.trains_arrived)
        # design: actions applied at cell entry
        assert trajectory.trains_arrived.iloc[0]["normalized_reward"] == 0.47710039429784656
        assert trajectory.trains_arrived.iloc[0]["success_rate"] == 0
        # design: actions applied at cell entry
        assert trajectory.trains_rewards_dones_infos["reward"].sum() == -1724
        cb = FlatlandEvaluatorCallbacks()
        TrajectoryEvaluator(trajectory, cb).evaluate(tqdm_kwargs={"disable": True})
        # design: actions applied at cell entry
        assert cb.get_evaluation() == {'normalized_reward': 0.47710039429784656,
                                       'percentage_complete': 0.0,
                                       'reward': -1724,
                                       'termination_cause': None}
