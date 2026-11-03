from typing import List

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.travelwise.passenger_env_policies import BoardForTargetPassengerPolicy
from flatland.envs.travelwise.travelwise_policy_runner import TravelWisePolicyRunner
from tests.flatland.envs.travelwise.test_passenger_env_scenario import _make_scenario_env


class _RecordingCallbacks(FlatlandCallbacks):
    """ Records the callbacks called, with the passenger env's elapsed steps. """

    def __init__(self):
        self.calls: List[str] = []

    def on_episode_start(self, *, env=None, **kwargs):
        self.calls.append(f"start {env.rail_env._elapsed_steps}")

    def on_episode_step(self, *, env=None, **kwargs):
        self.calls.append(f"step {env.rail_env._elapsed_steps}")

    def on_episode_end(self, *, env=None, **kwargs):
        self.calls.append(f"end {env.rail_env._elapsed_steps}")


def test_travelwise_policy_runner_runs_an_episode():
    """
    `run()` on the scenario of `test_passenger_env_scenario_single_rides`: returns the sum of each passenger's rewards
    there, and calls `on_episode_start` after the reset, `on_episode_step` after each of the 18 steps (the rail env is
    done at 18) and `on_episode_end` after the last one.
    """
    passenger_env = _make_scenario_env()
    callbacks = _RecordingCallbacks()

    total_rewards = TravelWisePolicyRunner(passenger_env, BoardForTargetPassengerPolicy(), callbacks=callbacks).run()

    assert total_rewards == {0: 2, 1: -2, 2: 2, 3: 2, 4: 3, 5: -25}
    assert passenger_env.done
    assert callbacks.calls == ["start 0"] + [f"step {step}" for step in range(1, 19)] + ["end 18"]

