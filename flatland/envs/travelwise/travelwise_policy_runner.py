from typing import Dict, Optional, Tuple

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.core.policy import Policy
from flatland.envs.travelwise.passenger_env import PassengerEnv


class TravelWisePolicyRunner:
    """
    The controller of a `PassengerEnv` (the TravelWise data model's `TravelWisePolicyRunner`): holds the passenger
    policy and steps the passenger env with the policy's actions, each time on the passengers' observations and infos
    of the previous step. The infos carry the static data (`infos["static"]`), so the policy gets them in
    `act_many()`.

    Steps the passenger env only: the passenger env steps the vehicles itself, with its vehicle policy. Records no
    trajectory - unlike `PolicyRunner`, which is bound to `RailEnv` and its trajectories.

        runner = TravelWisePolicyRunner(passenger_env, passenger_policy)
        total_rewards = runner.run()

    Or step by step, e.g. to inspect the passenger env after each step:

        runner.reset()
        while not passenger_env.done:
            observations, rewards, dones, infos = runner.step()
    """

    def __init__(self, passenger_env: PassengerEnv, passenger_policy: Policy, callbacks: Optional[FlatlandCallbacks] = None):
        self.passenger_env = passenger_env
        self.passenger_policy = passenger_policy
        self.callbacks = callbacks
        # the passengers' observations and infos of the last step, which the passenger policy acts on in the next step
        self.observations: Dict = {}
        self.infos: Dict = {}
        # per passenger handle, the sum of its rewards since `reset()`
        self.total_rewards: Dict[int, int] = {}

    def reset(self) -> Tuple[Dict, Dict]:
        """ Resets the passenger env. Returns the passengers' observations and the infos, and keeps them for the next step. """
        self.observations, self.infos = self.passenger_env.reset()
        self.total_rewards = {handle: 0 for handle in self.passenger_env.get_agent_handles()}
        return self.observations, self.infos

    def step(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Steps the passenger env with the passenger policy's actions on the observations and infos kept from the
        previous step, and keeps the new ones. Returns the passenger env's observations, rewards, dones and infos.
        """
        handles = self.passenger_env.get_agent_handles()
        actions = self.passenger_policy.act_many(handles, observations=[self.observations[handle] for handle in handles], infos=self.infos)
        self.observations, rewards, dones, self.infos = self.passenger_env.step(actions)
        for handle, reward in rewards.items():
            self.total_rewards[handle] += reward
        return self.observations, rewards, dones, self.infos

    def run(self) -> Dict[int, int]:
        """
        Runs an episode: `reset()`, then `step()` until the passenger env is done, calling the callbacks' `on_episode_start`
        after the reset, `on_episode_step` after each step and `on_episode_end` after the last one, as `PolicyRunner`
        does. Returns the sum of the rewards per passenger.
        """
        self.reset()
        if self.callbacks is not None:
            self.callbacks.on_episode_start(env=self.passenger_env)
        while not self.passenger_env.done:
            self.step()
            if self.callbacks is not None:
                self.callbacks.on_episode_step(env=self.passenger_env)
        if self.callbacks is not None:
            self.callbacks.on_episode_end(env=self.passenger_env)
        return self.total_rewards
