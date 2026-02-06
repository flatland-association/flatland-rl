import os
from pathlib import Path

import click
import numpy as np
import tqdm

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rewards import Rewards
from flatland.trajectories.trajectories import Trajectory
from flatland.utils.cli_utils import resolve_type


class TrajectoryEvaluator:
    def __init__(self, trajectory: Trajectory, callbacks: FlatlandCallbacks = None):
        self.trajectory = trajectory
        self.callbacks = callbacks

    def evaluate(
        self,
        start_step: int = None,
        end_step: int = None,
        snapshot_interval=0, tqdm_kwargs: dict = None,
        skip_rewards_dones_infos: bool = False,
        skip_rewards: bool = False,
        rewards: Rewards = None
    ) -> RailEnv:
        """
         Parameters
        ----------
        start_step : int
            start evaluation from intermediate step incl. (requires snapshot to be present). If not provided, defaults to 0.
        end_step : int
            stop evaluation at intermediate step excl. If not provided, defaults to env's max_episode_steps.
        snapshot_interval : int
            interval to write pkl snapshots to outputs/serialised_state subdirectory (not serialised_state subdirectory directly). 1 means at every step. 0 means never.
        tqdm_kwargs: dict
            additional kwargs for tqdm
        skip_rewards_dones_infos : bool
            skip verification of rewards/dones/infos
        rewards : Rewards
            Rewards used for evaluation. If not provided, defaults to the restored env's rewards.
        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        env = self.trajectory.load_env(start_step=start_step, rewards=rewards)
        if start_step is None:
            start_step = 0
        if end_step is None:
            end_step = env._max_episode_steps
        assert end_step >= start_step

        if snapshot_interval > 0:
            from flatland.trajectories.trajectory_snapshot_callbacks import TrajectorySnapshotCallbacks
            if self.callbacks is None:
                self.callbacks = TrajectorySnapshotCallbacks(self.trajectory, snapshot_interval=snapshot_interval)
            else:
                self.callbacks = make_multi_callbacks(self.callbacks, TrajectorySnapshotCallbacks(self.trajectory, snapshot_interval=snapshot_interval))

        if self.callbacks is not None:
            self.callbacks.on_episode_start(env=env, data_dir=self.trajectory.outputs_dir)

        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents

        action_cache, position_cache, trains_rewards_dones_infos_cache = self.trajectory.build_cache()

        done = False
        for elapsed_before_step in tqdm.tqdm(range(start_step, end_step), **tqdm_kwargs):
            action = {agent_id: action_cache[elapsed_before_step].get(agent_id, RailEnvActions.MOVE_FORWARD) for agent_id in range(n_agents)}
            assert env._elapsed_steps == elapsed_before_step
            _, rewards, dones, infos = env.step(action)
            if self.callbacks is not None:
                self.callbacks.on_episode_step(env=env, data_dir=self.trajectory.outputs_dir)

            elapsed_after_step = elapsed_before_step + 1

            done = dones['__all__']

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                expected_position = position_cache[elapsed_after_step][agent_id]
                actual_position = agent.current_configuration
                assert actual_position == expected_position, f"\n====================================================\n\n\n\n\n" \
                                                             f"- actual_position:\t{actual_position}\n" \
                                                             f"- expected_position:\t{expected_position}\n" \
                                                             f"- trajectory:\tTrajectory({self.trajectory.data_dir}, {self.trajectory.ep_id})\n" \
                                                             f"- agent:\t{agent} \n- state_machine:\t{agent.state_machine}\n" \
                                                             f"- speed_counter:\t{agent.speed_counter}\n" \
                                                             f"- breakpoint:\tself._elapsed_steps == {elapsed_after_step} and agent.handle == {agent.handle}\n" \
                                                             f"- motion check:\t{list(env.motion_check.stopped)}\n\n\n" \
                                                             f"- agents:\t{env.agents}"
                if not skip_rewards_dones_infos:
                    actual_reward = rewards[agent_id]
                    actual_done = dones[agent_id]
                    actual_info = {k: v[agent_id] for k, v in infos.items()}
                    expected_reward, expected_done, expected_info = trains_rewards_dones_infos_cache[elapsed_after_step][agent_id]
                    if not skip_rewards:
                        assert np.allclose(actual_reward, expected_reward), (elapsed_after_step, agent_id, actual_reward, expected_reward)
                    assert actual_done == expected_done, (elapsed_after_step, agent_id, actual_done, expected_done)
                    assert actual_info == expected_info, (elapsed_after_step, agent_id, actual_info, expected_info)

            if done:
                break
        if self.callbacks is not None:
            self.callbacks.on_episode_end(env=env, data_dir=self.trajectory.outputs_dir)

        if start_step == 0 and done:
            trains_arrived_episode = self.trajectory.trains_arrived_lookup()
            expected_success_rate = trains_arrived_episode['success_rate']
            actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
            print(f"{actual_success_rate * 100}% trains arrived. Expected {expected_success_rate * 100}%. {env._elapsed_steps - 1} elapsed steps.")

            assert np.isclose(expected_success_rate, actual_success_rate)
        return env


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True, path_type=Path),
              help="Path to folder containing Flatland episode",
              required=True
              )
@click.option('--ep-id',
              type=str,
              help="Episode ID.",
              required=True
              )
@click.option('--callbacks',
              type=str,
              help="Defaults to `None`. Can also be provided through env var CALLBACKS (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--callbacks-pkg',
              type=str,
              help="DEPRECATED: use --callbacks instead. Defaults to `None`. Can also be provided through env var CALLBACKS_OKG (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--callbacks-cls',
              type=str,
              help="DEPRECATED: use --callbacks instead. Defaults to `None`. Can also be provided through env var CALLBACKS_CLS (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--skip-rewards-dones-infos',
              type=bool,
              default=False,
              help="Skip verification of rewards/dones/infos.",
              required=False
              )
def evaluate_trajectory(
    data_dir: Path,
    ep_id: str,
    callbacks: str = None,
    callbacks_pkg: str = None,
    callbacks_cls: str = None,
    skip_rewards_dones_infos: bool = False
):
    if callbacks is None:
        callbacks = os.environ.get("CALLBACKS", None)
    if callbacks_pkg is None:
        callbacks_pkg = os.environ.get("CALLBACKS_PKG", None)
    if callbacks_cls is None:
        callbacks_cls = os.environ.get("CALLBACKS_CLS", None)
    callbacks = resolve_type(callbacks, callbacks_pkg, callbacks_cls)
    if callbacks is not None:
        callbacks = callbacks()

    trajectory = Trajectory.load_existing(data_dir=data_dir, ep_id=ep_id)
    TrajectoryEvaluator(trajectory, callbacks=callbacks).evaluate(skip_rewards_dones_infos=skip_rewards_dones_infos)
