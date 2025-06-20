from pathlib import Path

import click
import numpy as np
import tqdm

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.envs.rail_env import RailEnv
from flatland.trajectories.trajectories import Trajectory, SERIALISED_STATE_SUBDIR


class TrajectoryEvaluator:
    def __init__(self, trajectory: Trajectory, callbacks: FlatlandCallbacks = None):
        self.trajectory = trajectory
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        self.evaluate()

    def evaluate(self, start_step: int = None, end_step: int = None, snapshot_interval=0, tqdm_kwargs: dict = None,
                 skip_rewards_dones_infos: bool = False) -> RailEnv:
        """
         Parameters
        ----------
        start_step : int
            start evaluation from intermediate step incl. (requires snapshot to be present)
        end_step : int
            stop evaluation at intermediate step excl.
        rendering : bool
            render while evaluating
        snapshot_interval : int
            interval to write pkl snapshots to outputs/serialised_state subdirectory (not serialised_state subdirectory directly). 1 means at every step. 0 means never.
        tqdm_kwargs: dict
            additional kwargs for tqdm
        skip_rewards_dones_infos : bool
            skip verification of rewards/dones/infos
        """
        self.trajectory.load()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        env = self.trajectory.restore_episode(start_step)
        if env is None:
            raise FileNotFoundError(self.trajectory.data_dir / SERIALISED_STATE_SUBDIR / f"{self.trajectory.ep_id}.pkl")
        self.trajectory.outputs_dir.mkdir(exist_ok=True)

        if snapshot_interval > 0:
            from flatland.trajectories.trajectory_snapshot_callbacks import TrajectorySnapshotCallbacks
            if self.callbacks is None:
                self.callbacks = TrajectorySnapshotCallbacks(self.trajectory, snapshot_interval=snapshot_interval)
            else:
                self.callbacks = make_multi_callbacks(self.callbacks, TrajectorySnapshotCallbacks(self.trajectory, snapshot_interval=snapshot_interval))

        if self.callbacks is not None:
            self.callbacks.on_episode_start(env=env, data_dir=self.trajectory.outputs_dir)
        env.record_steps = True
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents
        if start_step is None:
            start_step = 0

        if end_step is None:
            end_step = env._max_episode_steps
        for elapsed_before_step in tqdm.tqdm(range(start_step, end_step), **tqdm_kwargs):
            action = {agent_id: self.trajectory.action_lookup(env_time=elapsed_before_step, agent_id=agent_id) for agent_id in range(n_agents)}
            assert env._elapsed_steps == elapsed_before_step
            _, rewards, dones, infos = env.step(action)
            if self.callbacks is not None:
                self.callbacks.on_episode_step(env=env, data_dir=self.trajectory.outputs_dir)

            elapsed_after_step = elapsed_before_step + 1

            done = dones['__all__']

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                expected_position = self.trajectory.position_lookup(env_time=elapsed_after_step, agent_id=agent_id)
                actual_position = (agent.position, agent.direction)
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
                    expected_reward, expected_done, expected_info = self.trajectory.trains_rewards_dones_infos_lookup(env_time=elapsed_after_step,
                                                                                                                      agent_id=agent_id)
                    assert actual_reward == expected_reward, (elapsed_after_step, agent_id, actual_reward, expected_reward)
                    assert actual_done == expected_done, (elapsed_after_step, agent_id, actual_done, expected_done)
                    assert actual_info == expected_info, (elapsed_after_step, agent_id, actual_info, expected_info)

            if done:
                break
        if self.callbacks is not None:
            self.callbacks.on_episode_end(env=env, data_dir=self.trajectory.outputs_dir)

        if start_step is None and end_step is None:
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
def evaluate_trajectory(data_dir: Path, ep_id: str):
    TrajectoryEvaluator(Trajectory(data_dir=data_dir, ep_id=ep_id)).evaluate()
