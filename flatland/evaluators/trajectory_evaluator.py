import os
from pathlib import Path

import click
import numpy as np
import tqdm

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.persistence import RailEnvPersister
from flatland.trajectories.trajectories import Trajectory, SERIALISED_STATE_SUBDIR


class TrajectoryEvaluator:
    def __init__(self, trajectory: Trajectory, callbacks: FlatlandCallbacks = None):
        self.trajectory = trajectory
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        self.evaluate()

    def evaluate(self, start_step: int = None, end_step: int = None, snapshot_interval=0):
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
        start_step : int
            start evaluation from intermediate step incl. (requires snapshot to be present)
        end_step : int
            stop evaluation at intermediate step excl.
        rendering : bool
            render while evaluating
        snapshot_interval : int
            interval to write pkl snapshots. 1 means at every step. 0 means never.
        """

        trains_positions = self.trajectory.read_trains_positions()
        actions = self.trajectory.read_actions()
        trains_arrived = self.trajectory.read_trains_arrived()

        env = self.trajectory.restore_episode(start_step)
        self.trajectory.outputs_dir.mkdir(exist_ok=True)
        if self.callbacks is not None:
            self.callbacks.on_episode_start(env=env, data_dir=self.trajectory.outputs_dir)
        env.record_steps = True
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents
        if start_step is None:
            start_step = 0

        if end_step is None:
            end_step = env._max_episode_steps
        for elapsed_before_step in tqdm.tqdm(range(start_step, end_step)):
            # TODO refactor as callback, too
            if snapshot_interval > 0 and elapsed_before_step % snapshot_interval == 0:
                RailEnvPersister.save(env, os.path.join(self.trajectory.data_dir, SERIALISED_STATE_SUBDIR,
                                                        f"{self.trajectory.ep_id}_step{elapsed_before_step:04d}.pkl"))

            action = {agent_id: self.trajectory.action_lookup(actions, env_time=elapsed_before_step, agent_id=agent_id) for agent_id in range(n_agents)}
            _, _, dones, _ = env.step(action)
            if self.callbacks is not None:
                self.callbacks.on_episode_step(env=env, data_dir=self.trajectory.outputs_dir)

            elapsed_after_step = elapsed_before_step + 1
            if snapshot_interval > 0 and (elapsed_after_step) % snapshot_interval == 0:
                RailEnvPersister.save(env,
                                      os.path.join(self.trajectory.data_dir, SERIALISED_STATE_SUBDIR,
                                                   f"{self.trajectory.ep_id}_step{(elapsed_after_step):04d}.pkl"))

            done = dones['__all__']

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                expected_position = self.trajectory.position_lookup(trains_positions, env_time=elapsed_after_step, agent_id=agent_id)
                actual_position = (agent.position, agent.direction)
                assert actual_position == expected_position, f"\n====================================================\n\n\n\n\n" \
                                                             f"- actual_position:\t{actual_position}\n" \
                                                             f"- expected_position:\t{expected_position}\n" \
                                                             f"- trajectory:\tTrajectory({self.trajectory.data_dir}, {self.trajectory.ep_id})\n" \
                                                             f"- agent:\t{agent} \n- state_machine:\t{agent.state_machine}\n" \
                                                             f"- speed_counter:\t{agent.speed_counter}\n" \
                                                             f"- breakpoint:\tself._elapsed_steps == {elapsed_after_step} and agent.handle == {agent.handle}\n" \
                                                             f"- motion check:\t{list(env.motionCheck.G.edges)}\n\n\n" \
                                                             f"- agents:\t{env.agents}"

            if done:
                break
        if self.callbacks is not None:
            self.callbacks.on_episode_end(env=env, data_dir=self.trajectory.outputs_dir)

        trains_arrived_episode = self.trajectory.trains_arrived_lookup(trains_arrived)
        expected_success_rate = trains_arrived_episode['success_rate']
        actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
        print(f"{actual_success_rate * 100}% trains arrived. Expected {expected_success_rate * 100}%. {env._elapsed_steps - 1} elapsed steps.")

        if start_step is None and end_step is None:
            assert np.isclose(expected_success_rate, actual_success_rate)


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True),
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
