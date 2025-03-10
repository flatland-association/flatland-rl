from pathlib import Path

import click
import tqdm

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.trajectories.trajectories import Trajectory


class TrajectoryEvaluator:
    def __init__(self, trajectory: Trajectory, callbacks: FlatlandCallbacks = None):
        self.trajectory = trajectory
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        self.evaluate()

    def evaluate(self):
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
        data_sub_dir subdirectory within BENCHMARK_EPISODES_FOLDER
        ep_id the episode ID
        """

        trains_positions = self.trajectory.read_trains_positions()
        actions = self.trajectory.read_actions()
        trains_arrived = self.trajectory.read_trains_arrived()

        env = self.trajectory.restore_episode()
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents

        for env_time in tqdm.tqdm(range(env._max_episode_steps)):
            action = {agent_id: self.trajectory.action_lookup(actions, env_time=env_time, agent_id=agent_id) for agent_id in range(n_agents)}
            _, _, dones, _ = env.step(action)
            if self.callbacks is not None:
                self.callbacks.on_episode_step(env=env)

            done = dones['__all__']

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = (agent.position, agent.direction)
                expected_position = self.trajectory.position_lookup(trains_positions, env_time=env_time + 1, agent_id=agent_id)
                assert actual_position == expected_position, (
                    self.trajectory.data_dir, self.trajectory.ep_id, env_time + 1, agent_id, actual_position, expected_position)

            if done:
                break

        trains_arrived_episode = self.trajectory.trains_arrived_lookup(trains_arrived)
        expected_success_rate = trains_arrived_episode['success_rate']
        actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
        print(f"{actual_success_rate * 100}% trains arrived. Expected {expected_success_rate * 100}%. {env._elapsed_steps - 1} elapsed steps.")
        assert expected_success_rate == actual_success_rate


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
