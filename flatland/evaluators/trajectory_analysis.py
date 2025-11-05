from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from pandas import DataFrame

from flatland.trajectories.trajectories import Trajectory


def data_frame_for_trajectories(root_data_dir: Path) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    all_actions = []
    all_trains_positions = []
    all_trains_arrived = []
    all_trains_rewards_dones_infos = []
    env_stats = []
    agent_stats = []

    data_dirs = sorted([serialised_state.parent for serialised_state in (root_data_dir.resolve().glob("**/serialised_state"))])
    print(data_dirs)

    for data_dir in data_dirs:
        snapshots = [snapshot for snapshot in (data_dir / "serialised_state").glob("*.pkl") if "step" not in snapshot.name]

        # must be data dir with single episode data
        assert len(snapshots) == 1, snapshots
        ep_id = snapshots[0].stem
        trajectory = Trajectory.load_existing(data_dir=data_dir, ep_id=ep_id)
        env = trajectory.load_env()

        all_actions.append(trajectory.actions)
        all_trains_positions.append(trajectory.trains_positions)
        all_trains_arrived.append(trajectory.trains_arrived)
        trajectory.trains_rewards_dones_infos["action_required"] = trajectory.trains_rewards_dones_infos["info"].map(lambda d: d["action_required"])
        trajectory.trains_rewards_dones_infos["malfunction"] = trajectory.trains_rewards_dones_infos["info"].map(lambda d: d["malfunction"])
        trajectory.trains_rewards_dones_infos["speed"] = trajectory.trains_rewards_dones_infos["info"].map(lambda d: d["speed"])
        trajectory.trains_rewards_dones_infos["state"] = trajectory.trains_rewards_dones_infos["info"].map(lambda d: d["state"])
        all_trains_rewards_dones_infos.append(trajectory.trains_rewards_dones_infos)

        env_stats.append(pd.DataFrame.from_records([{
            "episode_id": ep_id,
            "max_episode_steps": env._max_episode_steps,
            "num_agents": len(env.agents),
        }]))

        agent_stats.append(pd.DataFrame.from_records([{
            "episode_id": ep_id,
            "agent_id": agent.handle,
            "earliest_departure": agent.earliest_departure,
            "latest_arrival": agent.latest_arrival,
            "num_waypoints": len(agent.waypoints),
        } for agent in env.agents]))

    all_actions = pd.concat(all_actions)
    all_trains_positions = pd.concat(all_trains_positions)
    all_trains_arrived = pd.concat(all_trains_arrived)
    all_trains_rewards_dones_infos = pd.concat(all_trains_rewards_dones_infos)
    env_stats = pd.concat(env_stats)
    agent_stats = pd.concat(agent_stats)
    print(all_trains_arrived)

    return all_actions, all_trains_positions, all_trains_arrived, all_trains_rewards_dones_infos, env_stats, agent_stats


@click.command()
@click.option(
    '--root-data-dir',
    type=click.Path(exists=True, path_type=Path),
    help="Path to existing trjajectories. Defaults to current directory.",
    default=Path("."),
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, path_type=Path),
    help="Path store data frames to. Must be empty.",
    required=False,
    default=None
)
def cli(root_data_dir: Path, output_dir: Path):
    all_actions, all_trains_positions, all_trains_arrived, all_trains_rewards_dones_infos, env_stats, agent_stats = data_frame_for_trajectories(
        root_data_dir=root_data_dir)
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        assert len(list(output_dir.glob("*"))) == 0

        all_actions.to_csv(output_dir / "all_actions.csv", index=False)
        all_trains_positions.to_csv(output_dir / "all_trains_positions.csv", index=False)
        all_trains_arrived.to_csv(output_dir / "all_trains_arrived.csv", index=False)
        all_trains_rewards_dones_infos.to_csv(output_dir / "all_trains_rewards_dones_infos.csv", index=False)
        env_stats.to_csv(output_dir / "env_stats.csv", index=False)
        agent_stats.to_csv(output_dir / "agent_stats.csv", index=False)
