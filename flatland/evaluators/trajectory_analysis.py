from pathlib import Path

import pandas as pd

from flatland.trajectories.trajectories import Trajectory


def data_frame_for_trajectories(root_data_dir: Path):
    all_actions = []
    all_trains_positions = []
    all_trains_arrived = []
    all_trains_rewards_dones_infos = []
    env_stats = []
    agent_stats = []

    data_dirs = sorted(list(root_data_dir.glob("*")))
    for data_dir in data_dirs:
        snapshots = [snapshot for snapshot in (data_dir / "serialised_state").glob("*.pkl") if "step" not in snapshot.name]
        assert len(snapshots) == 1
        ep_id = snapshots[0].stem
        trajectory = Trajectory(data_dir=data_dir, ep_id=ep_id)
        trajectory.load()
        env = trajectory.restore_episode()

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
            "num_agents": len(env.agents)
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

    return all_actions, all_trains_positions, all_trains_arrived, all_trains_rewards_dones_infos, env_stats, agent_stats
