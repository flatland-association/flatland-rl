import tempfile
from pathlib import Path

import numpy as np

from flatland.env_generation.env_generator import env_generator
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.rewards import DefaultRewards, BasicMultiObjectiveRewards, PunctualityRewards
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy


def test_rewards_late_arrival():
    rewards = DefaultRewards()
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     arrival_time=12)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -2

    rewards = BasicMultiObjectiveRewards()
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (-2, 0, 0)


def test_rewards_early_arrival():
    rewards = DefaultRewards()
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=14,
                     arrival_time=12)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == 0

    rewards = BasicMultiObjectiveRewards()
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (0.0, 0, 0)


def test_rewards_intermediate_served_and_stopped_penalty():
    rewards = DefaultRewards()
    rewards.intermediate_not_served_penalty = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[(0, 0)], [(2, 2)], [(3, 3)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    agent.position = (2, 2)
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent, None, distance_map, 5)
    agent.state = TrainState.DONE
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == 0

    rewards = BasicMultiObjectiveRewards()
    rewards.intermediate_not_served_penalty = 33
    agent.position = (2, 2)
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent, None, distance_map, 5)
    agent.state = TrainState.DONE
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (0, 0, 0)


def test_rewards_intermediate_served_but_not_stopped_penalty():
    rewards = DefaultRewards()
    rewards.intermediate_not_served_penalty = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[(0, 0)], [(2, 2)], [(3, 3)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    agent.state = TrainState.MOVING
    rewards.step_reward(agent, None, distance_map, 5)
    agent.state = TrainState.DONE
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -33

    rewards = BasicMultiObjectiveRewards()
    rewards.intermediate_not_served_penalty = 33
    agent.state = TrainState.MOVING
    rewards.step_reward(agent, None, distance_map, 5)
    agent.state = TrainState.DONE
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (-33, 0, 0)


def test_rewards_intermediate_not_served_penalty():
    rewards = DefaultRewards()
    rewards.intermediate_not_served_penalty = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[(0, 0)], [(2, 2)], [(3, 3)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -33

    rewards = BasicMultiObjectiveRewards()
    rewards.intermediate_not_served_penalty = 33
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (-33, 0, 0)


def test_rewards_intermediate_intermediate_early_departure_penalty():
    rewards = DefaultRewards()
    rewards.intermediate_early_departure_penalty_factor = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=11,
                     waypoints=[[(0, 0)], [(2, 2)], [(3, 3)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 11],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.old_position = (2, 2)
    agent.position = (3, 3)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.state = TrainState.DONE
    assert rewards.end_of_episode_reward(agent, distance_map=distance_map, elapsed_steps=25) == -66


def test_rewards_intermediate_intermediate_late_arrival_penalty():
    rewards = DefaultRewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[(0, 0)], [(2, 2)], [(3, 3)]],
                     waypoints_earliest_departure=[3, 5, None],
                     waypoints_latest_arrival=[None, 2, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.old_position = (2, 2)
    agent.position = (3, 3)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.state = TrainState.DONE
    assert rewards.end_of_episode_reward(agent, distance_map=distance_map, elapsed_steps=25) == -99


def test_rewards_departed_but_never_arrived():
    rewards = DefaultRewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[(0, 0)], [(2, 2)], [(3, 3)]],
                     waypoints_earliest_departure=[3, 5, None],
                     waypoints_latest_arrival=[None, 2, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent=agent, agent_transition_data=AgentTransitionData(0.5, None, None, None, None, None, None, StateTransitionSignals()),
                        distance_map=distance_map,
                        elapsed_steps=5)
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -99 - 15


def test_energy_efficiency_smoothniss_in_morl():
    rewards = BasicMultiObjectiveRewards()
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=14,
                     arrival_time=12)

    agent.speed_counter.step(0)
    agent.state_machine.set_state(TrainState.WAITING)
    assert rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None) == (0, 0, 0)

    agent.speed_counter.step(1)
    agent.state_machine.set_state(TrainState.MOVING)
    assert rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None) == (0, -1, -1)

    agent.speed_counter.step(0.6)
    agent.state_machine.set_state(TrainState.MOVING)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None), (0, -0.36, -0.16))

    agent.speed_counter.step(0.6)
    agent.state_machine.set_state(TrainState.MALFUNCTION)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None), (0, 0, -0.36))

    agent.speed_counter.step(0.3)
    agent.state_machine.set_state(TrainState.MOVING)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None), (0, -0.09, -0.09))


def test_multi_objective_rewards():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory_morl = PolicyRunner.create_from_policy(
            env=env_generator(rewards=BasicMultiObjectiveRewards())[0],
            policy=RandomPolicy(), data_dir=data_dir / "morl",
            snapshot_interval=5,
        )
        assert trajectory_morl.trains_rewards_dones_infos["reward"].map(lambda r: r[0]).sum() == -1786.0
        assert trajectory_morl.trains_rewards_dones_infos["reward"].map(lambda r: r[1]).sum() == -914.0
        assert trajectory_morl.trains_rewards_dones_infos["reward"].map(lambda r: r[2]).sum() == -138.5625

        trajectory_default_rewards = PolicyRunner.create_from_policy(
            env=env_generator(rewards=DefaultRewards())[0], policy=RandomPolicy(),
            data_dir=data_dir / "default",
            snapshot_interval=5,
        )
        assert trajectory_default_rewards.trains_rewards_dones_infos["reward"].sum() == -1786.0


def test_punctuality_rewards_initial():
    rewards = PunctualityRewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(
        handle=0,
        initial_position=(0, 0),
        initial_direction=5,
        target=(3, 3),
        direction=3,
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=3,
        latest_arrival=10,
        waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
        waypoints_earliest_departure=[3, 5, None],
        waypoints_latest_arrival=[None, 2, 10],
        arrival_time=10
    )

    collect = []
    collect.append(rewards.empty())

    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)

    collect.append(rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5))
    collect.append(rewards.end_of_episode_reward(agent=agent, distance_map=distance_map, elapsed_steps=6))

    assert (0, 0) not in rewards.arrivals[0]
    assert rewards.departures[0][(0, 0)] == 5
    assert rewards.arrivals[0][(2, 2)] == 5
    assert (2, 2) not in rewards.departures[0]
    assert (3, 3) not in rewards.arrivals[0]
    assert (3, 3) not in rewards.departures[0]

    # on time only at initial
    assert rewards.cumulate(*collect) == (1, 3)


def test_punctuality_rewards_intermediate():
    rewards = PunctualityRewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(
        handle=0,
        initial_position=(0, 0),
        initial_direction=5,
        target=(3, 3),
        direction=3,
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=3,
        latest_arrival=10,
        waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
        waypoints_earliest_departure=[3, 5, None],
        waypoints_latest_arrival=[None, 2, 10],
        arrival_time=10
    )

    collect = []
    collect.append(rewards.empty())

    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    collect.append(rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=2))
    agent.old_position = (2, 2)
    agent.position = (4, 4)
    collect.append(rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5))
    collect.append(rewards.end_of_episode_reward(agent=agent, distance_map=distance_map, elapsed_steps=6))

    assert (0, 0) not in rewards.arrivals[0]
    assert rewards.departures[0][(0, 0)] == 2
    assert rewards.arrivals[0][(2, 2)] == 2
    assert rewards.departures[0][(2, 2)] == 5
    assert rewards.arrivals[0][(4, 4)] == 5
    assert (4, 4) not in rewards.departures[0]
    assert (3, 3) not in rewards.arrivals[0]
    assert (3, 3) not in rewards.departures[0]

    # on time only at intermediate
    assert rewards.cumulate(*collect) == (1, 3)


def test_punctuality_rewards_target():
    rewards = PunctualityRewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(
        handle=0, initial_position=(0, 0),
        initial_direction=5,
        target=(3, 3),
        direction=3,
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=3,
        latest_arrival=10,
        waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
        waypoints_earliest_departure=[3, 5, None],
        waypoints_latest_arrival=[None, 2, 10],
        arrival_time=10
    )

    collect = []
    collect.append(rewards.empty())

    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    collect.append(rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=2))
    agent.old_position = (2, 2)
    agent.position = (4, 4)
    collect.append(rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=4))
    agent.old_position = (4, 4)
    agent.position = (3, 3)
    collect.append(rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=10))
    collect.append(rewards.end_of_episode_reward(agent=agent, distance_map=distance_map, elapsed_steps=6))

    assert (0, 0) not in rewards.arrivals[0]
    assert rewards.departures[0][(0, 0)] == 2
    assert rewards.arrivals[0][(2, 2)] == 2
    assert rewards.departures[0][(2, 2)] == 4
    assert rewards.arrivals[0][(4, 4)] == 4
    assert rewards.departures[0][(4, 4)] == 10
    assert rewards.arrivals[0][(3, 3)] == 10
    assert (3, 3) not in rewards.departures[0]

    # on time only at target
    assert rewards.cumulate(*collect) == (1, 3)
