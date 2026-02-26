import tempfile
from pathlib import Path

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.env_generation.env_generator import env_generator
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.rewards import DefaultRewards, BasicMultiObjectiveRewards, PunctualityRewards
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.speed_counter import _pseudo_fractional
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.trajectories.policy_runner import PolicyRunner
from tests.trajectories.test_policy_runner import RandomPolicy


def test_rewards_late_arrival():
    rewards = DefaultRewards()
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     arrival_time=12)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == -2
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == 0

    rewards = BasicMultiObjectiveRewards()
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == (-2, 0, 0)
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (0, 0, 0)


def test_rewards_early_arrival():
    rewards = DefaultRewards()
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
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
    intermediate_not_served_penalty = 33
    rewards = DefaultRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    agent.current_configuration = ((2, 2), 2)
    # intermediate stop evaluation when done
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 5) == rewards.empty()
    agent.state = TrainState.DONE
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == rewards.empty()

    rewards = BasicMultiObjectiveRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 5) == rewards.empty()
    agent.state = TrainState.DONE
    # if agent is done, intermediate not served is handled in step reward and not in end_of_episode_reward
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == rewards.empty()


def test_rewards_intermediate_served_but_not_stopped_penalty():
    intermediate_not_served_penalty = 33
    rewards = DefaultRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    agent.state = TrainState.MOVING
    rewards.step_reward(agent, None, distance_map, 5)
    agent.state = TrainState.DONE
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == -intermediate_not_served_penalty
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == 0

    rewards = BasicMultiObjectiveRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent.state = TrainState.MOVING
    rewards.step_reward(agent, None, distance_map, 5)
    agent.state = TrainState.DONE
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == (-intermediate_not_served_penalty, 0, -1)
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (0, 0, 0)


def test_rewards_intermediate_not_served_penalty():
    intermediate_not_served_penalty = 33
    rewards = DefaultRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == -intermediate_not_served_penalty
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == 0

    rewards = BasicMultiObjectiveRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == (-intermediate_not_served_penalty, 0, 0)
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == (0, 0, 0)


def test_rewards_intermediate_intermediate_early_departure_penalty():
    intermediate_early_departure_penalty_factor = 33
    rewards = DefaultRewards(intermediate_early_departure_penalty_factor=intermediate_early_departure_penalty_factor)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=11,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 11],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.old_direction = 0
    agent.position = (2, 2)
    agent.direction = 2
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5) == 0
    agent.old_position = (2, 2)
    agent.old_direction = 2
    agent.position = (3, 3)
    agent.direction = 3
    agent.state = TrainState.DONE
    assert rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5) == -66
    assert rewards.end_of_episode_reward(agent, distance_map=distance_map, elapsed_steps=25) == 0


def test_rewards_intermediate_intermediate_late_arrival_penalty():
    intermediate_late_arrival_penalty_factor = 33
    rewards = DefaultRewards(intermediate_late_arrival_penalty_factor=intermediate_late_arrival_penalty_factor)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, 5, None],
                     waypoints_latest_arrival=[None, 2, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.old_direction = 0
    agent.position = (2, 2)
    agent.direction = 2
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.old_position = (2, 2)
    agent.old_direction = 2
    agent.position = (3, 3)
    agent.direction = 3
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.state = TrainState.DONE
    assert rewards.step_reward(agent, None, distance_map=distance_map, elapsed_steps=25) == -99
    assert rewards.end_of_episode_reward(agent, distance_map=distance_map, elapsed_steps=25) == 0


def test_rewards_departed_but_never_arrived():
    intermediate_late_arrival_penalty_factor = 33
    rewards = DefaultRewards(intermediate_late_arrival_penalty_factor=intermediate_late_arrival_penalty_factor)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, 5, None],
                     waypoints_latest_arrival=[None, 2, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.old_direction = 0
    agent.position = (2, 2)
    agent.direction = 2
    agent.state = TrainState.STOPPED
    rewards.step_reward(agent=agent, agent_transition_data=AgentTransitionData(0.5, None, None, None, StateTransitionSignals()),
                        distance_map=distance_map,
                        elapsed_steps=5)
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -99 - 15


def test_energy_efficiency_smoothniss_in_morl():
    rewards = BasicMultiObjectiveRewards()
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=14,
                     arrival_time=12)

    agent.speed_counter.step(_pseudo_fractional(0))
    agent.state_machine.set_state(TrainState.WAITING)
    assert rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None) == (0, 0, 0)

    agent.speed_counter.step(_pseudo_fractional(1))
    agent.state_machine.set_state(TrainState.MOVING)
    assert rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None) == (0, -1, -1)

    agent.speed_counter.step(_pseudo_fractional(0.6))
    agent.state_machine.set_state(TrainState.MOVING)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None), (0, -0.36, -0.16))

    agent.speed_counter.step(_pseudo_fractional(0.6))
    agent.state_machine.set_state(TrainState.MALFUNCTION)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None), (0, 0, -0.36))

    agent.speed_counter.step(_pseudo_fractional(0.3))
    agent.state_machine.set_state(TrainState.MOVING)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=None), (0, -0.09, -0.09))


def test_multi_objective_rewards():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory_morl = PolicyRunner.create_from_policy(
            env=env_generator(rewards=BasicMultiObjectiveRewards(), seed=42, )[0],
            policy=RandomPolicy(), data_dir=data_dir / "morl",
            snapshot_interval=5,
        )
        assert trajectory_morl.trains_rewards_dones_infos["reward"].map(lambda r: r[0]).sum() == -1786.0
        assert trajectory_morl.trains_rewards_dones_infos["reward"].map(lambda r: r[1]).sum() == -914.0
        assert trajectory_morl.trains_rewards_dones_infos["reward"].map(lambda r: r[2]).sum() == -138.5625

        trajectory_default_rewards = PolicyRunner.create_from_policy(
            env=env_generator(rewards=DefaultRewards(), seed=42, )[0], policy=RandomPolicy(),
            data_dir=data_dir / "default",
            snapshot_interval=5,
        )
        assert trajectory_default_rewards.trains_rewards_dones_infos["reward"].sum() == -1786.0


def test_punctuality_rewards_initial():
    rewards = PunctualityRewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(
        handle=0,
        initial_configuration=((0, 0), 0),
        targets={((3, 3), d) for d in Grid4TransitionsEnum},
        current_configuration=(None, 3),
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
        initial_configuration=((0, 0), 0),
        targets={((3, 3), d) for d in Grid4TransitionsEnum},
        current_configuration=(None, 3),
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
        handle=0, initial_configuration=((0, 0), 0),
        targets={((3, 3), d) for d in Grid4TransitionsEnum},
        current_configuration=(None, 3),
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


def test_arrival_recorded_once_per_waypoint():
    """Test that arrivals are only recorded once per waypoint, not every step (Bug #327)."""
    rewards = DefaultRewards()
    agent = EnvAgent(
        initial_configuration=((5, 5), 0),
        current_configuration=(None, None),
        targets={((10, 10), d) for d in Grid4TransitionsEnum},
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=0,
        latest_arrival=100
    )
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    # Move agent to position (5, 5) facing North (direction 0)
    agent.position = (5, 5)
    agent.direction = 0
    agent.old_position = (5, 4)
    agent.old_direction = 0

    transition_data = AgentTransitionData(1.0, None, None, None, StateTransitionSignals())

    # First step at this waypoint - should record arrival
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=10)
    wp = Waypoint((5, 5), 0)
    assert rewards._proxy.arrivals[agent.handle][wp] == 10

    # Agent dwells at same position for several steps
    agent.old_position = agent.position
    agent.old_direction = agent.direction

    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=11)
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=12)
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=13)

    # Arrival time should NOT be updated - still 10
    assert rewards._proxy.arrivals[agent.handle][wp] == 10, "Arrival should only be recorded once per waypoint"


def test_departure_only_when_moving():
    """Test that departure is only recorded when agent actually moves to new waypoint (Bug #327)."""
    rewards = DefaultRewards()
    agent = EnvAgent(
        initial_configuration=((5, 5), 0),
        current_configuration=(None, None),
        targets={((10, 10), d) for d in Grid4TransitionsEnum},
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=0,
        latest_arrival=100
    )
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    transition_data = AgentTransitionData(1.0, None, None, None, StateTransitionSignals())

    # agent off map
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=1)
    off_wp = Waypoint(None, 0)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]

    # Agent at initial position
    agent.position = (5, 5)
    agent.direction = 0
    agent.old_position = None
    agent.old_direction = 0

    # First step - arrival recorded, but no departure (agent hasn't left yet)
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=2)

    wp = Waypoint((5, 5), 0)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]
    assert wp in rewards._proxy.arrivals[agent.handle], "Should record arrival when agent enters map"
    assert wp not in rewards._proxy.departures[agent.handle], "Should not record departure when agent hasn't moved"

    # Agent moves to new position
    agent.old_position = (5, 5)
    agent.old_direction = 0
    agent.position = (5, 6)
    agent.direction = 0

    # Now departure from old position should be recorded
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=3)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]
    assert wp in rewards._proxy.departures[agent.handle], "Departure should be recorded when agent moves"
    assert rewards._proxy.departures[agent.handle][wp] == 3
    wp = Waypoint((5, 6), 0)
    assert wp in rewards._proxy.arrivals[agent.handle], "Arrival should be recorded when agent moves"
    assert rewards._proxy.arrivals[agent.handle][wp] == 3

    # Agent arrives at target
    agent.old_position = (5, 6)
    agent.old_direction = 0
    agent.current_configuration = None

    # Now departure from old position should be recorded, but no arrival for None
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=4)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]
    assert wp in rewards._proxy.departures[agent.handle], "Departure should be recorded when agent moves off map"
    assert rewards._proxy.departures[agent.handle][wp] == 4


def test_waypoint_comparison_uses_waypoint_objects():
    """Test that waypoint tracking correctly uses Waypoint objects, not tuples (Bug #327)."""
    rewards = DefaultRewards()
    agent = EnvAgent(
        initial_configuration=((7, 8), 1),
        current_configuration=(None, None),
        targets={((10, 10), d) for d in Grid4TransitionsEnum},
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=0,
        latest_arrival=100
    )
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    transition_data = AgentTransitionData(1.0, None, None, None, StateTransitionSignals())

    agent.position = (7, 8)
    agent.direction = 1
    agent.old_position = (7, 7)
    agent.old_direction = 1

    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=5)

    # Check that arrivals dict uses Waypoint as key, not tuple
    wp = Waypoint((7, 8), 1)
    assert wp in rewards._proxy.arrivals[agent.handle], "Should use Waypoint object as key"
    assert (7, 8) not in rewards._proxy.arrivals[agent.handle], "Should not have tuple as key"
