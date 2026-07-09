import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.rewards import DefaultPenalties, DefaultRewards, BaseDefaultRewards, BasicMultiObjectiveRewards, PunctualityRewards, ECML2026Rewards, \
    BaseECML2026Rewards
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.speed_counter import _pseudo_fractional
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.utils.simple_rail import make_simple_rail
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
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == rewards.empty()


def test_rewards_intermediate_served_and_stopped_multiple_times_no_earliest_latest_penalty():
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

    # intermediate stop evaluation while stopped
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 5) == rewards.empty()

    # second intermediate stop evaluation while stopped
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 15) == rewards.empty()

    agent.state = TrainState.DONE
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == rewards.empty()

    rewards = BasicMultiObjectiveRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 5) == rewards.empty()
    agent.state = TrainState.DONE
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == rewards.empty()


def test_rewards_intermediate_served_and_stopped_multiple_times_but_late_arrival_penalty():
    intermediate_not_served_penalty = 33
    rewards = DefaultRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((2, 2), 2)], [Waypoint((2, 2), 2)], [Waypoint((3, 3), None)]],
                     waypoints_latest_arrival=[None, 7, 14, 25],
                     waypoints_earliest_departure=[3, 7, 14, None],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))

    # intermediate stop evaluation while stopped
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 10) == rewards.empty()

    # depart
    agent.old_configuration = ((2, 2), 2)
    agent.current_configuration = ((4, 4), 4)
    assert rewards.step_reward(agent, None, distance_map, 11) == rewards.empty()

    # second intermediate stop evaluation while stopped
    agent.old_configuration = ((4, 4), 4)
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, AgentTransitionData(Fraction(0), None, None, None, None), distance_map, 15) == rewards.empty()

    # end of episode reward while done
    agent.state = TrainState.DONE
    agent.old_configuration = ((2, 2), 2)
    agent.current_configuration = ((3, 3), 3)
    # latest arrival is 7 at intermediate, but effectively at 10:
    # latest arrival is 14 at intermediate, but effectively at 15:
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == -4 * rewards.intermediate_late_arrival_penalty_factor

    rewards = BasicMultiObjectiveRewards(intermediate_not_served_penalty=intermediate_not_served_penalty)

    # intermediate stop evaluation while stopped
    agent.old_configuration = None
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, None, distance_map, 10) == rewards.empty()

    # depart
    agent.old_configuration = ((2, 2), 2)
    agent.current_configuration = ((4, 4), 4)
    assert rewards.step_reward(agent, None, distance_map, 11) == rewards.empty()

    # second intermediate stop evaluation while stopped
    agent.old_configuration = ((4, 4), 4)
    agent.current_configuration = ((2, 2), 2)
    agent.state = TrainState.STOPPED
    assert rewards.step_reward(agent, AgentTransitionData(Fraction(0), None, None, None, None), distance_map, 15) == rewards.empty()

    # end of episode reward while done
    agent.state = TrainState.DONE
    agent.old_configuration = ((2, 2), 2)
    agent.current_configuration = ((3, 3), 3)
    # latest arrival is 7 at intermediate, but effectively at 10:
    # latest arrival is 14 at intermediate, but effectively at 15:
    assert rewards.step_reward(agent, None, distance_map, elapsed_steps=25) == (-4 * rewards.intermediate_late_arrival_penalty_factor, 0, 0)


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


def test_rewards_departed_but_never_arrived_minimum_penalty():
    target_not_reached_minimum_penalty = 50  # has to be > 15 to be effective in this test
    rewards = DefaultRewards(target_not_reached_minimum_penalty=target_not_reached_minimum_penalty)
    agent = EnvAgent(initial_configuration=((0, 0), 0),
                     targets={((3, 3), d) for d in Grid4TransitionsEnum},
                     current_configuration=(None, 3),
                     state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[[Waypoint((0, 0), 0)], [Waypoint((3, 3), None)]],
                     waypoints_earliest_departure=[3, None],
                     waypoints_latest_arrival=[None, 10],
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
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -1 * target_not_reached_minimum_penalty


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
    assert rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=-1) == (0, 0, 0)

    agent.speed_counter.step(_pseudo_fractional(1))
    agent.state_machine.set_state(TrainState.MOVING)
    assert rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=-1) == (0, -1, -1)

    agent.speed_counter.step(_pseudo_fractional(0.6))
    agent.state_machine.set_state(TrainState.MOVING)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=-1), (0, -0.36, -0.16))

    agent.speed_counter.step(_pseudo_fractional(0.6))
    agent.state_machine.set_state(TrainState.MALFUNCTION)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=-1), (0, 0, -0.36))

    agent.speed_counter.step(_pseudo_fractional(0.3))
    agent.state_machine.set_state(TrainState.MOVING)
    assert np.allclose(rewards.step_reward(agent, agent_transition_data=None, distance_map=None, elapsed_steps=-1), (0, -0.09, -0.09))


@pytest.mark.parametrize(
    "rewards,expected_sums",
    [
        (DefaultRewards(), (-1786.0,)),
        (BaseDefaultRewards(), (-1786.0,)),
        (BasicMultiObjectiveRewards(), (-1786.0, -914.0, -138.5625)),
        (ECML2026Rewards(), (-11724.5,)),
        (BaseECML2026Rewards(), (-11724.5,)),
    ],
)
def test_rewards_via_policy_runner(rewards, expected_sums):
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            env=env_generator_legacy(rewards=rewards, seed=42, )[0],
            policy=RandomPolicy(), data_dir=data_dir,
            snapshot_interval=5,
        )
        reward_col = trajectory.trains_rewards_dones_infos["reward"]
        # dict-valued rewards (e.g. BaseDefaultRewards/BaseECML2026Rewards) are compared as a single total across all penalty keys.
        if len(expected_sums) == 1:
            assert reward_col.map(lambda r: sum(r.values()) if isinstance(r, dict) else r).sum() == expected_sums[0]
        else:
            for i, expected in enumerate(expected_sums):
                assert reward_col.map(lambda r: r[i]).sum() == expected


def test_default_rewards_via_policy_runner():
    """Not passing `rewards` at all (RailEnv's own default) must be equivalent to passing an explicit `DefaultRewards()`."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory_implicit = PolicyRunner.create_from_policy(
            env=env_generator_legacy(seed=42, )[0], policy=RandomPolicy(),
            data_dir=data_dir / "implicit",
            snapshot_interval=5,
        )
        trajectory_explicit = PolicyRunner.create_from_policy(
            env=env_generator_legacy(rewards=DefaultRewards(), seed=42, )[0], policy=RandomPolicy(),
            data_dir=data_dir / "explicit",
            snapshot_interval=5,
        )
        assert trajectory_implicit.trains_rewards_dones_infos["reward"].sum() == trajectory_explicit.trains_rewards_dones_infos["reward"].sum() == -1786.0


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
    assert rewards.departures[0][(0, 0)] == [5]
    assert rewards.arrivals[0][(2, 2)] == [5]
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
    assert rewards.departures[0][(0, 0)] == [2]
    assert rewards.arrivals[0][(2, 2)] == [2]
    assert rewards.departures[0][(2, 2)] == [5]
    assert rewards.arrivals[0][(4, 4)] == [5]
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
    assert rewards.departures[0][(0, 0)] == [2]
    assert rewards.arrivals[0][(2, 2)] == [2]
    assert rewards.departures[0][(2, 2)] == [4]
    assert rewards.arrivals[0][(4, 4)] == [4]
    assert rewards.departures[0][(4, 4)] == [10]
    assert rewards.arrivals[0][(3, 3)] == [10]
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
    assert rewards._proxy.arrivals[agent.handle][wp] == [10]

    # Agent dwells at same position for several steps
    agent.old_position = agent.position
    agent.old_direction = agent.direction

    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=11)
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=12)
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=13)

    # Arrival time should NOT be updated - still 10
    assert rewards._proxy.arrivals[agent.handle][wp] == [10], "Arrival should only be recorded once per waypoint"


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
    agent.current_configuration = ((5, 5), 0)
    agent.old_configuration = None

    # First step - arrival recorded, but no departure (agent hasn't left yet)
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=2)

    wp = Waypoint((5, 5), 0)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]
    assert wp in rewards._proxy.arrivals[agent.handle], "Should record arrival when agent enters map"
    assert wp not in rewards._proxy.departures[agent.handle], "Should not record departure when agent hasn't moved"

    # Agent moves to new position
    agent.old_configuration = ((5, 5), 0)
    agent.current_configuration = ((5, 6), 0)

    # Now departure from old position should be recorded
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=3)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]
    assert wp in rewards._proxy.departures[agent.handle], "Departure should be recorded when agent moves"
    assert rewards._proxy.departures[agent.handle][wp] == [3]
    wp = Waypoint((5, 6), 0)
    assert wp in rewards._proxy.arrivals[agent.handle], "Arrival should be recorded when agent moves"
    assert rewards._proxy.arrivals[agent.handle][wp] == [3]

    # Agent arrives at target
    agent.old_configuration = ((5, 6), 0)
    agent.current_configuration = None

    # Now departure from old position should be recorded, but no arrival for None
    rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=4)
    assert off_wp not in rewards._proxy.arrivals[agent.handle]
    assert off_wp not in rewards._proxy.departures[agent.handle]
    assert wp in rewards._proxy.departures[agent.handle], "Departure should be recorded when agent moves off map"
    assert rewards._proxy.departures[agent.handle][wp] == [4]


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


def test_ecml2026_rewards_normalization():
    values = [-5, 0, -6, -8, 0, -9]
    num_agents = 2
    expected_reshaped = np.array([[-5, -6, 0], [0, -8, -9]])
    assert np.array_equal(np.reshape(np.array(values), (num_agents, -1), order='F'), expected_reshaped)
    expected_sum_per_agent = np.array([-11, -17])
    assert np.array_equal(np.sum(expected_reshaped, axis=1), expected_sum_per_agent)

    max_episode_steps = 12
    assert ECML2026Rewards().normalize(*values, num_agents=num_agents, max_episode_steps=max_episode_steps) == (-11 + -12) / (
        num_agents * max_episode_steps) + 1


"""
Tests for the COLLISION penalty semantics (see docstring of `BaseDefaultRewards`):

    "Safety measures are implemented as penalties for collisions which are directly
     proportional to the train's speed at impact"

A controlled stop (STOP_MOVING issued by the policy, braking to zero) is not an
impact and must not be penalized. Only stops imposed by the env (motion check
conflict or invalid action) are collisions.
"""

COLLISION_FACTOR = 250.0


def _moving_agent():
    agent = EnvAgent(
        initial_configuration=((5, 5), 0),
        current_configuration=((5, 6), 0),
        old_configuration=((5, 6), 0),  # dwelling, no arrival/departure side effects
        targets={((10, 10), d) for d in Grid4TransitionsEnum},
        state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
        earliest_departure=0,
        latest_arrival=100,
    )
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    return agent, distance_map


def _stop_moving_agent(agent: EnvAgent, signals: StateTransitionSignals) -> None:
    """Run the real state machine step so that previous_state/state are set consistently."""
    agent.state_machine.set_transition_signals(signals)
    agent.state_machine.step()
    assert agent.state_machine.previous_state == TrainState.MOVING
    assert agent.state == TrainState.STOPPED


def test_no_collision_penalty_on_voluntary_stop():
    """STOP_MOVING issued by the policy, braking reaches zero, no env intervention -> no penalty."""
    rewards = BaseDefaultRewards(collision_factor=COLLISION_FACTOR)
    agent, distance_map = _moving_agent()

    signals = StateTransitionSignals(stop_action_given=True, new_speed_zero=True, movement_allowed=True)
    _stop_moving_agent(agent, signals)

    transition_data = AgentTransitionData(Fraction(1), None, Fraction(0), None, signals)
    d = rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=5)
    assert d[DefaultPenalties.COLLISION.value] == 0, \
        "A controlled stop must not incur the collision penalty"


def test_collision_penalty_on_env_forced_stop():
    """Motion check denies movement (conflict with another train) -> penalty proportional to speed."""
    rewards = BaseDefaultRewards(collision_factor=COLLISION_FACTOR)
    agent, distance_map = _moving_agent()

    signals = StateTransitionSignals(movement_action_given=True, movement_allowed=False)
    _stop_moving_agent(agent, signals)

    speed_at_impact = Fraction(1)
    transition_data = AgentTransitionData(speed_at_impact, None, speed_at_impact, None, signals)
    d = rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=5)
    assert d[DefaultPenalties.COLLISION.value] == -1 * speed_at_impact * COLLISION_FACTOR


def test_collision_penalty_when_braking_interrupted_by_conflict():
    """Policy brakes (fractional braking_delta) but is force-stopped before reaching zero speed
    -> still an impact, penalized with the residual speed."""
    rewards = BaseDefaultRewards(collision_factor=COLLISION_FACTOR)
    agent, distance_map = _moving_agent()

    signals = StateTransitionSignals(stop_action_given=True, new_speed_zero=False, movement_allowed=False)
    _stop_moving_agent(agent, signals)

    residual_speed = Fraction(1, 4)
    transition_data = AgentTransitionData(residual_speed, None, residual_speed, None, signals)
    d = rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=5)
    assert d[DefaultPenalties.COLLISION.value] == -1 * residual_speed * COLLISION_FACTOR


def test_collision_penalty_on_invalid_action_stop():
    """Invalid action (e.g. DO_NOTHING on symmetric switch) -> env intervenes -> penalized.
    See https://github.com/flatland-association/flatland-rl/issues/280 for the open design question."""
    rewards = BaseDefaultRewards(collision_factor=COLLISION_FACTOR)
    agent, distance_map = _moving_agent()

    signals = StateTransitionSignals(stop_action_given=False, new_speed_zero=True, movement_allowed=False)
    _stop_moving_agent(agent, signals)

    transition_data = AgentTransitionData(Fraction(1), None, Fraction(0), None, signals)
    d = rewards.step_reward(agent, transition_data, distance_map, elapsed_steps=5)
    assert d[DefaultPenalties.COLLISION.value] == -1 * Fraction(1) * COLLISION_FACTOR


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end regression tests: verify that rail_env.step() produces the signal combinations asserted above.
# Deterministic scenario on make_simple_rail's horizontal straight (row 3):
#
#    (3,0) <---- agent 1 (westbound, from (3,5))
#          ----> agent 0 (eastbound, from (3,1)) ----> (3,9)
# ---------------------------------------------------------------------------------------------------------------------

def _make_simple_env(n_agents: int) -> RailEnv:
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=n_agents,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset(random_seed=42)
    placements = [(((3, 1), Grid4TransitionsEnum.EAST), (3, 9)), (((3, 5), Grid4TransitionsEnum.WEST), (3, 0))]
    for agent, (initial_configuration, target) in zip(env.agents, placements):
        agent.initial_configuration = initial_configuration
        agent.current_configuration = None
        agent.earliest_departure = 0
        agent.latest_arrival = 50
        agent.target = target
        agent.targets = {(target, d) for d in Grid4TransitionsEnum}
    return env


def test_env_no_collision_penalty_on_voluntary_stop():
    """Single agent (no conflict possible) stops via STOP_MOVING at full speed -> no penalty."""
    env = _make_simple_env(n_agents=1)
    agent = env.agents[0]

    for _ in range(3):
        env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == 1

    _, rewards, _, _ = env.step({0: RailEnvActions.STOP_MOVING})
    assert agent.state_machine.previous_state == TrainState.MOVING
    assert agent.state == TrainState.STOPPED
    assert rewards[0][DefaultPenalties.COLLISION.value] == 0, \
        "A controlled stop must not incur the collision penalty"

    # resuming from the controlled stop works and is not penalized either
    _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert rewards[0][DefaultPenalties.COLLISION.value] == 0


def test_env_collision_penalty_on_head_on_conflict():
    """Two agents drive head-on; the motion check force-stops the losing agent -> penalty at full speed."""
    env = _make_simple_env(n_agents=2)
    agent_0, agent_1 = env.agents

    forward = {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD}
    for _ in range(3):
        _, rewards, _, _ = env.step(forward)
        assert rewards[0][DefaultPenalties.COLLISION.value] == 0
        assert rewards[1][DefaultPenalties.COLLISION.value] == 0

    # 4th step: agent 0 at (3,2) east, agent 1 at (3,4) west, one free cell (3,3) between them;
    # both request entry -> motion check awards the cell to agent 0 and force-stops agent 1
    _, rewards, _, _ = env.step(forward)
    assert agent_1.state_machine.previous_state == TrainState.MOVING
    assert agent_1.state == TrainState.STOPPED
    assert rewards[1][DefaultPenalties.COLLISION.value] == -1 * 1 * COLLISION_FACTOR, \
        "An env-forced stop (head-on conflict) must incur the collision penalty proportional to speed"
    # the agent winning the conflict resolution keeps moving unpenalized
    assert agent_0.state == TrainState.MOVING
    assert rewards[0][DefaultPenalties.COLLISION.value] == 0

    # 5th step: agents now face each other on adjacent cells (3,3)/(3,4); agent 0's move would
    # make it collide with agent 1, which the motion check forbids -> agent 0 force-stopped
    _, rewards, _, _ = env.step(forward)
    assert agent_0.state_machine.previous_state == TrainState.MOVING
    assert agent_0.state == TrainState.STOPPED
    assert rewards[0][DefaultPenalties.COLLISION.value] == -1 * 1 * COLLISION_FACTOR, \
        "An env-forced stop (swap prevention) must incur the collision penalty proportional to speed"
    assert rewards[1][DefaultPenalties.COLLISION.value] == 0, \
        "The penalty fires once on the MOVING -> STOPPED transition, not per deadlocked step"

    # deadlock persists: no positions change, no further collision penalties accrue
    _, rewards, _, _ = env.step(forward)
    assert (agent_0.position, agent_1.position) == ((3, 3), (3, 4))
    assert rewards[0][DefaultPenalties.COLLISION.value] == 0
    assert rewards[1][DefaultPenalties.COLLISION.value] == 0
