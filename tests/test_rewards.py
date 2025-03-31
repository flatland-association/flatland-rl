from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rewards import Rewards
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState


def test_rewards_late_arrival():
    rewards = Rewards()
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


def test_rewards_early_arrival():
    rewards = Rewards()
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


def test_rewards_intermediate_not_served_penalty():
    rewards = Rewards()
    rewards.intermediate_not_served_penalty = -33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[(0, 0), (2, 2), (3, 3)],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -33


def test_rewards_intermediate_intermediate_early_departure_penalty():
    rewards = Rewards()
    rewards.intermediate_early_departure_penalty_factor = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=11,
                     waypoints=[(0, 0), (2, 2), (3, 3)],
                     waypoints_earliest_departure=[3, 7, None],
                     waypoints_latest_arrival=[None, 11, 11],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.old_position = (2, 2)
    agent.position = (3, 3)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    assert rewards.end_of_episode_reward(agent, distance_map=distance_map, elapsed_steps=25) == -66


def test_rewards_intermediate_intermediate_late_arrival_penalty():
    rewards = Rewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.DONE),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[(0, 0), (2, 2), (3, 3)],
                     waypoints_earliest_departure=[3, 5, None],
                     waypoints_latest_arrival=[None, 2, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    agent.old_position = (2, 2)
    agent.position = (3, 3)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    assert rewards.end_of_episode_reward(agent, distance_map=distance_map, elapsed_steps=25) == -99


def test_rewards_departed_but_never_arrived():
    rewards = Rewards()
    rewards.intermediate_late_arrival_penalty_factor = 33
    agent = EnvAgent(initial_position=(0, 0),
                     initial_direction=5,
                     target=(3, 3),
                     direction=3,
                     state_machine=TrainStateMachine(initial_state=TrainState.MOVING),
                     earliest_departure=3,
                     latest_arrival=10,
                     waypoints=[(0, 0), (2, 2), (3, 3)],
                     waypoints_earliest_departure=[3, 5, None],
                     waypoints_latest_arrival=[None, 2, 10],
                     arrival_time=10)
    distance_map = DistanceMap(agents=[agent], env_height=20, env_width=20)
    distance_map.reset(agents=[agent], rail=RailGridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    agent.old_position = (0, 0)
    agent.position = (2, 2)
    rewards.step_reward(agent=agent, agent_transition_data=None, distance_map=distance_map, elapsed_steps=5)
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -99 - 15
