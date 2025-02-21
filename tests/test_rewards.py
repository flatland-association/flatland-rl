from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
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

    distance_map.reset(agents=[agent], rail=GridTransitionMap(20, 20, transitions=RailEnvTransitions()))
    assert rewards.end_of_episode_reward(agent, distance_map, elapsed_steps=25) == -2
