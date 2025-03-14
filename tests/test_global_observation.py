import numpy as np

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.states import TrainState


def test_get_global_observation():
    number_of_agents = 20

    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    env = RailEnv(width=50, height=50, rail_generator=sparse_rail_generator(max_num_cities=6,
                                                                            max_rails_between_cities=4,
                                                                            seed=15,
                                                                            grid_mode=False
                                                                            ),
                  line_generator=sparse_line_generator(speed_ration_map), number_of_agents=number_of_agents,
                  obs_builder_object=GlobalObsForRailEnv())
    env.reset()

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    obs, all_rewards, done, _ = env.step({i: RailEnvActions.MOVE_FORWARD for i in range(number_of_agents)})
    for i in range(len(env.agents)):
        agent: EnvAgent = env.agents[i]
        print("[{}] state={}, position={}, target={}, initial_position={}".format(i, agent.state, agent.position,
                                                                                  agent.target,
                                                                                  agent.initial_position))

    for i, agent in enumerate(env.agents):
        obs_agents_state = obs[i][1]
        obs_targets = obs[i][2]

        # test first channel of obs_targets: own target
        nr_agents = np.count_nonzero(obs_targets[:, :, 0])
        assert nr_agents == 1, "agent {}: something wrong with own target, found {}".format(i, nr_agents)

        # test second channel of obs_targets: other agent's target
        for r in range(env.height):
            for c in range(env.width):
                _other_agent_target = 0
                for other_i, other_agent in enumerate(env.agents):
                    if other_agent.target == (r, c):
                        _other_agent_target = 1
                        break
                assert obs_targets[(r, c)][
                           1] == _other_agent_target, "agent {}: at {} expected to be other agent's target = {}".format(
                    i, (r, c),
                    _other_agent_target)

        # test first channel of obs_agents_state: direction at own position
        for r in range(env.height):
            for c in range(env.width):
                if (agent.state.is_on_map_state() or agent.state == TrainState.DONE) and (
                    r, c) == agent.position:
                    assert np.isclose(obs_agents_state[(r, c)][0], agent.direction), \
                        "agent {} in state {} at {} expected to contain own direction {}, found {}" \
                            .format(i, agent.state, (r, c), agent.direction, obs_agents_state[(r, c)][0])
                elif (agent.state == TrainState.READY_TO_DEPART) and (r, c) == agent.initial_position:
                    assert np.isclose(obs_agents_state[(r, c)][0], agent.direction), \
                        "agent {} in state {} at {} expected to contain own direction {}, found {}" \
                            .format(i, agent.state, (r, c), agent.direction, obs_agents_state[(r, c)][0])
                else:
                    assert np.isclose(obs_agents_state[(r, c)][0], -1), \
                        "agent {} in state {} at {} expected contain -1 found {}" \
                            .format(i, agent.state, (r, c), obs_agents_state[(r, c)][0])

        # test second channel of obs_agents_state: direction at other agents position
        for r in range(env.height):
            for c in range(env.width):
                has_agent = False
                for other_i, other_agent in enumerate(env.agents):
                    if i == other_i:
                        continue
                    if other_agent.state in [TrainState.MOVING, TrainState.MALFUNCTION, TrainState.STOPPED, TrainState.DONE] and (
                        r, c) == other_agent.position:
                        assert np.isclose(obs_agents_state[(r, c)][1], other_agent.direction), \
                            "agent {} in state {} at {} should see other agent with direction {}, found = {}" \
                                .format(i, agent.state, (r, c), other_agent.direction, obs_agents_state[(r, c)][1])
                    has_agent = True
                if not has_agent:
                    assert np.isclose(obs_agents_state[(r, c)][1], -1), \
                        "agent {} in state {} at {} should see no other agent direction (-1), found = {}" \
                            .format(i, agent.state, (r, c), obs_agents_state[(r, c)][1])

        # test third and fourth channel of obs_agents_state: malfunction and speed of own or other agent in the grid
        for r in range(env.height):
            for c in range(env.width):
                has_agent = False
                for other_i, other_agent in enumerate(env.agents):
                    if other_agent.state in [TrainState.MOVING, TrainState.MALFUNCTION, TrainState.STOPPED,
                                             TrainState.DONE] and other_agent.position == (r, c):
                        assert np.isclose(obs_agents_state[(r, c)][2], other_agent.malfunction_handler.malfunction_down_counter), \
                            "agent {} in state {} at {} should see agent malfunction {}, found = {}" \
                                .format(i, agent.state, (r, c), other_agent.malfunction_handler.malfunction_down_counter,
                                        obs_agents_state[(r, c)][2])
                        assert np.isclose(obs_agents_state[(r, c)][3], other_agent.speed_counter.speed)
                        has_agent = True
                if not has_agent:
                    assert np.isclose(obs_agents_state[(r, c)][2], -1), \
                        "agent {} in state {} at {} should see no agent malfunction (-1), found = {}" \
                            .format(i, agent.state, (r, c), obs_agents_state[(r, c)][2])
                    assert np.isclose(obs_agents_state[(r, c)][3], -1), \
                        "agent {} in state {} at {} should see no agent speed (-1), found = {}" \
                            .format(i, agent.state, (r, c), obs_agents_state[(r, c)][3])

        # test fifth channel of obs_agents_state: number of agents ready to depart in to this cell
        for r in range(env.height):
            for c in range(env.width):
                count = 0
                for other_i, other_agent in enumerate(env.agents):
                    if other_agent.state == TrainState.READY_TO_DEPART and other_agent.initial_position == (r, c):
                        count += 1
                assert np.isclose(obs_agents_state[(r, c)][4], count), \
                    "agent {} in state {} at {} should see {} agents ready to depart, found{}" \
                        .format(i, agent.state, (r, c), count, obs_agents_state[(r, c)][4])
