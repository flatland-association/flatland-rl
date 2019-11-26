"""Test Utils."""
from typing import List, Tuple, Optional

import numpy as np
from attr import attrs, attrib

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.malfunction_generators import MalfunctionParameters, malfunction_from_params
from flatland.envs.rail_env import RailEnvActions, RailEnv
from flatland.envs.rail_generators import RailGenerator
from flatland.envs.schedule_generators import ScheduleGenerator
from flatland.utils.rendertools import RenderTool


@attrs
class Replay(object):
    position = attrib(type=Tuple[int, int])
    direction = attrib(type=Grid4TransitionsEnum)
    action = attrib(type=RailEnvActions)
    malfunction = attrib(default=0, type=int)
    set_malfunction = attrib(default=None, type=Optional[int])
    reward = attrib(default=None, type=Optional[float])
    status = attrib(default=None, type=Optional[RailAgentStatus])


@attrs
class ReplayConfig(object):
    replay = attrib(type=List[Replay])
    target = attrib(type=Tuple[int, int])
    speed = attrib(type=float)
    initial_position = attrib(type=Tuple[int, int])
    initial_direction = attrib(type=Grid4TransitionsEnum)


# ensure that env is working correctly with start/stop/invalidaction penalty different from 0
def set_penalties_for_replay(env: RailEnv):
    env.step_penalty = -7
    env.start_penalty = -13
    env.stop_penalty = -19
    env.invalid_action_penalty = -29


def run_replay_config(env: RailEnv, test_configs: List[ReplayConfig], rendering: bool = False, activate_agents=True):
    """
    Runs the replay configs and checks assertions.

    *Initially*
    - The `initial_position`, `initial_direction`, `target` and `speed` are taken from the `ReplayConfig` to initialize the agents.

    *Before each step*
    - `position` is verfified
    - `direction` is verified
    - `status` is verified (optionally, only if not `None` in `Replay`)
    - `set_malfunction` is applied (optionally, only if not `None` in `Replay`)
    - `malfunction` is verified
    - `action` must only be provided if action_required from previous step (initally all True)

    *Step*
    - performed with the given `action`

    *After each step*
    - `reward` is verified after step


    Parameters
    ----------
    activate_agents: should the agents directly be activated when the environment is initially setup by `reset()`?
    env: the environment; is `reset()` to set the agents' intial position, direction, target and speed
    test_configs: the `ReplayConfig`s, one for each agent
    rendering: should be rendered during replay?
    """
    if rendering:
        renderer = RenderTool(env)
        renderer.render_env(show=True, frames=False, show_observations=False)
    info_dict = {
        'action_required': [True for _ in test_configs]
    }

    for step in range(len(test_configs[0].replay)):
        if step == 0:
            for a, test_config in enumerate(test_configs):
                agent: EnvAgent = env.agents[a]
                # set the initial position
                agent.initial_position = test_config.initial_position
                agent.initial_direction = test_config.initial_direction
                agent.direction = test_config.initial_direction
                agent.target = test_config.target
                agent.speed_data['speed'] = test_config.speed
            env.reset(False, False, activate_agents)

        def _assert(a, actual, expected, msg):
            print("[{}] verifying {} on agent {}: actual={}, expected={}".format(step, msg, a, actual, expected))
            assert (actual == expected) or (
                np.allclose(actual, expected)), "[{}] agent {} {}:  actual={}, expected={}".format(step, a, msg,
                                                                                                   actual,
                                                                                                   expected)

        action_dict = {}

        for a, test_config in enumerate(test_configs):
            agent: EnvAgent = env.agents[a]
            replay = test_config.replay[step]

            _assert(a, agent.position, replay.position, 'position')
            _assert(a, agent.direction, replay.direction, 'direction')
            if replay.status is not None:
                _assert(a, agent.status, replay.status, 'status')

            if replay.action is not None:
                assert info_dict['action_required'][
                           a] == True or agent.status == RailAgentStatus.READY_TO_DEPART, "[{}] agent {} expecting action_required={} or agent status READY_TO_DEPART".format(
                    step, a, True)
                action_dict[a] = replay.action
            else:
                assert info_dict['action_required'][
                           a] == False, "[{}] agent {} expecting action_required={}, but found {}".format(
                    step, a, False, info_dict['action_required'][a])

            if replay.set_malfunction is not None:
                # As we force malfunctions on the agents we have to set a positive rate that the env
                # recognizes the agent as potentially malfuncitoning
                # We also set next malfunction to infitiy to avoid interference with our tests
                agent.malfunction_data['malfunction'] = replay.set_malfunction
                agent.malfunction_data['moving_before_malfunction'] = agent.moving
                agent.malfunction_data['fixed'] = False
            _assert(a, agent.malfunction_data['malfunction'], replay.malfunction, 'malfunction')
        print(step)
        _, rewards_dict, _, info_dict = env.step(action_dict)
        if rendering:
            renderer.render_env(show=True, show_observations=True)

        for a, test_config in enumerate(test_configs):
            replay = test_config.replay[step]

            _assert(a, rewards_dict[a], replay.reward, 'reward')


def create_and_save_env(file_name: str, schedule_generator: ScheduleGenerator, rail_generator: RailGenerator):
    stochastic_data = MalfunctionParameters(malfunction_rate=1000,  # Rate of malfunction occurence
                                            min_duration=15,  # Minimal duration of malfunction
                                            max_duration=50  # Max duration of malfunction
                                            )

    env = RailEnv(width=30,
                  height=30,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  remove_agents_at_target=True)
    env.reset(True, True)
    env.save(file_name)
