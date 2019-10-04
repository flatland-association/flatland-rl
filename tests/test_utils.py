"""Test Utils."""
from typing import List, Tuple, Optional

import numpy as np
from attr import attrs, attrib

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.rail_env import RailEnvActions, RailEnv
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


# ensure that env is working correctly with start/stop/invalidaction penalty different from 0
def set_penalties_for_replay(env: RailEnv):
    env.step_penalty = -7
    env.start_penalty = -13
    env.stop_penalty = -19
    env.invalid_action_penalty = -29


def run_replay_config(env: RailEnv, test_configs: List[ReplayConfig], rendering: bool = False):
    """
    Runs the replay configs and checks assertions.

    *Initially*
    - the position, direction, target and speed of the initial step are taken to initialize the agents

    *Before each step*
    - action must only be provided if action_required from previous step (initally all True)
    - position, direction before step are verified
    - optionally, set_malfunction is applied
    - malfunction is verified
    - status is verified (optionally)

    *After each step*
    - reward is verified after step


    Parameters
    ----------
    env
    test_configs
    rendering
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
                replay = test_config.replay[0]
                # set the initial position
                agent.position = replay.position
                agent.direction = replay.direction
                agent.target = test_config.target
                agent.speed_data['speed'] = test_config.speed

        def _assert(a, actual, expected, msg):
            assert np.allclose(actual, expected), "[{}] agent {} {}:  actual={}, expected={}".format(step, a, msg,
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
                assert info_dict['action_required'][a] == True, "[{}] agent {} expecting action_required={}".format(
                    step, a, True)
                action_dict[a] = replay.action
            else:
                assert info_dict['action_required'][a] == False, "[{}] agent {} expecting action_required={}".format(
                    step, a, False)

            if replay.set_malfunction is not None:
                agent.malfunction_data['malfunction'] = replay.set_malfunction
                agent.malfunction_data['moving_before_malfunction'] = agent.moving
            _assert(a, agent.malfunction_data['malfunction'], replay.malfunction, 'malfunction')

        _, rewards_dict, _, info_dict = env.step(action_dict)
        if rendering:
            renderer.render_env(show=True, show_observations=True)

        for a, test_config in enumerate(test_configs):
            replay = test_config.replay[step]
            _assert(a, rewards_dict[a], replay.reward, 'reward')
