"""Test Utils."""
from typing import List, Tuple, Optional

import numpy as np
from attr import attrs, attrib

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.line_generators import LineGenerator
from flatland.envs.malfunction_generators import MalfunctionParameters, malfunction_from_params
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnvActions, RailEnv
from flatland.envs.rail_generators import RailGenerator
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool


@attrs
class Replay(object):
    position = attrib(type=Tuple[int, int])
    direction = attrib(type=Grid4TransitionsEnum)
    action = attrib(type=RailEnvActions)
    malfunction = attrib(default=0, type=int)
    set_malfunction = attrib(default=None, type=Optional[int])
    reward = attrib(default=None, type=Optional[float])
    state = attrib(default=None, type=Optional[TrainState])
    speed = attrib(default=None, type=Optional[float])
    distance = attrib(default=None, type=Optional[float])


@attrs
class ReplayConfig(object):
    replay = attrib(type=List[Replay])
    target = attrib(type=Tuple[int, int])
    speed = attrib(type=float)
    initial_position = attrib(type=Tuple[int, int])
    initial_direction = attrib(type=Grid4TransitionsEnum)
    max_speed = attrib(default=None, type=Optional[float])


# ensure that env is working correctly with start/stop/invalidaction penalty different from 0
def set_penalties_for_replay(env: RailEnv):
    env.step_penalty = -7
    env.start_penalty = -13
    env.stop_penalty = -19
    env.invalid_action_penalty = -29


def run_replay_config(env: RailEnv, test_configs: List[ReplayConfig], rendering: bool = False, activate_agents=True,
                      skip_reward_check=False, set_ready_to_depart=False, skip_action_required_check=False):
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
    - `action` must only be provided if action_required from previous step (initially all True)
    - `speed` is verified after step
    - `distance` is verified after step

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

    env.record_steps = True

    for step in range(len(test_configs[0].replay)):
        if step == 0:
            for a, test_config in enumerate(test_configs):
                agent: EnvAgent = env.agents[a]
                # set the initial position
                agent.initial_position = test_config.initial_position
                agent.initial_direction = test_config.initial_direction
                agent.direction = test_config.initial_direction
                agent.target = test_config.target
                agent.speed_counter = SpeedCounter(speed=test_config.speed,
                                                   max_speed=test_config.max_speed if test_config.max_speed is not None else test_config.speed)
            env.reset(False, False)

            if set_ready_to_depart:
                # Set all agents to ready to depart
                for i_agent in range(len(env.agents)):
                    env.agents[i_agent].earliest_departure = 0
                    env.agents[i_agent]._set_state(TrainState.READY_TO_DEPART)

            elif activate_agents:
                assert len(set([a.initial_position for a in env.agents])) == len(env.agents)
                for a_idx in range(len(env.agents)):
                    env.agents[a_idx].position = env.agents[a_idx].initial_position
                    env.agents[a_idx]._set_state(TrainState.MOVING)

        def _assert(a, actual, expected, msg, close: bool = True):
            print("[{}] verifying {} on agent {}: actual={}, expected={}".format(step, msg, a, actual, expected))
            _msg = "[{}] agent {}:  actual={}, expected={}".format(step, a, msg, actual, expected)
            assert (actual == expected) or (close and np.allclose(actual, expected)), _msg

        action_dict = {}
        print(f"[{step}] BEFORE stepping: verify position/direction/state/malfunction")
        for a, test_config in enumerate(test_configs):
            agent: EnvAgent = env.agents[a]
            replay = test_config.replay[step]
            # if not agent.position == replay.position:
            # import pdb; pdb.set_trace()
            _assert(a, agent.position, replay.position, 'position')
            _assert(a, agent.direction, replay.direction, 'direction')
            if replay.state is not None:
                _assert(a, TrainState(agent.state).name, TrainState(replay.state).name, 'state', close=False)

            if replay.speed is not None:
                _assert(a, agent.speed_counter.speed, replay.speed, "speed")
            if replay.distance is not None:
                _assert(a, agent.speed_counter.distance, replay.distance, "distance")

            if replay.action is not None:
                if not skip_action_required_check:
                    print("[{}] verifying action_required on agent {}: actual={}, expected={}".format(step, a, action_dict.get(a), replay.action))
                    assert info_dict['action_required'][
                               a] == True or agent.state == TrainState.READY_TO_DEPART, "[{}] agent {} expecting action_required={} or agent status READY_TO_DEPART".format(
                        step, a, True)
                action_dict[a] = replay.action
            else:
                if not skip_action_required_check:
                    assert info_dict['action_required'][
                               a] == False, "[{}] agent {} expecting action_required={}, but found {}".format(
                        step, a, False, info_dict['action_required'][a])

            if replay.set_malfunction is not None:
                # As we force malfunctions on the agents we have to set a positive rate that the env
                # recognizes the agent as potentially malfuncitoning
                # We also set next malfunction to infitiy to avoid interference with our tests
                env.agents[a].malfunction_handler._set_malfunction_down_counter(replay.set_malfunction)
            _assert(a, agent.malfunction_handler.malfunction_down_counter, replay.malfunction, 'malfunction')
        print(f"[{step}] STEPping with actions {action_dict}")
        _, rewards_dict, _, info_dict = env.step(action_dict)
        print(f"[{step}] AFTER stepping: verify rewards")
        # import pdb; pdb.set_trace()
        if rendering:
            renderer.render_env(show=True, show_observations=True)

        for a, test_config in enumerate(test_configs):
            replay = test_config.replay[step]

            if not skip_reward_check:
                _assert(a, rewards_dict[a], replay.reward, 'reward')


def create_and_save_env(file_name: str, line_generator: LineGenerator, rail_generator: RailGenerator):
    stochastic_data = MalfunctionParameters(malfunction_rate=1000,  # Rate of malfunction occurence
                                            min_duration=15,  # Minimal duration of malfunction
                                            max_duration=50  # Max duration of malfunction
                                            )

    env = RailEnv(width=30,
                  height=30,
                  rail_generator=rail_generator,
                  line_generator=line_generator,
                  number_of_agents=10,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  remove_agents_at_target=True)
    env.reset(True, True)
    # env.save(file_name)
    RailEnvPersister.save(env, file_name)
    return env
