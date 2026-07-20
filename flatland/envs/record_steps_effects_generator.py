from typing import Dict, List, Optional

from flatland.core.effects_generator import EffectsGenerator
from flatland.envs.rail_env_action import RailEnvActions


class RecordStepsEffectsGenerator(EffectsGenerator["RailEnv"]):
    """
    Records agent positions, orientations, malfunction status, state and deadlock status for each step
    into `env.cur_episode`, and the actions into `self.list_actions`, whenever `self.record_steps` is set.
    """

    def __init__(self, record_steps: bool = False):
        super().__init__()
        self.record_steps = record_steps  # whether to save timesteps
        self.list_actions = []  # save actions in here

    def on_episode_step_end(self, env: "RailEnv", action_dict: Optional[Dict[int, RailEnvActions]] = None, *args, **kwargs) -> "RailEnv":
        # `env` may be `None` here if an earlier effects generator composed alongside this one in a
        # `MultiEffectsGeneratorWrapped` chain does not return it (e.g. mutates in place and returns `None`).
        if env is None or not self.record_steps:
            return env

        list_agents_state = []
        for i_agent in range(env.get_num_agents()):
            agent = env.agents[i_agent]
            # the int cast is to avoid numpy types which may cause problems with msgpack
            # in env v2, agents may have position None, before starting
            if agent.position is None:
                pos = (None, None)
                dir = None
            else:
                pos = (int(agent.position[0]), int(agent.position[1]))
                dir = int(agent.direction)
            list_agents_state.append([
                *pos, dir,
                agent.malfunction_handler.malfunction_down_counter,
                agent.state.value,
                int(agent.position in env.motion_check.deadlocked),
            ])

        env.cur_episode.append(list_agents_state)
        self.list_actions.append(action_dict)
        return env

    def set_state(self, list_actions: List[Optional[Dict[int, RailEnvActions]]]):
        """
        Restore `list_actions` from persisted state, e.g. the "actions" recorded by `RailEnvPersister.save_episode`.
        Not part of the generic `EffectsGenerator.__getstate__`/`__setstate__` roundtrip, since actions are stored
        under a dedicated top-level key rather than embedded in the serialized `effects_generator` state.
        """
        self.list_actions = list(list_actions)

    def __getstate__(self):
        return {"cls": self.fullname}
