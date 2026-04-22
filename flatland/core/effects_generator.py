from typing import Callable, TypeVar, Generic, Dict, Any

from flatland.core.env import Environment
from flatland.utils.cli_utils import resolve_type

EnvType = TypeVar('EnvType', bound=Environment, covariant=True)

StateDict = Dict[str, Any]
Specs = Dict[str, Any]


class EffectsGenerator(Generic[EnvType]):
    """
    Hook for external events modifying the env (state) before observations and rewards are computed.

    See https://github.com/flatland-association/flatland-workshop-2024/blob/main/next-flatland/documentation/core_concept.md#effect-creator
    """

    def __init__(
        self,
        on_episode_start: Callable[[EnvType], EnvType] = None,
        on_episode_step_start: Callable[[EnvType], EnvType] = None,
        on_episode_step_end: Callable[[EnvType], EnvType] = None,
    ):
        self._on_episode_start = on_episode_start
        self._on_episode_step_start = on_episode_step_start
        self._on_episode_step_end = on_episode_step_end

    def on_episode_start(self, env: EnvType, *args, **kwargs) -> EnvType:
        """
        Called by env at the end of reset before computing observations and infos.

        In the future, will receive immutable state instead of full env.

        Naming similar to https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_start.html#ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_start, but modifying.

        Parameters
        ----------
        env
        args
        kwargs

        Returns
        -------

        """
        if self._on_episode_start is None:
            return env
        return self._on_episode_start(*args, **kwargs)

    def on_episode_step_start(self, env: EnvType, *args, **kwargs) -> EnvType:
        """
        Called by env at the beginning of step before evaluating the agent's actions.

        In the future, will receive immutable state instead of full env.

        No naming similar to RLlib equivalent, see https://docs.ray.io/en/latest/rllib/rllib-callback.html

        Parameters
        ----------
        env
        args
        kwargs

        Returns
        -------

        """
        if self._on_episode_step_start is None:
            return env
        return self._on_episode_step_start(*args, **kwargs)

    def on_episode_step_end(self, env: EnvType, *args, **kwargs) -> EnvType:
        """
        Called by env at the end of step before computing observations and infos.

        In the future, will receive immutable state instead of full env.

        Naming similar to https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_step.html#ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_step, but modifying.

        Parameters
        ----------
        env: Ra
        args
        kwargs

        Returns
        -------

        """
        if self._on_episode_step_end is None:
            return env
        return self._on_episode_step_end(*args, **kwargs)

    @property
    def fullname(self):
        klass = self.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__  # avoid outputs like 'builtins.str'
        return module + '.' + klass.__qualname__

    def __getstate__(self) -> StateDict:
        return {
            "cls": self.fullname,
            # TODO https://github.com/flatland-association/flatland-rl/issues/242 generalize serialization
        }

    def __setstate__(self, state) -> "EffectsGenerator":
        specs = state.get("specs", {})
        self.__init__(*specs.get("args", []), **specs.get("kwargs", {}))

    @classmethod
    def from_state(cls, state_dict: StateDict) -> "EffectsGenerator":
        eg = resolve_type(state_dict["cls"])()
        eg.__setstate__(state_dict)
        return eg



class MultiEffectsGeneratorWrapped(EffectsGenerator[EnvType]):
    def __init__(self, *effects_generators: EffectsGenerator[EnvType]):
        self.effects_generators = effects_generators

    def on_episode_start(self, env: EnvType, *args, **kwargs) -> EnvType:
        for eff in self.effects_generators:
            env = eff.on_episode_start(env)
        return env

    def on_episode_step_start(self, env: EnvType, *args, **kwargs) -> EnvType:
        for eff in self.effects_generators:
            env = eff.on_episode_step_start(env)
        return env

    def on_episode_step_end(self, env: EnvType, *args, **kwargs) -> EnvType:
        for eff in self.effects_generators:
            env = eff.on_episode_step_end(env)
        return env

    def __getstate__(self):
        return {
            "cls": self.fullname,
            "specs": {
                "args": [eff.__getstate__() for eff in self.effects_generators],
            }
        }

    def __setstate__(self, state_dict):
        specs = state_dict.get("specs", {})
        self.__init__(*[EffectsGenerator.from_state(state_dict_) for state_dict_ in specs.get("args", [])])


def make_multi_effects_generator(*effects_generators: EffectsGenerator[EnvType]) -> EffectsGenerator[EnvType]:
    return MultiEffectsGeneratorWrapped(*effects_generators)
