from typing import Callable, TypeVar, Generic

from flatland.core.env import Environment

EnvironmentType = TypeVar('EnvironmentType', bound=Environment, covariant=True)


class EffectsGenerator(Generic[EnvironmentType]):
    """
    Hook for external events modifying the env (state) before observations and rewards are computed.

    See https://github.com/flatland-association/flatland-workshop-2024/blob/main/next-flatland/documentation/core_concept.md#effect-creator
    """

    def __init__(
            self,
            post_reset: Callable[[EnvironmentType], EnvironmentType] = None,
            pre_step: Callable[[EnvironmentType], EnvironmentType] = None,
            post_step: Callable[[EnvironmentType], EnvironmentType] = None,
    ):
        self._post_reset = post_reset
        self._pre_step = pre_step
        self._pre_step = pre_step
        self._post_step = post_step

    def on_episode_start(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
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
        if self._post_reset is None:
            return env
        return self._post_reset(*args, **kwargs)

    def on_episode_step_start(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
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
        if self._pre_step is None:
            return env
        return self._pre_step(*args, **kwargs)

    def on_episode_step_end(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
        """
        Called by env at the end of step before computing observations and infos.

        In the future, will receive immutable state instead of full env.

        Naming similar to  to https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_step.html#ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_step, but modifying.

        Parameters
        ----------
        env: Ra
        args
        kwargs

        Returns
        -------

        """
        if self._post_step is None:
            return env
        return self._post_step(*args, **kwargs)


def effects_generator_wrapper(*effects_generators: EffectsGenerator[EnvironmentType]) -> EffectsGenerator[EnvironmentType]:
    class _EffectsGeneratorWrapped(EffectsGenerator[EnvironmentType]):
        def on_episode_start(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
            for eff in effects_generators:
                env = eff.on_episode_start(env)
            return env

        def on_episode_step_start(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
            for eff in effects_generators:
                env = eff.on_episode_step_start(env)
            return env

        def on_episode_step_end(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
            for eff in effects_generators:
                env = eff.on_episode_step_start(env)
            return env

    return _EffectsGeneratorWrapped()
