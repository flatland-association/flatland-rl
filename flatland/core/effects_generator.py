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

    def post_reset(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
        """
        Called by env at the end of reset before computing observations and infos.

        In the future, will receive immutable state instead of full env.

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

    def pre_step(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
        """
        Called by env at the beginning of step before evaluating the agent's actions.

        In the future, will receive immutable state instead of full env.

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

    def post_step(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
        """
        Called by env at the end of step before computing observations and infos.

        In the future, will receive immutable state instead of full env.

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
        def post_reset(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
            for eff in effects_generators:
                env = eff.post_reset(env)
            return env

        def pre_step(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
            for eff in effects_generators:
                env = eff.pre_step(env)
            return env

        def post_step(self, env: EnvironmentType, *args, **kwargs) -> EnvironmentType:
            for eff in effects_generators:
                env = eff.pre_step(env)
            return env

    return _EffectsGeneratorWrapped()
