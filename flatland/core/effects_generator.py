from typing import Callable, TypeVar

from flatland.core.env import Environment

EnvironmentType = TypeVar('EnvironmentType', bound=Environment, covariant=True)


class EffectsGenerator[U]:
    """
    Hook for external events modifying the env (state) before observations and rewards are computed.

    See https://github.com/flatland-association/flatland-workshop-2024/blob/main/next-flatland/documentation/core_concept.md#effect-creator
    """

    def __init__(self, hook: Callable[[U], U] = None):
        self._hook = hook

    def __call__(self, env: U, *args, **kwargs) -> U:
        if self._hook is None:
            return env
        return self._hook(*args, **kwargs)


def effects_generator_wrapper[V](*post_hooks: EffectsGenerator[V]):
    def strict_compose(*funcs):
        *funcs, penultimate, last = funcs
        if funcs:
            penultimate = strict_compose(*funcs, penultimate)
        return lambda *args, **kwargs: penultimate(last(*args, **kwargs))

    return EffectsGenerator[V](strict_compose(*post_hooks))
