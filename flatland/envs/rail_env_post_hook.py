from typing import Callable


class PostHook:
    """
    PostHook for external events modifying the env (state) before observations and rewards are computed.
    """

    def __init__(self, hook: Callable[["RailEnv"], "RailEnv"] = None):
        self._hook = hook

    def __call__(self, rail_env: "RailEnv", *args, **kwargs) -> None:
        if self._hook is None:
            return rail_env
        return self._hook(*args, **kwargs)


def post_hook_wrapper(*post_hooks: PostHook):
    def strict_compose(*funcs):
        *funcs, penultimate, last = funcs
        if funcs:
            penultimate = strict_compose(*funcs, penultimate)
        return lambda *args, **kwargs: penultimate(last(*args, **kwargs))

    return PostHook(strict_compose(*post_hooks))

# TODO move to core
# TODO test imple
# TODO test persistence
