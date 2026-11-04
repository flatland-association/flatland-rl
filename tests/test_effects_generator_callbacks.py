from flatland.core.effects_generator import EffectsGenerator, make_multi_effects_generator


class _Env:
    pass


def test_effects_generator_callbacks_receive_env():
    """
    Callbacks passed to the `EffectsGenerator` constructor are called with the env, like the overridden methods, and
    their return value is the env passed on.

    - Setup: an effects generator with a callback per hook, each recording the env it gets and returning it.
    - Each hook is called with the env and further arguments: the callback gets the env first, then the arguments.
    - Without callbacks, each hook returns the env unchanged.
    """
    env = _Env()
    calls = []

    def callback(name):
        def _callback(env, *args, **kwargs):
            calls.append((name, env, args, kwargs))
            return env

        return _callback

    eg = EffectsGenerator(on_episode_start=callback("start"), on_episode_step_start=callback("step_start"),
                          on_episode_step_end=callback("step_end"))
    assert eg.on_episode_start(env) is env
    assert eg.on_episode_step_start(env, 1, action_dict={0: 2}) is env
    assert eg.on_episode_step_end(env, action_dict={}) is env
    assert calls == [("start", env, (), {}), ("step_start", env, (1,), {"action_dict": {0: 2}}), ("step_end", env, (), {"action_dict": {}})]

    assert EffectsGenerator().on_episode_step_end(env) is env


def test_effects_generator_callbacks_in_multi_effects_generator():
    """
    Callbacks passed to the constructor also get the env inside `make_multi_effects_generator()`, in order.
    """
    env = _Env()
    order = []
    eg = make_multi_effects_generator(EffectsGenerator(on_episode_step_end=lambda e, **kwargs: order.append(1) or e),
                                      EffectsGenerator(on_episode_step_end=lambda e, **kwargs: order.append(2) or e))
    assert eg.on_episode_step_end(env, action_dict={}) is env
    assert order == [1, 2]
