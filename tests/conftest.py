def derived_state(env, agent):
    """
    An agent's TrainState via EnvAgent.derived_state() - correct whether or not env is wrapped via
    RailEnvStateMachineWrapper (see its own docstring). A thin (env, agent) wrapper around the method
    of the same name so a test can call it without threading env._elapsed_steps through by hand.
    """
    return agent.derived_state(elapsed_steps=env._elapsed_steps)


def assert_state(env, agent, wrapped, expected):
    """
    Assert this agent's TrainState is `expected` - via derived_state() always, and - when wrapped -
    independently against the live agent.state too, so the same assertion line runs against both an
    unwrapped and a wrapped env via a single @pytest.mark.parametrize("wrapped", [True, False]) axis,
    without hardcoding either.

    Checking both sources here (rather than relying solely on
    AbstractRailEnv._check_derived_state_matches_state_postcondition, which already asserts
    derived_state() == agent.state on every step when wrapped) matters because that postcondition
    only verifies the two sources agree with *each other* - it can't tell "both correct" apart from
    "both wrong in the same way", since it never compares against an actual expected value. This does.
    """
    actual = derived_state(env, agent)
    assert actual == expected, (agent.handle, "derived_state", actual, expected)
    if wrapped:
        assert agent.state == expected, (agent.handle, "state", agent.state, expected)
