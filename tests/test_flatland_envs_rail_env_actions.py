from flatland.envs.rail_env_action import RailEnvActions


def test_process_illegal_action():
    assert RailEnvActions.from_value(None) == RailEnvActions.DO_NOTHING
    assert RailEnvActions.from_value(0) == RailEnvActions.DO_NOTHING
    assert RailEnvActions.from_value(RailEnvActions.DO_NOTHING) == RailEnvActions.DO_NOTHING
    assert RailEnvActions.from_value("Alice") == RailEnvActions.DO_NOTHING
    assert RailEnvActions.from_value("MOVE_LEFT") == RailEnvActions.MOVE_LEFT
    assert RailEnvActions.from_value("RailEnvActions.MOVE_LEFT") == RailEnvActions.MOVE_LEFT
    assert RailEnvActions.from_value(1) == RailEnvActions.MOVE_LEFT
