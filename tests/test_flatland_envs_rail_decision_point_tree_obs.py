from flatland.envs.rail_decision_point_tree_obs import DecisionPointTreeObs
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.utils.simple_rail import make_simple_rail


def test_decisions_point_tree_obs():
    rail, rail_map, optionals = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  number_of_agents=1,
                  obs_builder_object=DecisionPointTreeObs(depth=2),
                  random_seed=56)
    env.reset()
    assert env.agents[0].initial_position == (6, 6)
    # a bit hacky: initial position at dead-end needs to have correct initial direction
    env.agents[0].initial_direction = 0

    obs, _, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD})
    obs, _, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD})
