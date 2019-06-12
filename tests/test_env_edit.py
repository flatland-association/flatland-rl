from flatland.envs.agent_utils import EnvAgentStatic
from flatland.envs.rail_env import RailEnv


def test_load_env():
    env = RailEnv(10, 10)
    env.load_resource('env-data.tests', 'test-10x10.mpk')

    agent_static = EnvAgentStatic((0, 0), 2, (5, 5), False)
    env.add_agent_static(agent_static)
    assert env.get_num_agents() == 1
