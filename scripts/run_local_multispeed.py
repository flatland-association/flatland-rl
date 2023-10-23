from flatland.envs.agent_utils import TrainState
from flatland.envs.malfunction_generators import ParamMalfunctionGen
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
import random
import r2_solver
import sys
import time

import matplotlib.pyplot as plt
import PIL
from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output
from IPython.display import display


def GetTestParams(tid):
    seed = tid * 19997 + 997
    random.seed(seed)
    width = 30 + random.randint(0, 11)
    height = 30 + random.randint(0, 11)
    nr_cities = 4 + random.randint(0, (width + height) // 10)
    nr_trains = min(nr_cities * 4, 10 + random.randint(0, 10))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, 5)
    malfunction_rate = 0#1/100 + random.randint(0, 5)
    malfunction_min_duration = 0#1 + random.randint(0, 5)
    malfunction_max_duration = 0#6 + random.randint(0, 10)
    return (seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate,
            malfunction_min_duration, malfunction_max_duration)

def render_env(env,wait=True,cnt=0):

    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    # clear_output(wait=True)
    pil_image.save("images/"+str(cnt)+".png")


def ShouldRunTest(tid):
    # return tid == 5
    # return tid >= 3
    return True


DEFAULT_SPEED_RATIO_MAP = {1.: 0.25,
                           1. / 2.: 0.25,
                           1. / 3.: 0.25,
                           1. / 4.: 0.25}

NUM_TESTS = 200

d_base = {}

# f = open("scores.txt", "r")
# for line in f.readlines():
#     lsplit = line.split(" ")
#     if len(lsplit) >= 4:
#         test_id = int(lsplit[0])
#         num_done_agents = int(lsplit[1])
#         percentage_num_done_agents = float(lsplit[2])
#         score = float(lsplit[3])
#         d_base[test_id] = (num_done_agents, score)
# f.close()

f = open("tmp-scores.txt", "w")

total_percentage_num_done_agents = 0.0
total_score = 0.0
total_base_percentage_num_done_agents = 0.0
total_base_score = 0.0

num_tests = 0
cnt = 0
for test_id in range(NUM_TESTS):
    print(test_id)
    seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate, \
    malfunction_min_duration, malfunction_max_duration = GetTestParams(test_id)
    if not ShouldRunTest(test_id):
        continue

    rail_generator = sparse_rail_generator(max_num_cities=nr_cities,
                                           seed=seed,
                                           grid_mode=False,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rail_pairs_in_city=max_rails_in_cities,
                                           )

    line_generator = sparse_line_generator(DEFAULT_SPEED_RATIO_MAP, seed=seed)

    stochastic_data = MalfunctionParameters(malfunction_rate = malfunction_rate,
                                            min_duration = malfunction_min_duration,
                                            max_duration = malfunction_max_duration,
                                            )

    observation_builder = GlobalObsForRailEnv()

    env = RailEnv(width=width,
                  height=height,
                  rail_generator=rail_generator,
                  line_generator=line_generator,
                  number_of_agents=nr_trains,
                  malfunction_generator=ParamMalfunctionGen(stochastic_data),
                  obs_builder_object=observation_builder,
                  remove_agents_at_target=True,
                  random_seed=seed
                  )
    obs = env.reset()
    render_env(env)
    solver = r2_solver.Solver(test_id)
    score = 0.0
    num_steps = 15 * (width + height)
    all_rewards = {}
    print(
        "test_id=%d seed=%d nr_trains=%d nr_cities=%d num_steps=%d" % (test_id, seed, nr_trains, nr_cities, num_steps))
    for step in range(num_steps):
        moves = solver.GetMoves(env.agents, obs[0], env.distance_map, env._max_episode_steps)
        next_obs, all_rewards, done, _ = env.step(moves)
        # render_env(env, True, cnt)
        cnt += 1
        # print("step",cnt)
        for a in range(env.get_num_agents()):
            score += float(all_rewards[a])
        obs = next_obs.copy()
        if done['__all__']:
            break
    # print(env._elapsed_steps)
    # for a in range(env.get_num_agents()):
    #     print(a, float(all_rewards[a]))
    print("--Reward : ", sum(list(all_rewards.values())))
    num_done_agents = 0
    for aid, agent in enumerate(env.agents):
        if agent.state == TrainState.DONE:
            num_done_agents += 1
    percentage_num_done_agents = 100.0 * num_done_agents / len(env.agents)
    total_percentage_num_done_agents += percentage_num_done_agents
    total_score += score
    num_tests += 1

    base_num_done_agents = 0
    base_score = -1e9
    if test_id in d_base:
        base_num_done_agents, base_score = d_base[test_id]
    base_percentage_num_done_agents = 100.0 * base_num_done_agents / len(env.agents)
    total_base_percentage_num_done_agents += base_percentage_num_done_agents
    total_base_score += base_score

    avg_nda = total_percentage_num_done_agents / num_tests
    avg_nda_dif = (total_percentage_num_done_agents - total_base_percentage_num_done_agents) / num_tests

    print(
        "\n### test_id=%d nda=%d(dif=%d) pnda=%.6f(dif=%.6f) score=%.6f(dif=%.6f) avg_nda=%.6f(dif=%.6f) avg_sc=%.6f(dif=%.6f)\n" % (
            test_id, num_done_agents, num_done_agents - base_num_done_agents, percentage_num_done_agents,
            percentage_num_done_agents - base_percentage_num_done_agents, score, score - base_score, avg_nda, avg_nda_dif,
            total_score / num_tests, (total_score - total_base_score) / num_tests))
    f.write("%d %d% .10f %.10f %d %.10f %.10f\n" % (
        test_id, num_done_agents, percentage_num_done_agents, score, num_done_agents - base_num_done_agents,
        percentage_num_done_agents - base_percentage_num_done_agents, avg_nda_dif))
    f.flush()

# f.close()
