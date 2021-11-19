from flatland.envs.agent_utils import TrainState
from flatland.envs.malfunction_generators import ParamMalfunctionGen
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env_action import RailEnvActions
import random
# import r2_solver
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
    return

    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    # clear_output(wait=True)
    pil_image.save("images/"+str(cnt)+".png")


def ShouldRunTest(tid):
    return tid >= 7
    # return tid >= 3
    return True

def getactions(step):
    with open('/Users/dipam/Downloads/actions_ms.txt', 'r') as f:
        line = f.readline()
        while line.strip() != f'time=  {step}':
            line = f.readline()
        _ = f.readline()
        actions = {}
        line = f.readline()
        while line.strip().split(' ')[0] != 'time=':
            lsplit = line.strip().split(' ')
            act = int(lsplit[-1])
            i_agent = int(lsplit[0])
            actions[i_agent] = act
            line = f.readline()
    return actions


DEFAULT_SPEED_RATIO_MAP = {1.: 0.25,
                           1. / 2.: 0.25,
                           1. / 3.: 0.25,
                           1. / 4.: 0.25}

NUM_TESTS = 1

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

# f = open("tmp-scores.txt", "w")

total_percentage_num_done_agents = 0.0
total_score = 0.0
total_base_percentage_num_done_agents = 0.0
total_base_score = 0.0

num_tests = 0
cnt = 0
for test_id in range(NUM_TESTS):
    seed, width, height, nr_trains, nr_cities, max_rails_between_cities, max_rails_in_cities, malfunction_rate, \
    malfunction_min_duration, malfunction_max_duration = GetTestParams(test_id)
    # if not ShouldRunTest(test_id):
    #     continue

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
    # stochastic_data = MalfunctionParameters(
    #     malfunction_rate=1/10000,   # Rate of malfunction occurence
    #     min_duration=15,  # Minimal duration of malfunction
    #     max_duration=50   # Max duration of malfunction
    # )
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
    # solver = r2_solver.Solver(test_id)
    score = 0.0
    num_steps = 80 * (width + height + 20)
    print(
        "test_id=%d seed=%d nr_trains=%d nr_cities=%d num_steps=%d" % (test_id, seed, nr_trains, nr_cities, num_steps))
    for step in range(num_steps):
        # moves = solver.GetMoves(env.agents, obs[0])

        moves = getactions(step)
        if env.agents[1].speed_counter.is_cell_exit:
            moves[1] = 4
        if env._elapsed_steps > 25 and env._elapsed_steps < 41:
            a1 = env.agents[1]
            a5 = env.agents[5]
            print("Step", env._elapsed_steps, "Agent 1", a1.position, a1.state, a1.speed_counter.counter, moves[1], 
                            env.agents[1].speed_counter.is_cell_exit, env.agents[1].speed_counter.counter)
        next_obs, all_rewards, done, _ = env.step(moves)

        old_agent_positions = env.agent_positions.copy()
        # for ag in self.agents:
        #     if ag.state == TrainState.STOPPED and ag.state_machine.previous_state == TrainState.MALFUNCTION_OFF_MAP and \
        #        action_dict_.get(ag.handle, 4) == RailEnvActions.DO_NOTHING:
        #         import pdb; pdb.set_trace()

        

        positions = {}
        for ag in env.agents:
            if ag.position in positions:
                import pdb; pdb.set_trace()

            if ag.position is not None:
                positions[ag.position] = ag.handle, ag.speed_counter.speed
        
        
            
        # if env._elapsed_steps > 30:
            # import pdb; pdb.set_trace()


        
        render_env(env, True, cnt)
        cnt += 1
        for a in range(env.get_num_agents()):
            score += float(all_rewards[a])

        obs = next_obs.copy()
        if done['__all__']:
            break

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

    # print(
    #     "\n### test_id=%d nda=%d(dif=%d) pnda=%.6f(dif=%.6f) score=%.6f(dif=%.6f) avg_nda=%.6f(dif=%.6f) avg_sc=%.6f(dif=%.6f)\n" % (
    #     test_id, num_done_agents, num_done_agents - base_num_done_agents, percentage_num_done_agents,
    #     percentage_num_done_agents - base_percentage_num_done_agents, score, score - base_score, avg_nda, avg_nda_dif,
    #     total_score / num_tests, (total_score - total_base_score) / num_tests))
    # f.write("%d %d% .10f %.10f %d %.10f %.10f\n" % (
    # test_id, num_done_agents, percentage_num_done_agents, score, num_done_agents - base_num_done_agents,
    # percentage_num_done_agents - base_percentage_num_done_agents, avg_nda_dif))
    # f.flush()

# f.close()
