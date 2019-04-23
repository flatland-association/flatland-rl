from flatland.envs.rail_env import *
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.utils.rendertools import *
from flatland.baselines.dueling_double_dqn import Agent
from collections import deque
import torch,random

random.seed(1)
np.random.seed(1)

"""
transition_probability = [1.0,  # empty cell - Case 0
                          3.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          3.0,  # Case 3 - diamond drossing
                          2.0,  # Case 4 - single slip
                          1.0,  # Case 5 - double slip
                          1.0,  # Case 6 - symmetrical
                          1.0]  # Case 7 - dead end
"""
# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
transition_probability = [1.0,  # empty cell - Case 0
                          1.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          0.3,  # Case 3 - diamond drossing
                          0.5,  # Case 4 - single slip
                          0.5,  # Case 5 - double slip
                          0.2,  # Case 6 - symmetrical
                          0.0]  # Case 7 - dead end

# Example generate a random rail
env = RailEnv(width=7,
              height=7,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              number_of_agents=1)
env_renderer = RenderTool(env)
handle = env.get_agent_handles()

state_size = 105
action_size = 4
n_trials = 5000
eps = 1.
eps_end = 0.005
eps_decay = 0.998
action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []

agent = Agent(state_size, action_size, "FC", 0)

for trials in range(1, n_trials + 1):

    # Reset environment
    obs = env.reset()
    # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)


    score = 0
    env_done = 0

    # Run episode
    for step in range(100):

        # Action
        for a in range(env.number_of_agents):
            action = agent.act(np.array(obs[a]), eps=eps)
            action_dict.update({a: action})

        # Environment step
        next_obs, all_rewards, done, _ = env.step(action_dict)



        # Update replay buffer and train agent
        for a in range(env.number_of_agents):
            agent.step(obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a])
            score += all_rewards[a]

        obs = next_obs.copy()

        if all(done):
            env_done = 1
            break
    # Epsioln decay
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    done_window.append(env_done)
    scores_window.append(score)  # save most recent score
    scores.append(np.mean(scores_window))
    dones_list.append((np.mean(done_window)))

    print('\rTraining {} Agents.\tEpisode {}\tAverage Score: {:.0f}\tDones: {:.2f}%\tEpsilon: {:.2f}'.format(env.number_of_agents,
                                                                                                             trials,
                                                                                                             np.mean(
                                                                                                                 scores_window),
                                                                                                             100 * np.mean(
                                                                                                                 done_window),
                                                                                                             eps),
          end=" ")
    if trials % 100 == 0:
        print(
            '\rTraining {} Agents.\tEpisode {}\tAverage Score: {:.0f}\tDones: {:.2f}%\tEpsilon: {:.2f}'.format(env.number_of_agents,
                                                                                                               trials,
                                                                                                               np.mean(
                                                                                                                   scores_window),
                                                                                                               100 * np.mean(
                                                                                                                   done_window),
                                                                                                               eps))
        torch.save(agent.qnetwork_local.state_dict(), '../flatland/baselines/Nets/avoid_checkpoint' + str(trials) + '.pth')
