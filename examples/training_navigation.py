from flatland.envs.rail_env import *
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.utils.rendertools import *
from flatland.baselines.dueling_double_dqn import Agent
from collections import deque
import torch, random

random.seed(1)
np.random.seed(1)

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
transition_probability = [10.0,  # empty cell - Case 0
                          50.0,  # Case 1 - straight
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
action_prob = [0]*4
agent = Agent(state_size, action_size, "FC", 0)

for trials in range(1, n_trials + 1):

    # Reset environment
    obs = env.reset()
    for a in range(env.number_of_agents):
        if np.max(obs[a]) > 0 and np.max(obs[a]) < np.inf:
            obs[a] = np.clip(obs[a] / np.max(obs[a]), -1, 1)

    # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)

    score = 0
    env_done = 0

    # Run episode
    for step in range(100):
        #if trials > 114:
        #    env_renderer.renderEnv(show=True)

        # Action
        for a in range(env.number_of_agents):
            action = agent.act(np.array(obs[a]), eps=eps)
            action_prob[action] += 1
            action_dict.update({a: action})

        # Environment step
        next_obs, all_rewards, done, _ = env.step(action_dict)
        for a in range(env.number_of_agents):
            if np.max(next_obs[a]) > 0 and np.max(next_obs[a]) < np.inf:
                next_obs[a] = np.clip(next_obs[a] / np.max(next_obs[a]), -1, 1)
        # Update replay buffer and train agent
        for a in range(env.number_of_agents):
            agent.step(obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a])
            score += all_rewards[a]

        obs = next_obs.copy()
        if done['__all__']:
            env_done = 1
            break
    # Epsioln decay
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    done_window.append(env_done)
    scores_window.append(score)  # save most recent score
    scores.append(np.mean(scores_window))
    dones_list.append((np.mean(done_window)))

    print('\rTraining {} Agents.\tEpisode {}\tAverage Score: {:.0f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
        env.number_of_agents,
        trials,
        np.mean(
            scores_window),
        100 * np.mean(
            done_window),
        eps, action_prob/np.sum(action_prob)),
          end=" ")
    if trials % 100 == 0:
        print(
            '\rTraining {} Agents.\tEpisode {}\tAverage Score: {:.0f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.number_of_agents,
                trials,
                np.mean(
                    scores_window),
                100 * np.mean(
                    done_window),
                eps, action_prob / np.sum(action_prob)))
        torch.save(agent.qnetwork_local.state_dict(),
                   '../flatland/baselines/Nets/avoid_checkpoint' + str(trials) + '.pth')
