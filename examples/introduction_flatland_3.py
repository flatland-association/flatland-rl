import numpy as np
import os

# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters, ParamMalfunctionGen

from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
#from flatland.envs.sparse_rail_gen import SparseRailGen
from flatland.envs.line_generators import sparse_line_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

# This is an introduction example for the Flatland 2.1.* version.
# Changes and highlights of this version include
# - Stochastic events (malfunctions)
# - Different travel speeds for differet agents
# - Levels are generated using a novel generator to reflect more realistic railway networks
# - Agents start outside of the environment and enter at their own time
# - Agents leave the environment after they have reached their goal
# Use the new sparse_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment
# We start by importing the necessary rail and schedule generators
# The rail generator will generate the railway infrastructure
# The schedule generator will assign tasks to all the agent within the railway network

# The railway infrastructure can be build using any of the provided generators in env/rail_generators.py
# Here we use the sparse_rail_generator with the following parameters

width = 16 * 7  # With of map
height = 9 * 7  # Height of map
nr_trains = 50  # Number of trains that have an assigned task in the env
cities_in_map = 20  # Number of cities where agents can start or end
seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 6  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rail_pairs_in_city=max_rail_in_cities,
                                       )

#rail_generator = SparseRailGen(max_num_cities=cities_in_map,
#                                       seed=seed,
#                                       grid_mode=grid_distribution_of_cities,
#                                       max_rails_between_cities=max_rails_between_cities,
#                                       max_rails_in_city=max_rail_in_cities,
#                                       )


# The schedule generator can make very basic schedules with a start point, end point and a speed profile for each agent.
# The speed profiles can be adjusted directly as well as shown later on. We start by introducing a statistical
# distribution of speed profiles

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

line_generator = sparse_line_generator(speed_ration_map)

# We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
# during an episode.

stochastic_data = MalfunctionParameters(malfunction_rate=1/10000,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )
# Custom observation builder without predictor
observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor, uncomment line below if you want to try this one
# observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              line_generator=line_generator,
              number_of_agents=nr_trains,
              obs_builder_object=observation_builder,
              #malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
              malfunction_generator=ParamMalfunctionGen(stochastic_data),
              remove_agents_at_target=True)
env.reset()

# Initiate the renderer
env_renderer = RenderTool(env,
                          agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                          show_debug=False,
                          screen_height=600,  # Adjust these parameters to fit your resolution
                          screen_width=800)  # Adjust these parameters to fit your resolution


# The first thing we notice is that some agents don't have feasible paths to their target.
# We first look at the map we have created

# nv_renderer.render_env(show=True)
# time.sleep(2)
# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent instead
class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT,
                                 RailEnvActions.STOP_MOVING])

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


# Initialize the agent with the parameters corresponding to the environment and observation_builder
controller = RandomAgent(218, env.action_space[0])

# We start by looking at the information of each agent
# We can see the task assigned to the agent by looking at
print("\n Agents in the environment have to solve the following tasks: \n")
for agent_idx, agent in enumerate(env.agents):
    print(
        "The agent with index {} has the task to go from its initial position {}, facing in the direction {} to its target at {}.".format(
            agent_idx, agent.initial_position, agent.direction, agent.target))

# The agent will always have a status indicating if it is currently present in the environment or done or active
# For example we see that agent with index 0 is currently not active
print("\n Their current statuses are:")
print("============================")

for agent_idx, agent in enumerate(env.agents):
    print("Agent {} status is: {} with its current position being {}".format(agent_idx, str(agent.status),
                                                                             str(agent.position)))

# The agent needs to take any action [1,2,3] except do_nothing or stop to enter the level
# If the starting cell is free they will enter the level
# If multiple agents want to enter the same cell at the same time the lower index agent will enter first.

# Let's check if there are any agents with the same start location
agents_with_same_start = set()
print("\n The following agents have the same initial position:")
print("=====================================================")
for agent_idx, agent in enumerate(env.agents):
    for agent_2_idx, agent2 in enumerate(env.agents):
        if agent_idx != agent_2_idx and agent.initial_position == agent2.initial_position:
            print("Agent {} as the same initial position as agent {}".format(agent_idx, agent_2_idx))
            agents_with_same_start.add(agent_idx)

# Lets try to enter with all of these agents at the same time
action_dict = dict()

for agent_id in agents_with_same_start:
    action_dict[agent_id] = 1  # Try to move with the agents

# Do a step in the environment to see what agents entered:
env.step(action_dict)

# Current state and position of the agents after all agents with same start position tried to move
print("\n This happened when all tried to enter at the same time:")
print("========================================================")
for agent_id in agents_with_same_start:
    print(
        "Agent {} status is: {} with the current position being {}.".format(
            agent_id, str(env.agents[agent_id].status),
            str(env.agents[agent_id].position)))

# As you see only the agents with lower indexes moved. As soon as the cell is free again the agents can attempt
# to start again.

# You will also notice, that the agents move at different speeds once they are on the rail.
# The agents will always move at full speed when moving, never a speed inbetween.
# The fastest an agent can go is 1, meaning that it moves to the next cell at every time step
# All slower speeds indicate the fraction of a cell that is moved at each time step
# Lets look at the current speed data of the agents:

print("\n The speed information of the agents are:")
print("=========================================")

for agent_idx, agent in enumerate(env.agents):
    print(
        "Agent {} speed is: {:.2f} with the current fractional position being {}".format(
            agent_idx, agent.speed_data['speed'], agent.speed_data['position_fraction']))

# New the agents can also have stochastic malfunctions happening which will lead to them being unable to move
# for a certain amount of time steps. The malfunction data of the agents can easily be accessed as follows
print("\n The malfunction data of the agents are:")
print("========================================")

for agent_idx, agent in enumerate(env.agents):
    print(
        "Agent {} is OK = {}".format(
            agent_idx, agent.malfunction_data['malfunction'] < 1))

# Now that you have seen these novel concepts that were introduced you will realize that agents don't need to take
# an action at every time step as it will only change the outcome when actions are chosen at cell entry.
# Therefore the environment provides information about what agents need to provide an action in the next step.
# You can access this in the following way.

# Chose an action for each agent
for a in range(env.get_num_agents()):
    action = controller.act(0)
    action_dict.update({a: action})
# Do the environment step
observations, rewards, dones, information = env.step(action_dict)
print("\n The following agents can register an action:")
print("========================================")
for info in information['action_required']:
    print("Agent {} needs to submit an action.".format(info))

# We recommend that you monitor the malfunction data and the action required in order to optimize your training
# and controlling code.

# Let us now look at an episode playing out with random actions performed

print("\nStart episode...")

# Reset the rendering system
env_renderer.reset()

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository


score = 0
# Run episode
frame_step = 0

os.makedirs("tmp/frames", exist_ok=True)

for step in range(500):
    # Chose an action for each agent in the environment
    for a in range(env.get_num_agents()):
        action = controller.act(observations[a])
        action_dict.update({a: action})

    # Environment step which returns the observations for all agents, their corresponding
    # reward and whether their are done

    next_obs, all_rewards, done, _ = env.step(action_dict)

    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    env_renderer.gl.save_image('tmp/frames/flatland_frame_{:04d}.png'.format(step))
    frame_step += 1
    # Update replay buffer and train agent
    for a in range(env.get_num_agents()):
        controller.step((observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        score += all_rewards[a]

    observations = next_obs.copy()
    if done['__all__']:
        break
    print('Episode: Steps {}\t Score = {}'.format(step, score))
