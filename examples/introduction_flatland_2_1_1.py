import time

# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

# This is an introduction example for the Flatland 2.1.1 version.
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

width = 100  # With of map
height = 100  # Height of ap
nr_trains = 10  # Number of trains that have an assigned task in the env
cities_in_map = 20  # Number of cities where agents can start or end
seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 6  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )

# The schedule generator can make very basic schedules with a start point, end point and a speed profile for each agent.
# The speed profiles can be adjusted directly as well as shown later on. We start by introducing a statistical
# distribution of speed profiles

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)

# We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
# during an episode.

stochastic_data = {'prop_malfunction': 0.3,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }

# Custom observation builder without predictor
observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor, uncomment line below if you want to try this one
# observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=observation_builder,
              remove_agents_at_target=True  # Removes agents at the end of their journey to make space for others
              )

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=1000,  # Adjust these parameters to fit your resolution
                          screen_width=1000)  # Adjust these parameters to fit your resolution


# We first look at the map we have created

# nv_renderer.render_env(show=True)
#time.sleep(2)
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
        return 2  # np.random.choice(np.arange(self.action_size))

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
agents_with_same_start = []
print("\n The following agents have the same initial position:")
print("============================")
for agent_idx, agent in enumerate(env.agents):
    for agent_2_idx, agent2 in enumerate(env.agents):
        if agent_idx != agent_2_idx and agent.initial_position == agent2.initial_position:
            print("Agent {} as the same initial position as agent {}".format(agent_idx, agent_2_idx))
            agents_with_same_start.append(agent_idx)

# Lets try to enter with all of these agents at the same time
action_dict = {}

for agent_id in agents_with_same_start:
    action_dict[agent_id] = 1  # Set agents to moving

print("\n This happened when all tried to enter at the same time:")
print("========================================================")
for agent_id in agents_with_same_start:
    print("Agent {} status is: {} with its current position being {}".format(agent_id, str(env.agents[agent_id].status),
                                                                             str(env.agents[agent_id].position)))

# Do a step in the environment to see what agents entered:
env.step(action_dict)

# Current state and position of the agents after all agents with same start position tried to move
print("\n This happened when all tried to enter at the same time:")
print("========================================================")
for agent_id in agents_with_same_start:
    print(
        "Agent {} status is: {} with its current position being {} which is the same as the start position {} and orientation {}".format(
            agent_id, str(env.agents[agent_id].status),
            str(env.agents[agent_id].position), env.agents[agent_id].initial_position, env.agents[agent_id].direction))
# Empty dictionary for all agent action
action_dict = dict()

print("Start episode...")
# Reset environment and get initial observations for all agents
start_reset = time.time()
obs, info = env.reset()
end_reset = time.time()
print(end_reset - start_reset)
print(env.get_num_agents(), )
# Reset the rendering sytem
env_renderer.reset()

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository

score = 0
# Run episode
frame_step = 0
for step in range(500):
    # Chose an action for each agent in the environment
    for a in range(env.get_num_agents()):
        action = controller.act(obs[a])
        action_dict.update({a: action})

    # Environment step which returns the observations for all agents, their corresponding
    # reward and whether their are done
    next_obs, all_rewards, done, _ = env.step(action_dict)
    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    frame_step += 1
    # Update replay buffer and train agent
    for a in range(env.get_num_agents()):
        controller.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        score += all_rewards[a]

    obs = next_obs.copy()
    if done['__all__']:
        break

print('Episode: Steps {}\t Score = {}'.format(step, score))
