Flatland 2.0 Introduction
=========================

## What's new?

In this version of **Flat**land, we are moving closer to realistic and more complex railway problems. 
Earlier versions of **Flat**land introduced you to the concept of restricted transitions, but they were still too simplistic to give us feasible solutions for daily operations. 
Thus the following changes are coming in the next version to be closer to real railway network challenges:

- **New Level Generator** provide less connections between different nodes in the network and thus agent densities on rails are much higher.
- **Stochastic Events** cause agents to stop and get stuck for different numbers of time steps.
- **Different Speed Classes** allow agents to move at different speeds and thus enhance complexity in the search for optimal solutions.

Below we explain these changes in more detail and how you can play with their parametrization. We appreciate *your feedback* on the performance and the difficulty on these levels to help us shape the best possible **Flat**land 2.0 environment.

## Generate levels

We are currently working on different new level generators and you can expect that the levels in the submission testing will not all come from just one but rather different level generators to be sure that the controllers can handle any railway specific challenge.

Let's have a look at the `sparse_rail_generator`.

### Sparse Rail Generator
![Example_Sparse](https://i.imgur.com/DP8sIyx.png)

The idea behind the sparse rail generator is to mimic classic railway structures where dense nodes (cities) are sparsely connected to each other and where you have to manage traffic flow between the nodes efficiently. 
The cities in this level generator are much simplified in comparison to real city networks but it mimics parts of the problems faced in daily operations of any railway company.

There are a few parameters you can tune to build your own map and test different complexity levels of the levels. 
**Warning** some combinations of parameters do not go well together and will lead to infeasible level generation. 
In the worst case, the level generator currently issues a warning when it cannot build the environment according to the parameters provided. 
This will lead to a crash of the whole env. 
We are currently working on improvements here and are **happy for any suggestions from your side**.

To build an environment you instantiate a `RailEnv` as follows:

```
# Initialize the generator
rail_generator=sparse_rail_generator(
    num_cities=10,  # Number of cities in map
    num_intersections=10,  # Number of interesections in map
    num_trainstations=50,  # Number of possible start/targets on map
    min_node_dist=6,  # Minimal distance of nodes
    node_radius=3,  # Proximity of stations to city center
    num_neighb=3,  # Number of connections to other cities
    seed=5,  # Random seed
    grid_mode=False  # Ordered distribution of nodes
)

# Build the environment
env = RailEnv(
    width=50,
    height=50,
    rail_generator=rail_generator
    schedule_generator=sparse_schedule_generator(),
    number_of_agents=10,
    obs_builder_object=TreeObsForRailEnv(max_depth=3,predictor=shortest_path_predictor)
)
```

You can see that you now need both a `rail_generator` and a `schedule_generator` to generate a level. These need to work nicely together. The `rail_generator` will only generate the railway infrastructure and provide hints to the `schedule_generator` about where to place agents. The `schedule_generator` will then generate a schedule, meaning it places agents at different train stations and gives them tasks by providing individual targets.

You can tune the following parameters in the `sparse_rail_generator`:

- `num_cities` is the number of cities on a map. Cities are the only nodes that can host start and end points for agent tasks (Train stations). Here you have to be carefull that the number is not too high as all the cities have to fit on the map. When `grid_mode=False` you have to be carefull when chosing `min_node_dist` because leves will fails if not all cities (and intersections) can be placed with at least `min_node_dist` between them.
- `num_intersections` is the number of nodes that don't hold any trainstations. They are also the first priority that a city connects to. We use these to allow for sparse connections between cities.
- `num_trainstations` defines the *Total* number of trainstations in the network. This also sets the max number of allowed agents in the environment. This is also a delicate parameter as there is only a limitid amount of space available around nodes and thus if the number is too high the level generation will fail. *Important*: Only the number of agents provided to the environment will actually produce active train stations. The others will just be present as dead-ends (See figures below).
- `min_node_dist` is only used if `grid_mode=False` and represents the minimal distance between two nodes.
- `node_radius` defines the extent of a city. Each trainstation is placed at a distance to the closes city node that is smaller or equal to this number.
- `num_neighb`defines the number of neighbouring nodes that connect to each other. Thus this changes the connectivity and thus the amount of alternative routes in the network.
- `grid_mode` True -> Nodes evenly distriubted in env, False-> Random distribution of nodes
- `enhance_intersection`: True -> Extra rail elements added at intersections
- `seed` is used to initialize the random generator


If you run into any bugs with sets of parameters please let us know.

Here is a network with `grid_mode=False` and the parameters from above.

![sparse_random](https://i.imgur.com/Xg7nifF.png)

and here with `grid_mode=True`

![sparse_ordered](https://i.imgur.com/jyA7Pt4.png)

## Add Stochasticity

Another area where we improved **Flat**land 2.0 are stochastic events added during the episodes. 
This is very common for railway networks where the initial plan usually needs to be rescheduled during operations as minor events such as delayed departure from trainstations, malfunctions on trains or infrastructure or just the weather lead to delayed trains.

We implemted a poisson process to simulate delays by stopping agents at random times for random durations. The parameters necessary for the stochastic events can be provided when creating the environment.

```
# Use a the malfunction generator to break agents from time to time

stochastic_data = {
    'prop_malfunction': 0.5,  # Percentage of defective agents
    'malfunction_rate': 30,  # Rate of malfunction occurence
    'min_duration': 3,  # Minimal duration of malfunction
    'max_duration': 10  # Max duration of malfunction
}
```

The parameters are as follows:

- `prop_malfunction` is the proportion of agents that can malfunction. `1.0` means that each agent can break.
- `malfunction_rate` is the mean rate of the poisson process in number of environment steps.
- `min_duration` and `max_duration` set the range of malfunction durations. They are sampled uniformly

You can introduce stochasticity by simply creating the env as follows:

```
env = RailEnv(
    ...
    stochastic_data=stochastic_data,  # Malfunction data generator
    ...    
)
```
In your controller, you can check whether an agent is malfunctioning: 
```
obs, rew, done, info = env.step(actions) 
...
action_dict = dict()
for a in range(env.get_num_agents()):
    if info['malfunction'][a] == 0:
        action_dict.update({a: ...})

# Custom observation builder
tree_observation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

env = RailEnv(width=50,
              height=50,
              rail_generator=sparse_rail_generator(num_cities=20,  # Number of cities in map (where train stations are)
                                                   num_intersections=5,  # Number of intersections (no start / target)
                                                   num_trainstations=15,  # Number of possible start/targets on map
                                                   min_node_dist=3,  # Minimal distance of nodes
                                                   node_radius=2,  # Proximity of stations to city center
                                                   num_neighb=4,  # Number of connections to other cities/intersections
                                                   seed=15,  # Random seed
                                                   grid_mode=True,
                                                   enhance_intersection=True
                                                   ),
              schedule_generator=sparse_schedule_generator(speed_ration_map),
              number_of_agents=10,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=tree_observation)
```

You will quickly realize that this will lead to unforeseen difficulties which means that **your controller** needs to observe the environment at all times to be able to react to the stochastic events.

## Add different speed profiles

One of the main contributions to the complexity of railway network operations stems from the fact that all trains travel at different speeds while sharing a very limited railway network. 
In **Flat**land 2.0 this feature will be enabled as well and will lead to much more complex configurations. Here we count on your support if you find bugs or improvements  :).

The different speed profiles can be generated using the `schedule_generator`, where you can actually chose as many different speeds as you like. 
Keep in mind that the *fastest speed* is 1 and all slower speeds must be between 1 and 0. 
For the submission scoring you can assume that there will be no more than 5 speed profiles.


 
Later versions of **Flat**land might have varying speeds during episodes. Therefore, we return the agent speeds. 
Notice that we do not guarantee that the speed will be computed at each step, but if not costly we will return it at each step.
In your controller, you can get the agents' speed from the `info` returned by `step`: 
```
obs, rew, done, info = env.step(actions) 
...
for a in range(env.get_num_agents()):
    speed = info['speed'][a]
```

## Actions and observation with different speed levels

Because the different speeds are implemented as fractions the agents ability to perform actions has been updated. 
We **do not allow actions to change within the cell **. 
This means that each agent can only chose an action to be taken when entering a cell. 
This action is then executed when a step to the next cell is valid. For example

- Agent enters switch and choses to deviate left. Agent fractional speed is 1/4 and thus the agent will take 4 time steps to complete its journey through the cell. On the 4th time step the agent will leave the cell deviating left as chosen at the entry of the cell.
    - All actions chosen by the agent during its travels within a cell are ignored
    - Agents can make observations at any time step. Make sure to discard observations without any information. See this [example](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/training_navigation.py) for a simple implementation.
- The environment checks if agent is allowed to move to next cell only at the time of the switch to the next cell

In your controller, you can check whether an agent requires an action by checking `info`: 
```
obs, rew, done, info = env.step(actions) 
...
action_dict = dict()
for a in range(env.get_num_agents()):
    if info['action_required'][a] and info['malfunction'][a] == 0:
        action_dict.update({a: ...})

```
Notice that `info['action_required'][a]` does not mean that the action will have an effect: 
if the next cell is blocked or the agent breaks down, the action cannot be performed and an action will be required again in the next step. 

## Rail Generators and Schedule Generators
The separation between rail generator and schedule generator reflects the organisational separation in the railway domain
- Infrastructure Manager (IM): is responsible for the layout and maintenance of tracks
- Railway Undertaking (RU): operates trains on the infrastructure
Usually, there is a third organisation, which ensures discrimination-free access to the infrastructure for concurrent requests for the infrastructure in a **schedule planning phase**.
However, in the **Flat**land challenge, we focus on the re-scheduling problem during live operations.

Technically, 
``` 
RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Any]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]

AgentPosition = Tuple[int, int]
ScheduleGeneratorProduct = Tuple[List[AgentPosition], List[AgentPosition], List[AgentPosition], List[float]]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any]], ScheduleGeneratorProduct]
```

We can then produce `RailGenerator`s by currying:
```
def sparse_rail_generator(num_cities=5, num_intersections=4, num_trainstations=2, min_node_dist=20, node_radius=2,
                          num_neighb=3, grid_mode=False, enhance_intersection=False, seed=0):

    def generator(width, height, num_agents, num_resets=0):
    
        # generate the grid and (optionally) some hints for the schedule_generator
        ...
         
        return grid_map, {'agents_hints': {
            'num_agents': num_agents,
            'agent_start_targets_nodes': agent_start_targets_nodes,
            'train_stations': train_stations
        }}

    return generator
```
And, similarly, `ScheduleGenerator`s:
```
def sparse_schedule_generator(speed_ratio_map: Mapping[float, float] = None) -> ScheduleGenerator:
    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None):
        # place agents:
        # - initial position
        # - initial direction
        # - (initial) speed
        # - malfunction
        ...
                
        return agents_position, agents_direction, agents_target, speeds, agents_malfunction

    return generator
```
Notice that the `rail_generator` may pass `agents_hints` to the  `schedule_generator` which the latter may interpret.
For instance, the way the `sparse_rail_generator` generates the grid, it already determines the agent's goal and target.
Hence, `rail_generator` and `schedule_generator` have to match if `schedule_generator` presupposes some specific `agents_hints`.

The environment's `reset` takes care of applying the two generators:
```
    def __init__(self,
            ...
             rail_generator: RailGenerator = random_rail_generator(),
             schedule_generator: ScheduleGenerator = random_schedule_generator(),
             ...
             ):
        self.rail_generator: RailGenerator = rail_generator
        self.schedule_generator: ScheduleGenerator = schedule_generator
        
    def reset(self, regen_rail=True, replace_agents=True):
        rail, optionals = self.rail_generator(self.width, self.height, self.get_num_agents(), self.num_resets)

        ...

        if replace_agents:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']
            self.agents_static = EnvAgentStatic.from_lists(
                *self.schedule_generator(self.rail, self.get_num_agents(), hints=agents_hints))
```


## Example code

To see all the changes in action you can just run the `flatland_example_2_0.py` file in the examples folder. The file can be found [here](https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/flatland_2_0_example.py).
