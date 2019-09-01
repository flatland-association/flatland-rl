# Flatland 2.0 Introduction

## Whats new

In this version of **Flat**land we are moving closer to realistic and more complex railway problems. Earlier versions of **Flat**land which introduced you to the concept of restricted transitions was still to simplified to give us feasible solutions for daily operations. Thus the following changes are coming in the next version to be closer to real railway network challenges:

- **New Level Generator** with less connections between different nodes in the network and thus much higher agent densities on rails.
- **Stochastic Events** that cause agents to stop and get stuck for different number of time steps.
- **Different Speed Classes** allow agents to move at different speeds and thus enhance complexity in the search for optimal solutions.

Below we explain these changes in more detail and how you can play with their parametrization. We appreciate *your feedback* on the performance and the difficulty on these levels to help us shape the best possible **Flat**land 2.0 environment.

## Get the new level generators
Since this is currently still in *beta* phase you can only install this version of **Flat**land through the gitlab repository. Once you have downloaded the [Flatland Repository](https://gitlab.aicrowd.com/flatland/flatland) you have to switch to the [147_new_level_generator](https://gitlab.aicrowd.com/flatland/flatland/tree/147_new_level_generator) branch to be able access the latest changes in **Flat**land.

Once you have switched to this branch install **Flat**land by running `python setup.py install`.

## Generate levels

We are currently working on different new level generators and you can expect that the levels in the submission testing will not all come from just one but rather different level generators to be sure that the controllers can handle any railway specific challenge.

For this early **beta** testing we suggest you have a look at the `sparse_rail_generator` and `realistic_rail_generator`.

### Sparse Rail Generator
![Example_Sparse](https://i.imgur.com/DP8sIyx.png)

The idea behind the sparse rail generator is to mimic classic railway structures where dense nodes (cities) are sparsly connected to each other and where you have to manage traffic flow between the nodes efficiently. The cities in this level generator are much simplified in comparison to real city networks but it mimics parts of the problems faced in daily operations of any railway company.

There are a few parameters you can tune to build your own map and test different complexity levels of the levels. **Warning** some combinations of parameters do not go well together and will lead to infeasible level generation. In the worst case, the level generator currently issues a warning when it cannot build the environment according to the parameters provided. This will lead to a crash of the whole env. We are currently working on improvements here and are **happy for any suggestions from your side**.

To build en environment you instantiate a `RailEnv` follows

```
# Initialize the generator
RailGenerator = sparse_rail_generator(num_cities=10,                        # Number of cities in map
                                                   num_intersections=10,    # Number of interesections in map
                                                   num_trainstations=50,    # Number of possible start/targets on map
                                                   min_node_dist=6,         # Minimal distance of nodes
                                                   node_radius=3,           # Proximity of stations to city center
                                                   num_neighb=3,            # Number of connections to other cities
                                                   seed=5,                  # Random seed
                                                   grid_mode=True      # Ordered distribution of nodes
                                                   )

# Build the environment
env = RailEnv(width=50,
              height=50,
              rail_generator=RailGenerator,
              number_of_agents=10,
              obs_builder_object=TreeObsForRailEnv(max_depth=3,predictor=shortest_path_predictor)
              )
```

You can tune the following parameters:

- `num_citeis` is the number of cities on a map. Cities are the only nodes that can host start and end points for agent tasks (Train stations). Here you have to be carefull that the number is not too high as all the cities have to fit on the map. When `grid_mode=False` you have to be carefull when chosing `min_node_dist` because leves will fails if not all cities (and intersections) can be placed with at least `min_node_dist` between them.
- `num_intersections` is the number of nodes that don't hold any trainstations. They are also the first priority that a city connects to. We use these to allow for sparse connections between cities.
- `num_trainstations`defines the *Total* number of trainstations in the network. This also sets the max number of allowed agents in the environment. This is also a delicate parameter as there is only a limitid amount of space available around nodes and thus if the number is too high the level generation will fail. *Important*: Only the number of agents provided to the environment will actually produce active train stations. The others will just be present as dead-ends (See figures below).
- `min_node_dist`is only used if `grid_mode=False` and represents the minimal distance between two nodes.
- `node_radius` defines the extent of a city. Each trainstation is placed at a distance to the closes city node that is smaller or equal to this number.
- `num_neighb`defines the number of neighbouring nodes that connect to each other. Thus this changes the connectivity and thus the amount of alternative routes in the network.
- `seed` is used to initialize the random generator
- `grid_mode` currently only changes how the nodes are distirbuted. If it is set to `True` the nodes are evenly spreas out and cities and intersecitons are set between each other.

If you run into any bugs with sets of parameters please let us know.

Here is a network with `grid_mode=False` and the parameters from above.

![sparse_random](https://i.imgur.com/Xg7nifF.png)

and here with `grid_mode=True`

![sparse_ordered](https://i.imgur.com/jyA7Pt4.png)

## Add Stochasticity

Another area where we improve **Flat**land 2.0 is by adding stochastic events during the episodes. This is very common for railway networks where the initial plan usually needs to be rescheduled during operations as minor events such as delayed departure from trainstations, malfunctions on trains or infrastructure or just the weather lead to delayed trains.

We implemted a poisson process to simulate delays by stopping agents at random times for random durations. The parameters necessary for the stochastic events can be provided when creating the environment.

```
# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.5,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 10  # Max duration of malfunction
                   }

```

The parameters are as follows:

- `prop_malfunction` is the proportion of agents that can malfunction. `1.0` means that each agent can break.
- `malfunction_rate` is the mean rate of the poisson process in number of environment steps.
- `min_dutation` and `max_duration` set the range of malfunction durations. They are sampled uniformly

You can introduce stochasticity by simply creating the env as follows:

```
# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.1,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }

# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

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
              obs_builder_object=TreeObservation)
```

You will quickly realize that this will lead to unforseen difficulties which means that **your controller** needs to observe the environment at all times to be able to react to the stochastic events.

## Add different speed profiles

One of the main contributions to the complexity of railway network operations stems from the fact that all trains travel at different speeds while sharing a very limited railway network. In **Flat**land 2.0 this feature will be enabled as well and will lead to much more complex configurations. This is still in early *beta* and even though stock observation builders and predictors do support these changes we have not yet fully tested them. Here we count on your support :).

The different speed profiles can be generated using the `schedule_generator`. The schedule 

Where you can actually chose as many different speeds as you like. Keep in mind that the *fastest speed* is 1 and all slower speeds must be between 1 and 0. For the submission scoring you can assume that there will be no more than 5 speed profiles.

## Actions and observation with different speed levels

Because the different speeds are implemented as fractions the agents ability to perform actions has been updated. We **do not allow actions to change within the cell **. This means that each agent can only chose an action to be taken when entering a cell. This action is then executed when a step to the next cell is valid. For example

- Agent enters switch and choses to deviate left. Agent fractional speed is 1/4 and thus the agent will take for time steps to complete its journey through the cell. On the 4th time step the agent will leave the cell deviating left as chosen at the entry of the cell.
    - All actions chosen by the agent during its travels within a cell are ignored
    - Agents can make observations at any time step. Make sure to dscard observations without any information. See this [example](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/training_navigation.py) for a simple implementation.
- The environment checks if agent is allowed to move to next cell only at the time of the switch to the next cell


## Example code

To see all the changes in action you can just run the `flatland_example_2_0.py` file in the examples folder. The file can be found [here](https://gitlab.aicrowd.com/flatland/flatland/blob/147_new_level_generator/examples/flatland_2_0_example.py).
