# Flatland 2.0 Introduction (Beta)

Curious to see whats coming in *Flat*land 2.0? Have a look at the current development and report bugs and give us feedback on the environment.

*WARNING*: Flatlnadn 2.0 Beta is under current development and not stable nor final. We would however like you to play with the code and help us get the best possible environment for multi-agent control problems.

## Whats new

In this version of *Flat*land we are moving closer to realistic and more complex railway problems. Earlier versions of *Flat*land which introduced you to the concept of restricted transitions was still to simplified to give us feasible solutions for daily operations. Thus the following changes are coming in the next version to be closer to real railway network challenges:

- *New Level Generator* with less connections between different nodes in the network and thus much higher agent densities on rails.
- *Stochastic Events* that cause agents to stop and get stuck for different number of time steps.
- *Different Speed Classes* allow agents to move at different speeds and thus enhance complexity in the search for optimal solutions.

Below we explain these changes in more detail and how you can play with their parametrization. We appreciate *your feedback* on the performance and the difficulty on these levels to help us shape the best possible *Flat*land 2.0 environment.

## Get the new level generators
Since this is currently still in *beta* phase you can only install this version of *Flat*land through the gitlab repository. Once you have downloaded the [Flatland Repository](https://gitlab.aicrowd.com/flatland/flatland) you have to switch to the [147_new_level_generator](https://gitlab.aicrowd.com/flatland/flatland/tree/147_new_level_generator) branch to be able access the latest changes in *Flat*land.

Once you have switched to this branch install *Flat*land by running `python setup.py install`.

## Generate levels

We are currently working on different new level generators and you can expect that the levels in the submission testing will not all come from just one but rather different level generators to be sure that the controllers can handle any railway specific challenge.

For this early *beta* testing we suggest you have a look at the `sparse_rail_generator` and `realistic_rail_generator`.

### Sparse Rail Generator
![Example_Sparse](https://i.imgur.com/DP8sIyx.png)

The idea behind the sparse rail generator is to mimic classic railway structures where dense nodes (cities) are sparsly connected to each other and where you have to manage traffic flow between the nodes efficiently. The cities in this level generator are much simplified in comparison to real city networks but it mimics parts of the problems faced in daily operations of any railway company.

There are a few parameters you can tune to build your own map and test different complexity levels of the levels. *Warning* some combinations of parameters do not go well together and will lead to infeasible level generation. In the worst case, the level generator currently issues a warning when it cannot build the environment according to the parameters provided. This will lead to a crash of the whole env. We are currently working on improvements here and are *happy for any suggestions from your side*.

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
                                                   realistic_mode=True      # Ordered distribution of nodes
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

- `num_citeis` is the number of cities on a map. Cities are the only nodes that can host start and end points for agent tasks (Train stations). Here you have to be carefull that the number is not too high as all the cities have to fit on the map. When `realistic_mode=False` you have to be carefull when chosing `min_node_dist` because leves will fails if not all cities (and intersections) can be placed with at least `min_node_dist` between them.
- `num_intersections` is the number of nodes that don't hold any trainstations. They are also the first priority that a city connects to. We use these to allow for sparse connections between cities.
- `num_trainstations`defines the *Total* number of trainstations in the network. This also sets the max number of allowed agents in the environment. This is also a delicate parameter as there is only a limitid amount of space available around nodes and thus if the number is too high the level generation will fail.
- `min_node_dist`is only used if `realistic_mode=False` and represents the minimal distance between two nodes.
- `node_radius` defines the extent of a city. Each trainstation is placed at a distance to the closes city node that is smaller or equal to this number.
- `num_neighb`defines the number of neighbouring nodes that connect to each other. Thus this changes the connectivity and thus the amount of alternative routes in the network.
- `seed` is used to initialize the random generator
- `realistic_mode` currently only changes how the nodes are distirbuted. If it is set to `True` the nodes are evenly spreas out and cities and intersecitons are set between each other.

If you run into any bugs with sets of parameters please let us know.

Here is a network with `realistic_mode=False`

![sparse_random](https://i.imgur.com/Xg7nifF.png)

and here with `realistic_mode=True`

![sparse_ordered](https://i.imgur.com/jyA7Pt4.png)

## Add Stochasticity

## Add different speed profiles