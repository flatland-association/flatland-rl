# Level Generation Tutorial

We are currently working on different new level generators and you can expect that the levels in the submission testing will not all come from just one but rather different level generators to be sure that the controllers can handle any railway specific challenge.

Let's have a look at the `sparse_rail_generator`.

## Sparse Rail Generator
![Example_Sparse](https://i.imgur.com/DP8sIyx.png)

The idea behind the sparse rail generator is to mimic classic railway structures where dense nodes (cities) are sparsely connected to each other and where you have to manage traffic flow between the nodes efficiently.
The cities in this level generator are much simplified in comparison to real city networks but it mimics parts of the problems faced in daily operations of any railway company.

There are a few parameters you can tune to build your own map and test different complexity levels of the levels.
**Warning** some combinations of parameters do not go well together and will lead to infeasible level generation.
In the worst case, the level generator currently issues a warning when it cannot build the environment according to the parameters provided.
This will lead to a crash of the whole env.
We are currently working on improvements here and are **happy for any suggestions from your side**.

To build an environment you instantiate a `RailEnv` as follows:

```python
 Initialize the generator
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

 Build the environment
env = RailEnv(
    width=50,
    height=50,
    rail_generator=rail_generator
    schedule_generator=sparse_schedule_generator(),
    number_of_agents=10,
    obs_builder_object=TreeObsForRailEnv(max_depth=3,predictor=shortest_path_predictor)
)
 Call reset on the environment
env.reset()
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

## Example code

To see all the changes in action you can just run the `flatland_example_2_0.py` file in the examples folder. The file can be found [here](https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/flatland_2_0_example.py).
