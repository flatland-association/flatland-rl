# Stochasticity Tutorial

Another area where we improved **Flat**land 2.0 are stochastic events added during the episodes.
This is very common for railway networks where the initial plan usually needs to be rescheduled during operations as minor events such as delayed departure from trainstations, malfunctions on trains or infrastructure or just the weather lead to delayed trains.

We implemted a poisson process to simulate delays by stopping agents at random times for random durations. The parameters necessary for the stochastic events can be provided when creating the environment.

```python
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

```python
env = RailEnv(
    ...
    stochastic_data=stochastic_data,  # Malfunction data generator
    ...
)
```
In your controller, you can check whether an agent is malfunctioning:
```python
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
env.reset()
```

You will quickly realize that this will lead to unforeseen difficulties which means that **your controller** needs to observe the environment at all times to be able to react to the stochastic events.

## Example code

To see all the changes in action you can just run the `flatland_example_2_0.py` file in the examples folder. The file can be found [here](https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/flatland_2_0_example.py).
