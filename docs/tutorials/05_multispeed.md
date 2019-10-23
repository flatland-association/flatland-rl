# Different speed profiles Tutorial

One of the main contributions to the complexity of railway network operations stems from the fact that all trains travel at different speeds while sharing a very limited railway network.
In **Flat**land 2.0 this feature will be enabled as well and will lead to much more complex configurations. Here we count on your support if you find bugs or improvements  :).

The different speed profiles can be generated using the `schedule_generator`, where you can actually chose as many different speeds as you like.
Keep in mind that the *fastest speed* is 1 and all slower speeds must be between 1 and 0.
For the submission scoring you can assume that there will be no more than 5 speed profiles.



Later versions of **Flat**land might have varying speeds during episodes. Therefore, we return the agent speeds.
Notice that we do not guarantee that the speed will be computed at each step, but if not costly we will return it at each step.
In your controller, you can get the agents' speed from the `info` returned by `step`:
```python
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
```python
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
```python
RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Any]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]

AgentPosition = Tuple[int, int]
Schedule = collections.namedtuple('Schedule',   'agent_positions '
                                                'agent_directions '
                                                'agent_targets '
                                                'agent_speeds '
                                                'agent_malfunction_rates '
                                                'max_episode_steps')
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Schedule]
```

We can then produce `RailGenerator`s by currying:
```python
def sparse_rail_generator(num_cities=5, num_intersections=4, num_trainstations=2, min_node_dist=20, node_radius=2,
                          num_neighb=3, grid_mode=False, enhance_intersection=False, seed=1):

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
```python
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
```python
    def __init__(self,
            ...
             rail_generator: RailGenerator = random_rail_generator(),
             schedule_generator: ScheduleGenerator = random_schedule_generator(),
             ...
             ):
        self.rail_generator: RailGenerator = rail_generator
        self.schedule_generator: ScheduleGenerator = schedule_generator

    def reset(self, regenerate_rail=True, regenerate_schedule=True):
        rail, optionals = self.rail_generator(self.width, self.height, self.get_num_agents(), self.num_resets)

        ...

        if replace_agents:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']
            self.agents_static = EnvAgentStatic.from_lists(
                self.schedule_generator(self.rail, self.get_num_agents(), hints=agents_hints))
```


## Example code

To see all the changes in action you can just run the `flatland_example_2_0.py` file in the examples folder. The file can be found [here](https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/flatland_2_0_example.py).
