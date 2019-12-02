Changelog
==========
Changes since Flatland 2.1.0
--------------------------

### Changes in 'schedule_generators'
- Schedule generators now provide the max number of steps allowed per episode
- Pickle files generated with older versions of Flatland need to be regenerated in order to include `_max_episode_steps`
Changes since Flatland 2.0.0
--------------------------
### Changes in `EnvAgent`
- class `EnvAgentStatic` was removed, so there is only class `EnvAgent` left which should simplify the handling of agents. The member `self.agents_static` of `RailEnv` was therefore also removed. Old Scence saved as pickle files cannot be loaded anymore.

### Changes in malfunction behavior
- agent attribute `next_malfunction`is not used anymore, it will be removed fully in future versions.
- `break_agent()` function is introduced which induces malfunctions in agent according to poisson process
- `_fix_agent_after_malfunction()` fixes agents after attribute `malfunction == 0`
- Introduced the concept of malfunction generators. Here you can add different malfunction models in future updates. Currently it only loads from files and parameters.

### Changes in `Environment`
- moving of member variable `distance_map_computed` to new class `DistanceMap`

### Changes in rail generator and `RailEnv`
- renaming of `distance_maps` into `distance_map`
- by default the reset method of RailEnv is not called in the constructor of RailEnv anymore (compliance for OpenAI Gym). Therefore the reset method needs to be called after the creation of a RailEnv object
- renaming of parameters RailEnv.reset(): from `regen_rail` to `regenerate_rail`, from `replace_agents` to `regenerate_schedule`

### Changes in schedule generation
- return value of schedule generator has changed to the named tuple `Schedule`. From the point of view of a consumer, nothing has changed, this is just a type hint which is introduced where the attributes of `Schedule` have names.

Changes since Flatland 1.0.0
--------------------------
### Changes in stock predictors
The stock `ShortestPathPredictorForRailEnv` now respects the different agent speeds and updates their prediction accordingly.

### Changes in stock observation biulders

- `TreeObsForRailEnv` now has **11** features!
    - 10th feature now indicates if a malfunctioning agent has been detected and how long the malfunction will still be present
    - 11th feautre now indicates the minimal observed fractional speed of agents traveling in the same direction
- `GlobalObsForRailEnv` now has new features!
    - Targets and other agent targets still represented in same way
    - `obs_agents_state` now contains 4 channels
        - 0th channel -> agent direction at agent position
        - 1st channel -> other agents direction at their positions
        - 2nd channel -> all agent malfunction duration at their positions
        - 3rd channel -> all agent fractional speeds at their positions
- `LocalObsForRailEnv` was not update to Flatland 2.0 because it was never used by participants of the challenge.


### Changes in level generation


- Separation of `schedule_generator` from `rail_generator`:
  - Renaming of `flatland/envs/generators.py` to `flatland/envs/rail_generators.py`
  - `rail_generator` now only returns the grid and optionally hints (a python dictionary); the hints are currently use for distance_map and communication of start and goal position in complex rail generator.
  - `schedule_generator` takes a `GridTransitionMap` and the number of agents and optionally the `agents_hints` field of the hints dictionary.
  - Inrodcution of types hints:

```python
RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Any]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]
AgentPosition = Tuple[int, int]
ScheduleGeneratorProduct = Tuple[List[AgentPosition], List[AgentPosition], List[AgentPosition], List[float]]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any]], ScheduleGeneratorProduct]
```

### Multi Speed

- Different agent speeds are introduced. Agents now travel at a max speed which is a fraction. Meaning that they only advance parts within a cell and need several steps to move to the next cell.
    - Fastest speed is 1. At this speed an agent can move to a new cell at each time step t.
    - Slower speeds are smaller than one. At each time step an agent moves the fraction of its speed forward within a cell. It only changes cell when it's fractional position is greater or equal to 1.
    - Multi-speed introduces the challenge of ordering the trains correctly when traveling in the same direction.
- Agents always travel at their full speed when moving.

To set up multiple speeds you have to modify the `agent.speed_data` within your `schedule_generator`. See [this file](https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/schedule_generators.py#L59) for a good example.

**ATTENTION** multi speed means that the agents actions are not registered on every time step. Only at new cell entry can new actions be chosen! Beware to respect this with your controller as actions are only important at the specific time steps! This is shown as an example in the [navigation training](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/training_navigation.py#L163)

### Stochastic events
Just like in real-worl transportation systems we introduced stochastic events to disturb normal traffic flow. Currently we implemented a malfunction process that stops agents at random time intervalls for a random time of duration.
Currently the Flatland environment can be initiated with the following poisson process parameters:

```python
# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.1,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }
```
The duration of a malfunction is uniformly drawn from the intervall `[min_duration,max_duration0]` and the occurance of malfunctions follows a point poisson process with mean rate `malfunctin_rate`.

**!!!!IMPORTANT!!!!** Once a malfunction duration has finished, the agent will **automatically** resume movement. This is important because otherwise it can get stuck in fractional positions and your code might forget to restart the agent at the first possible time. Therefore this has been automated. You can however stop the agent again at the next cell. This might in rare occasions lead to unexpected behavior, we are looking into this and will push a fix soon.


## Baselines repository

The baselines repository is not yet fully updated to handle multi-speed and stochastic events. Training needs to be modified to omitt all states inbetween the states where an agent can chose an action. Simple navigation training is already up to date. See [here](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/training_navigation.py) for more details.

Changes since Flatland 0.2
--------------------------
Please list all major changes since the last version:

- Refactoring of rendering code: CamelCase functions changed to snake_case
- Tree Observation Added a new Featuer: `unusable_switch` which indicates switches that are not branchingpoints for the observing agent
- Updated the shortest path predictor
- Updated conflict detection with predictor
- Episodes length can be set as maximum number of steps allowed.
