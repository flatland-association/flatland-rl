# Keeping track of major Flatland Changes

## Changes since Flatland 0.3
### Changes in stock predictors
The stock `ShortestPathPredictorForRailEnv` now respects the different agent speeds and updates their prediction accordingly.

### Changes in stock observation biulders
- `TreeObsForRailEnv` now has **11** features!
    - 10th feature now indicates if a malfunctioning agent has been detected and how long the malfunction will still be present
    - 11th feautre now indicates the minimal observed fractional speed of agents traveling in the same direction
### Changes in level generation
- Separation of `schedule_generator` from `rail_generator`: 
  - Renaming of `flatland/envs/generators.py` to `flatland/envs/rail_generators.py`
  - `rail_generator` now only returns the grid and optionally hints (a python dictionary); the hints are currently use for distance_map and communication of start and goal position in complex rail generator.
  - `schedule_generator` takes a `GridTransitionMap` and the number of agents and optionally the `agents_hints` field of the hints dictionary.
  - Inrodcution of types hints: 
``` 
RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Any]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]
AgentPosition = Tuple[int, int]
ScheduleGeneratorProduct = Tuple[List[AgentPosition], List[AgentPosition], List[AgentPosition], List[float]]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any]], ScheduleGeneratorProduct]
```

### Multi Speed

- Different agent speeds are introduced. Agents now travel at a max speed which is afraction of 1.
    - Fastest speed is 1. At this speed an agent can move to a new cell at each time step t.
    - Slower speeds are smaller than one. At each time step an agent moves the fraction of its speed forward within a cell. It only changes cell when it's fractional position is greater or equal to 1.
    - Multi-speed introduces the challenge of ordering the trains correctly when traveling in the same direction.
- Agents always travel at their full speed when moving.

To set up multiple speeds you have to modify the `agent.speed_data` within your `schedule_generator`. See [this file](https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/schedule_generators.py#L59) for a good example.


### Stochastic events
Just like in real-worl transportation systems we introduced stochastic events to disturb normal traffic flow. Currently we implemented a malfunction process that stops agents at random time intervalls for a random time of duration.
Currently the Flatland environment can be initiated with the following poisson process parameters:

```
# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.1,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }
```
The duration of a malfunction is uniformly drawn from the intervall `[min_duration,max_duration0]` and the occurance of malfunctions follows a point poisson process with mean rate `malfunctin_rate`.

## Changes since Flatland 0.2

Please list all major changes since the last version:

- Refactoring of rendering code: CamelCase functions changed to snake_case
- Tree Observation Added a new Featuer: `unusable_switch` which indicates switches that are not branchingpoints for the observing agent
- Updated the shortest path predictor
- Updated conflict detection with predictor
- Episodes length can be set as maximum number of steps allowed.