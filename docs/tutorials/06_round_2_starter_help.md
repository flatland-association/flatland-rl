# How to get started in Round 2

- [Environment Changes](#environment-changes)
- [Level generation](#level-generation)
- [Observations](#observations)
- [Predictions](#predictions)

## Environment Changes
There have been some major changes in how agents are being handled in the environment in this Flatland update.
### Agents
Agents are no more permant entities in the environment. Now agents will be removed from the environment as soon as they finsish their task. To keep interactions with the environment as simple as possible we do not modify the dimensions of the observation vectors nor the number of agents. Agents that have finished do not require any special treatment from the controller. Any action provided to these agents is simply ignored, just like before.

Start positions of agents are *not unique* anymore. This means that many agents can start from the same position on the railway grid. It is important to keep in mind that whatever agent moves first will block the rest of the agents from moving into the same cell. Thus, the controller can already decide the ordering of the agents from the first step.

## Level Generation
The levels are now generated using the `sparse_rail_generator` and the `sparse_schedule_generator`
### Rail Generation
The rail generation is done in a sequence of steps:
1. A number of city centers are placed in a a grid of size `(height, width)`
2. Each city is connected to two neighbouring cities
3. Internal parallel tracks are generated in each city


### Schedule Generation
The `sparse_schedule_generator` produces tasks for the agents by selecting a starting city and a target city. The agent is then placed on an even track number on the starting city and faced such that a path exists to the target city. The task for the agent is to reach the target position as fast as possible.

In the future we will update how these schedules are generated to allow for more complex tasks

## Observations
Observations have been updated to reflect the novel features and behaviors of Flatland. Have a look at [observation](https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py) or the documentation for more details on the observations.

## Predicitons