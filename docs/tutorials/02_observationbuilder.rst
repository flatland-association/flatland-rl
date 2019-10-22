Custom observations and custom predictors Tutorial
==================================================

Overview
--------

One of the main objectives of the Flatland-Challenge_ is to find a suitable observation (relevant features for the problem at hand) to solve the task. Therefore **Flatland** was built with as much flexibility as possible when it comes to building your custom observations: observations in Flatland environments are fully customizable.
Whenever an environment needs to compute new observations for each agent, it queries an object derived from the :code:`ObservationBuilder` base class, which takes the current state of the environment and returns the desired observation.


.. _Flatland-Challenge: https://www.aicrowd.com/challenges/flatland-challenge

Example 1 : Simple (but useless) observation
--------------------------------------------
In this first example we implement all the functions necessary for the observation builder to be valid and work with **Flatland**.
Custom observation builder objects need to derive from the `flatland.core.env_observation_builder.ObservationBuilder`_
base class and must implement two methods, :code:`reset(self)` and :code:`get(self, handle)`.

.. _`flatland.core.env_observation_builder.ObservationBuilder` : https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/core/env_observation_builder.py#L13

Below is a simple example that returns observation vectors of size 5 featuring only the ID (handle) of the agent whose
observation vector is being computed:

.. code-block:: python

    class SimpleObs(ObservationBuilder):
        """
        Simplest observation builder. The object returns observation vectors with 5 identical components,
        all equal to the ID of the respective agent.
        """

        def reset(self):
            return

        def get(self, handle):
            observation = handle * np.ones(5)
            return observation

We can pass an instance of our custom observation builder :code:`SimpleObs` to the :code:`RailEnv` creator as follows:

.. code-block:: python

    env = RailEnv(width=7,
                  height=7,
                  rail_generator=random_rail_generator(),
                  number_of_agents=3,
                  obs_builder_object=SimpleObs())
    env.reset()

Anytime :code:`env.reset()` or :code:`env.step()` is called, the observation builder will return the custom observation of all agents initialized in the env.
In the next example we highlight how to derive from existing observation builders and how to access internal variables of **Flatland**.


Example 2 : Single-agent navigation
-------------------------------------

Observation builder objects can of course derive from existing concrete subclasses of ObservationBuilder.
For example, it may be useful to extend the TreeObsForRailEnv_ observation builder.
A feature of this class is that on :code:`reset()`, it pre-computes the lengths of the shortest paths from all
cells and orientations to the target of each agent, i.e. a distance map for each agent.

In this example we exploit these distance maps by implementing an observation builder that shows the current shortest path for each agent as a one-hot observation vector of length 3, whose components represent the possible directions an agent can take (LEFT, FORWARD, RIGHT). All values of the observation vector are set to :code:`0` except for the shortest direction where it is set to :code:`1`.

Using this observation with highly engineered features indicating the agent's shortest path, an agent can then learn to take the corresponding action at each time-step; or we could even hardcode the optimal policy.
Note that this simple strategy fails when multiple agents are present, as each agent would only attempt its greedy solution, which is not usually `Pareto-optimal <https://en.wikipedia.org/wiki/Pareto_efficiency>`_ in this context.

.. _TreeObsForRailEnv: https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py#L14

.. code-block:: python

    from flatland.envs.observations import TreeObsForRailEnv

    class SingleAgentNavigationObs(TreeObsForRailEnv):
        """
        We derive our observation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
        the minimum distances from each grid node to each agent's target.

        We then build a representation vector with 3 binary components, indicating which of the 3 available directions
        for each agent (Left, Forward, Right) lead to the shortest path to its target.
        E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
        will be [1, 0, 0].
        """
        def __init__(self):
            super().__init__(max_depth=0)
            # We set max_depth=0 in because we only need to look at the current
            # position of the agent to decide what direction is shortest.

        def reset(self):
            # Recompute the distance map, if the environment has changed.
            super().reset()

        def get(self, handle):
            # Here we access agent information from the environment.
            # Information from the environment can be accessed but not changed!
            agent = self.env.agents[handle]

            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
            num_transitions = np.count_nonzero(possible_transitions)

            # Start from the current orientation, and see which transitions are available;
            # organize them as [left, forward, right], relative to the current orientation
            # If only one transition is possible, the forward branch is aligned with it.
            if num_transitions == 1:
                observation = [0, 1, 0]
            else:
                min_distances = []
                for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                    if possible_transitions[direction]:
                        new_position = self._new_position(agent.position, direction)
                        min_distances.append(self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                    else:
                        min_distances.append(np.inf)

                observation = [0, 0, 0]
                observation[np.argmin(min_distances)] = 1

            return observation

    env = RailEnv(width=7,
                  height=7,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, \
                    min_dist=8, max_dist=99999, seed=1),
                  number_of_agents=2,
                  obs_builder_object=SingleAgentNavigationObs())
    env.reset()

    obs, all_rewards, done, _ = env.step({0: 0, 1: 1})
    for i in range(env.get_num_agents()):
        print(obs[i])

Finally, the following is an example of hard-coded navigation for single agents that achieves optimal single-agent
navigation to target, and shows the path taken as an animation.

.. code-block:: python

    env = RailEnv(width=50,
                  height=50,
                  rail_generator=random_rail_generator(),
                  number_of_agents=1,
                  obs_builder_object=SingleAgentNavigationObs())
    env.reset()

    obs, all_rewards, done, _ = env.step({0: 0})

    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env(show=True, frames=True, show_observations=False)

    for step in range(100):
        action = np.argmax(obs[0])+1
        obs, all_rewards, done, _ = env.step({0:action})
        print("Rewards: ", all_rewards, "  [done=", done, "]")

        env_renderer.render_env(show=True, frames=True, show_observations=False)
        time.sleep(0.1)

The code examples above appear in the example file `custom_observation_example.py <https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/custom_observation_example.py>`_. You can run it using :code:`python examples/custom_observation_example.py` from the root folder of the flatland repo.  The two examples are run one after the other.

Example 3 : Using custom predictors and rendering observation
-------------------------------------------------------------

Because the re-scheduling task of the Flatland-Challenge_ requires some short time planning we allow the possibility to use custom predictors that help predict upcoming conflicts and help agent solve them in a timely manner.
In the **Flatland Environment** we included an initial predictor ShortestPathPredictorForRailEnv_ to give you an idea what you can do with these predictors.

Any custom predictor can be passed to the observation builder and then be used to build the observation. In this example_ we illustrate how an observation builder can be used to detect conflicts using a predictor.

The observation is incomplete as it only contains information about potential conflicts and has no feature about the agent objectives.

In addition to using your custom predictor you can also make your custom observation ready for rendering. (This can be done in a similar way for your predictor).
All you need to do in order to render your custom observation is to populate  :code:`self.env.dev_obs_dict[handle]` for every agent (all handles). (For the predictor use  :code:`self.env.dev_pred_dict[handle]`).

In contrast to the previous examples we also implement the :code:`def get_many(self, handles=None)` function for this custom observation builder. The reasoning here is that we want to call the predictor only once per :code:`env.step()`. The base implementation of :code:`def get_many(self, handles=None)` will call the :code:`get(handle)` function for all handles, which mean that it normally does not need to be reimplemented, except for cases as the one below.

.. _ShortestPathPredictorForRailEnv: https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/predictions.py#L81
.. _example: https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/custom_observation_example.py#L110

.. code-block:: python

    class ObservePredictions(TreeObsForRailEnv):
        """
        We use the provided ShortestPathPredictor to illustrate the usage of predictors in your custom observation.

        We derive our observation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
        the minimum distances from each grid node to each agent's target.

        This is necessary so that we can pass the distance map to the ShortestPathPredictor

        Here we also want to highlight how you can visualize your observation
        """

        def __init__(self, predictor):
            super().__init__(max_depth=0)
            self.predictor = predictor

        def reset(self):
            # Recompute the distance map, if the environment has changed.
            super().reset()

        def get_many(self, handles=None):
            '''
            Because we do not want to call the predictor seperately for every agent we implement the get_many function
            Here we can call the predictor just ones for all the agents and use the predictions to generate our observations
            :param handles:
            :return:
            '''

            self.predictions = self.predictor.get()

            self.predicted_pos = {}
            for t in range(len(self.predictions[0])):
                pos_list = []
                for a in handles:
                    pos_list.append(self.predictions[a][t][1:3])
                # We transform (x,y) coodrinates to a single integer number for simpler comparison
                self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
            observations = {}

            # Collect all the different observation for all the agents
            for h in handles:
                observations[h] = self.get(h)
            return observations

        def get(self, handle):
            '''
            Lets write a simple observation which just indicates whether or not the own predicted path
            overlaps with other predicted paths at any time. This is useless for the task of navigation but might
            help when looking for conflicts. A more complex implementation can be found in the TreeObsForRailEnv class

            Each agent recieves an observation of length 10, where each element represents a prediction step and its value
            is:
             - 0 if no overlap is happening
             - 1 where n i the number of other paths crossing the predicted cell

            :param handle: handeled as an index of an agent
            :return: Observation of handle
            '''

            observation = np.zeros(10)

            # We are going to track what cells where considered while building the obervation and make them accesible
            # For rendering

            visited = set()
            for _idx in range(10):
                # Check if any of the other prediction overlap with agents own predictions
                x_coord = self.predictions[handle][_idx][1]
                y_coord = self.predictions[handle][_idx][2]

                # We add every observed cell to the observation rendering
                visited.add((x_coord, y_coord))
                if self.predicted_pos[_idx][handle] in np.delete(self.predicted_pos[_idx], handle, 0):
                    # We detect if another agent is predicting to pass through the same cell at the same predicted time
                    observation[handle] = 1

            # This variable will be access by the renderer to visualize the observation
            self.env.dev_obs_dict[handle] = visited

            return observation

We can then use this new observation builder and the renderer to visualize the observation of each agent.


.. code-block:: python

    # Initiate the Predictor
    CustomPredictor = ShortestPathPredictorForRailEnv(10)

    # Pass the Predictor to the observation builder
    CustomObsBuilder = ObservePredictions(CustomPredictor)

    # Initiate Environment
    env = RailEnv(width=10,
                  height=10,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=1, min_dist=8, max_dist=99999, seed=1),
                  number_of_agents=3,
                  obs_builder_object=CustomObsBuilder)
    env.reset()

    obs, info = env.reset()
    env_renderer = RenderTool(env, gl="PILSVG")

    # We render the initial step and show the obsered cells as colored boxes
    env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)

    action_dict = {}
    for step in range(100):
        for a in range(env.get_num_agents()):
            action = np.random.randint(0, 5)
            action_dict[a] = action
        obs, all_rewards, done, _ = env.step(action_dict)
        print("Rewards: ", all_rewards, "  [done=", done, "]")
        env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)
        time.sleep(0.5)

How to access environment and agent data for observation builders
------------------------------------------------------------------

When building your custom observation builder, you might want to aggregate and define your own features that are different from the raw env data. In this section we introduce how such information can be accessed and how you can build your own features out of them.

Transitions maps
~~~~~~~~~~~~~~~~

The transition maps build the base for all movement in the environment. They contain all the information about allowed transitions for the agent at any given position. Because railway movement is limited to the railway tracks, these are important features for any controller that want to interact with the environment. All functionality and features of a transition map can be found here_.

.. _here: https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/core/transition_map.py

**Get Transitions for cell**

To access the possible transitions at any given cell there are different possibilites:

1. You provide a cell position and a orientation in that cell (usually the orientation of the agent) and call :code:`cell_transitions = env.rail.get_transitions(*position, direction)` and in return you get a 4d vector with the transition probability ordered as :code:`[North, East, South, West]` given the initial orientation. The position is a tuple of the form :code:`(x, y)` where :code:`x in [0, height]` and :code:`y in [0, width]`. This can be used for branching in a tree search and when looking for all possible allowed paths of an agent as it will provide a simple way to get the possible trajectories.

2. When more detailed information about the cell in general is necessary you can also get the full transitions of a cell by calling :code:`transition_int = env.rail.get_full_transitions(*position)`. This will return an :code:`int16` for the cell representing the allowed transitions. To understand the transitions returned it is best to represent it as a binary number :code:`bin(transition_int)`, where the bits have to following meaning: :code:`NN NE NS NW EN EE ES EW SN SE SS SW WN WE WS WW`. For example the binary code :code:`1000 0000 0010 0000`, represents a straigt where an agent facing north can transition north and an agent facing south can transition south and no other transitions are possible. To get a better feeling what the binary representations of the elements look like go to this Link_

.. _Link: https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/core/grid/rail_env_grid.py#L29


These two objects can be used for example to detect switches that are usable by other agents but not the observing agent itself. This can be an important feature when actions have to be taken in order to avoid conflicts.

.. code-block:: python

    cell_transitions = self.env.rail.get_transitions(*position, direction)
    transition_bit = bin(self.env.rail.get_full_transitions(*position))

    total_transitions = transition_bit.count("1")
    num_transitions = np.count_nonzero(cell_transitions)

    # Detect Switches that can only be used by other agents.
    if total_transitions > 2 > num_transitions:
        unusable_switch_detected = True


Agent information
~~~~~~~~~~~~~~~~~~

The agents are represented as an agent class and are provided when the environment is instantiated. Because agents can have different properties it is helpful to know how to access this information.

You can simply acces the three main types of agent information in the following ways with :code:`agent = env.agents[handle]`:

**Agent basic information**
All the agent in the initiated environment can be found in the :code:`env.agents` class. Given the index of the agent you have acces to:

- Agent position :code:`agent.position` which returns the current coordinates :code:`(x, y)` of the agent.
- Agent target :code:`agent.target`  which returns the target coordinates :code:`(x, y)`.
- Agent direction :code:`agent.direction` which is an int representing the current orientation :code:`{0: North, 1: East, 2: South, 3: West}`
- Agent moving :code:`agent.moving` where 0 means the agent is currently not moving and 1 indicates agent is moving.

**Agent speed information**

Beyond the basic agent information we can also access more details about the agents type by looking at speed data:

- Agent max speed :code:`agent.speed_data["speed"]` wich defines the traveling speed when the agent is moving.
- Agent position fraction :code:`agent.speed_data["position_fraction"]` which is a number between 0 and 1 and indicates when the move to the next cell will occur. Each speed of an agent is 1 or a smaller fraction. At each :code:`env.step()` the agent moves at its fractional speed forwards and only changes to the next cell when the cumulated fractions are :code:`agent.speed_data["position_fraction"] >= 1.`
- Agent can move at different speed which can be set up by modifying the agent.speed_data within the schedule_generator. For example refer this _Link_Schedule_Generators.

.. _Link_Schedule_Generators: https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/schedule_generators.py#L59

**Agent malfunction information**

Similar to the speed data you can also access individual data about the malfunctions of an agent. All data is available through :code:`agent.malfunction_data` with:

- Indication how long the agent is still malfunctioning :code:`'malfunction'` by an integer counting down at each time step. 0 means the agent is ok and can move.
- Possion rate at which malfunctions happen for this agent :code:`'malfunction_rate'`
- Number of steps untill next malfunction will occur :code:`'next_malfunction'`
- Number of malfunctions an agent have occured for this agent so far :code:`nr_malfunctions'`

