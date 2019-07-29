=====
Getting Started with custom observations
=====

Overview
--------------

One of the main objectives of the Flatland-Challenge_ is to find a suitable observation (relevant features for the problem at hand) to solve the task. Therefore **Flatland** was build with as much flexibility as possible when it comes to building your custom observations. Observations in Flatland environments are fully customizable. Whenever an environment needs to compute new observations for each agent, it queries an object derived from the :code:`ObservationBuilder` base class, which takes the current state of the environment and returns the desired observation.


.. _Flatland-Challenge: https://www.aicrowd.com/challenges/flatland-challenge

Example 1 : Simple (but useless) observation
--------------
In this first example we implement all the functions necessary for the observation builder to be valid and work with **Flatland**.
Custom observation builder objects need to derive from the `flatland.core.env_observation_builder.ObservationBuilder`_
base class and must implement two methods, :code:`reset(self)` and :code:`get(self, handle)`.

.. _`flatland.core.env_observation_builder.ObservationBuilder` : https://gitlab.aicrowd.com/flatland/flatland/blob/obsbuildertut/flatland/core/env_observation_builder.py#L13

Following is a simple example that returns observation vectors of size :code:`observation_space = 5` featuring only the ID (handle) of the agent whose
observation vector is being computed:

.. code-block:: python

    class SimpleObs(ObservationBuilder):
        """
        Simplest observation builder. The object returns observation vectors with 5 identical components,
        all equal to the ID of the respective agent.
        """
        def __init__(self):
            self.observation_space = [5]

        def reset(self):
            return

        def get(self, handle):
            observation = handle * np.ones((self.observation_space[0],))
            return observation

We can pass our custom observation builder :code:`SimpleObs` to the :code:`RailEnv` creator as follows:

.. code-block:: python

    env = RailEnv(width=7,
                  height=7,
                  rail_generator=random_rail_generator(),
                  number_of_agents=3,
                  obs_builder_object=SimpleObs())

Anytime :code:`env.reset()` or :code:`env.step()` is called the observation builder will return the custom observation of all agents initialized in the env.
In the next example we want to highlight how you can derive from already implemented observation builders and how to access internal variables of **Flatland**.


Example 2 : Single-agent navigation
--------------

Observation builder objects can also derive existing implementations of classes derived from the ObservationBuilder
base class. For example, it may be useful to derive observations from the TreeObsForRailEnv_ implemented observation
builder. An advantage of this class is that on :code:`reset()`, it pre-computes the length of the shortest paths from all
cells and orientations to the target of each agent, e.g. a distance map for each agent.
In this example we want to exploit these distance maps by implementing and observation builder that shows the current shortest path for each agent as a binary observation vector of length 3, whose components represent the possible directions an agent can take (LEFT, FORWARD, RIGHT). All values of the observation vector are set to :code:`0` except for the shortest direction where it is set to :code:`1`.
Using this observation with highly engineer features indicating the agents shortest path an agent can then learn to take the corresponding action at each time-step, or we could even hardcode the optimal policy. Please do note, however, that this simple strategy fails when multiple agents are present, as each agent would only attempt its greedier solution, which is not usually Pareto-optimal in this context.

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
            # We set max_depth=0 in because we only need to look at the current position of the agent to deside what direction is shortest.
            self.observation_space = [3]

        def reset(self):
            # Recompute the distance map, if the environment has changed.
            super().reset()

        def get(self, handle):
            # Here we acces agent information of the instantiated environment. Any information of the environment can be accessed but not changed!
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
                        min_distances.append(self.distance_map[handle, new_position[0], new_position[1], direction])
                    else:
                        min_distances.append(np.inf)

                observation = [0, 0, 0]
                observation[np.argmin(min_distances)] = 1

            return observation

    env = RailEnv(width=7,
                  height=7,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999, seed=0),
                  number_of_agents=2,
                  obs_builder_object=SingleAgentNavigationObs())

    obs, all_rewards, done, _ = env.step({0: 0, 1: 1})
    for i in range(env.get_num_agents()):
        print(obs[i])

Finally, the following is an example of hard-coded navigation for single agents that achieves optimal single-agent
navigation to target, and show the taken path as an animation.

.. code-block:: python

    env = RailEnv(width=50,
                  height=50,
                  rail_generator=random_rail_generator(),
                  number_of_agents=1,
                  obs_builder_object=SingleAgentNavigationObs())

    obs, all_rewards, done, _ = env.step({0: 0})

    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env(show=True, frames=True, show_observations=False)

    for step in range(100):
        action = np.argmax(obs[0])+1
        obs, all_rewards, done, _ = env.step({0:action})
        print("Rewards: ", all_rewards, "  [done=", done, "]")

        env_renderer.render_env(show=True, frames=True, show_observations=False)
        time.sleep(0.1)



