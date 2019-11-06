Getting Started Tutorial
========================

Overview
--------

Following are three short tutorials to help new users get acquainted with how
to create RailEnvs, how to train simple DQN agents on them, and how to customize
them.

To use flatland in a project:

.. code-block:: python

    import flatland


Simple Example 1 : Basic Usage
------------------------------
The basic usage of RailEnv environments consists in creating a RailEnv object
endowed with a rail generator, that generates new rail networks on each reset,
and an observation generator object, that is supplied with environment-specific
information at each time step and provides a suitable observation vector to the
agents. After the RailEnv environment is created, one need to call reset() on the
environment in order to fully initialize the environment

The simplest rail generators are envs.rail_generators.rail_from_manual_specifications_generator
and envs.rail_generators.random_rail_generator.

The first one accepts a list of lists whose each element is a 2-tuple, whose
entries represent the 'cell_type' (see core.transitions.RailEnvTransitions) and
the desired clockwise rotation of the cell contents (0, 90, 180 or 270 degrees).
For example,

.. code-block:: python

    specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
             [(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
             [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)],
             [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]

    env = RailEnv(width=6,
                  height=4,
                  rail_generator=rail_from_manual_specifications_generator(specs),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2))
    env.reset()

Alternatively, a random environment can be generated (optionally specifying
weights for each cell type to increase or decrease their proportion in the
generated rail networks).

.. code-block:: python

    # Relative weights of each cell type to be used by the random rail generators.
    transition_probability = [1.0,  # empty cell - Case 0
                              1.0,  # Case 1 - straight
                              1.0,  # Case 2 - simple switch
                              0.3,  # Case 3 - diamond drossing
                              0.5,  # Case 4 - single slip
                              0.5,  # Case 5 - double slip
                              0.2,  # Case 6 - symmetrical
                              0.0,  # Case 7 - dead end
                              0.2,  # Case 8 - turn left
                              0.2,  # Case 9 - turn right
                              1.0]  # Case 10 - mirrored switch

    # Example generate a random rail
    env = RailEnv(width=10,
                  height=10,
                  rail_generator=random_rail_generator(
                            cell_type_relative_proportion=transition_probability
                            ),
                  number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2))
    env.reset()

Environments can be rendered using the utils.rendertools utilities, for example:

.. code-block:: python

    env_renderer = RenderTool(env)
    env_renderer.render_env(show=True)


Finally, the environment can be run by supplying the environment step function
with a dictionary of actions whose keys are agents' handles (returned by
env.get_agent_handles() ) and the corresponding values the selected actions.
For example, for a 2-agents environment:

.. code-block:: python

    handles = env.get_agent_handles()
    action_dict = {handles[0]:0, handles[1]:0}
    obs, all_rewards, done, _ = env.step(action_dict)

where 'obs', 'all_rewards', and 'done' are also dictionary indexed by the agents'
handles, whose values correspond to the relevant observations, rewards and terminal
status for each agent. Further, the 'dones' dictionary returns an extra key
'__all__' that is set to True after all agents have reached their goals.


In the specific case a TreeObsForRailEnv observation builder is used, it is
possible to print a representation of the returned observations with the
following code. Also, tree observation data is displayed by RenderTool by default.

.. code-block:: python

    for i in range(env.get_num_agents()):
        env.obs_builder.util_print_obs_subtree(
                tree=obs[i],
                )

The complete code for this part of the Getting Started guide can be found in

* `examples/simple_example_1.py <https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/simple_example_1.py>`_
* `examples/simple_example_2.py <https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/simple_example_2.py>`_


Part 2 : Training a Simple an Agent on Flatland
---------------------------------------------------------

This is a brief tutorial on how to train an agent on Flatland.
Here we use a simple random agent to illustrate the process on how to interact with the environment.
The corresponding code can be found in examples/training_example.py and in the baselines repository
you find a tutorial to train a `DQN <https://arxiv.org/abs/1312.5602>`_ agent to solve the navigation task.

We start by importing the necessary Flatland libraries

.. code-block:: python

    from flatland.envs.rail_generators import complex_rail_generator
    from flatland.envs.schedule_generators import complex_schedule_generator
    from flatland.envs.rail_env import RailEnv

The complex_rail_generator is used in order to guarantee feasible railway network configurations for training.
Next we configure the difficulty of our task by modifying the complex_rail_generator parameters.

.. code-block:: python

    env = RailEnv(  width=15,
                    height=15,
                    rail_generator=complex_rail_generator(
                                        nr_start_goal=10,
                                        nr_extra=10,
                                        min_dist=10,
                                        max_dist=99999,
                                        seed=1),
                    number_of_agents=5)
    env.reset()

The difficulty of a railway network depends on the dimensions (`width` x `height`) and the number of agents in the network.
By varying the number of start and goal connections (nr_start_goal) and the number of extra railway elements added (nr_extra)
the number of alternative paths of each agents can be modified. The more possible paths an agent has to reach its target the easier the task becomes.
Here we don't specify any observation builder but rather use the standard tree observation. If you would like to use a custom obervation please follow
the instructions in the next tutorial.
Feel free to vary these parameters to see how your own agent holds up on different setting. The evalutation set of railway configurations will
cover the whole spectrum from easy to complex tasks.

Once we are set with the environment we can load our preferred agent from either RLlib or any other ressource. Here we use a random agent to illustrate the code.

.. code-block:: python

    agent = RandomAgent(state_size, action_size)

We start every trial by resetting the environment

.. code-block:: python

    obs, info = env.reset()

Which provides the initial observation for all agents (obs = array of all observations).
In order for the environment to step forward in time we need a dictionar of actions for all active agents.

.. code-block:: python

        for handle in range(env.get_num_agents()):
            action = agent.act(obs[handle])
            action_dict.update({handle: action})

This dictionary is then passed to the environment which checks the validity of all actions and update the environment state.

.. code-block:: python

    next_obs, all_rewards, done, _ = env.step(action_dict)

The environment returns an array of new observations, reward dictionary for all agents as well as a flag for which agents are done.
This information can be used to update the policy of your agent and if done['__all__'] == True the episode terminates.

The full source code of this example can be found in `examples/training_example.py <https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/training_example.py>`_.
