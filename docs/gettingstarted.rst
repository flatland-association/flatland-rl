=====
Getting Started
=====

Overview
--------------

Following are three short tutorials to help new users get acquainted with how 
to create RailEnvs, how to train simple DQN agents on them, and how to customize 
them.

To use flatland in a project:

.. code-block:: python

    import flatland


Part 1 : Basic Usage
--------------

The basic usage of RailEnv environments consists in creating a RailEnv object 
endowed with a rail generator, that generates new rail networks on each reset, 
and an observation generator object, that is supplied with environment-specific 
information at each time step and provides a suitable observation vector to the 
agents.

The simplest rail generators are envs.generators.rail_from_manual_specifications_generator 
and envs.generators.random_rail_generator.

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
                  rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
                  number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2))

Environments can be rendered using the utils.rendertools utilities, for example:

.. code-block:: python

    env_renderer = RenderTool(env, gl="QT")
    env_renderer.renderEnv(show=True)


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
        env.obs_builder.util_print_obs_subtree(tree=obs[i], num_features_per_node=5)

The complete code for this part of the Getting Started guide can be found in 
examples/simple_example_1.py, examples/simple_example_2.py and 
examples/simple_example_3.py



Part 2 : Training a Simple DQN Agent
--------------



Part 3 : Customizing Observations and Level Generators
--------------



