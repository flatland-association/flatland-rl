
Environment Wrappers
====================

..

   We provide various environment wrappers to work with both the rail env and the petting zoo interface.


Background
----------

These wrappers changes certain environment behavior which can help to get better reinforcement learning training.

Supported Inbuilt Wrappers
--------------------------

We provide 2 sample wrappers for ShortestPathAction wrapper and SkipNoChoice wrapper. The wrappers requires many env properties that are only created on environment reset. Hence before using the wrapper, we must reset the rail env. To use the wrappers, simply pass the resetted rail env. Code samples are shown below for each wrapper.

ShortestPathAction Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the ShortestPathAction Wrapper, simply wrap the rail env as follows

.. code-block:: python

   rail_env.reset(random_seed=1)
   rail_env = ShortestPathActionWrapper(rail_env)

The shortest path action wrapper maps the existing action space into 3 actions - Shortest Path (\ ``0``\ ), Next Shortest Path (\ ``1``\ ) and Stop (\ ``2``\ ).  Hence, we must ensure that the predicted action should always be one of these (0, 1 and 2) actions. To route all agents in the shortest path, pass ``0`` as the action.

SkipNoChoice Wrapper
^^^^^^^^^^^^^^^^^^^^

To use the SkipNoChoiceWrapper, simply wrap the rail env as follows

.. code-block:: python

   rail_env.reset(random_seed=1)
   rail_env = SkipNoChoiceCellsWrapper(rail_env, accumulate_skipped_rewards=False, discounting=0.0)
