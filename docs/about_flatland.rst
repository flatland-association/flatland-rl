About Flatland
==============

.. image:: https://i.imgur.com/rKGEmsk.gif
  :align: center



Flatland is a toolkit for developing and comparing multi agent reinforcement learning algorithms on grids.
The base environment is a two-dimensional grid in which many agents can be placed. Each agent must solve one or more tasks in the grid world.
In general, agents can freely navigate from cell to cell. However, cell-to-cell navigation can be restricted by transition maps.
Each cell can hold an own transition map. By default, each cell has a default transition map defined which allows all transitions to its
eight neighbor cells (go up and left, go up, go up and right, go right, go down and right, go down, go down and left, go left).
So, the agents can freely move from cell to cell.

The general purpose of the implementation allows to implement any kind of two-dimensional gird based environments.
It can be used for many learning task where a two-dimensional grid could be the base of the environment.

Flatland delivers a python implementation which can be easily extended. And it provides different baselines for different environments.
Each environment enables an interesting task to solve. For example, the mutli-agent navigation task for railway train dispatching is a very exciting topic.
It can be easily extended or adapted to the airplane landing problem. This can further be the basic implementation for many other tasks in transportation and logistics.

Mapping a railway infrastructure into a grid world is an excellent example showing how the movement of an agent must be restricted.
As trains can normally not run backwards and they have to follow rails the transition for one cell to the other depends also on train's orientation, respectively on train's travel direction.
Trains can only change the traveling path at switches. There are two variants of switches. The first kind of switch is the splitting "switch", where trains can change rails and in consequence they can change the traveling path.
The second kind of switch is the fusion switch, where train can change the sequence. That means two rails come together. Thus, the navigation behavior of a train is very restricted.
The railway planning problem where many agents share same infrastructure is a very complex problem.

Furthermore, trains have a departing location where they cannot depart earlier than the committed departure time.
Then they must arrive at destination not later than the committed arrival time. This makes the whole planning problem
very complex. In such a complex environment cooperation is essential. Thus, agents must learn to cooperate in a way that all trains (agents) arrive on time.

This library was developed by `SBB <https://www.sbb.ch/en/>`_ , `AIcrowd <https://www.aicrowd.com/>`_ and numerous contributors and AIcrowd research fellows from the AIcrowd community. 

This library was developed specifically for the `Flatland Challenge <https://www.aicrowd.com/challenges/flatland-challenge>`_ in which we strongly encourage you to take part in. 


.. image:: https://i.imgur.com/pucB84T.gif
  :align: center
  :width: 600px

.. image:: https://i.imgur.com/xgWGRse.gif
  :align: center
  :width: 600px