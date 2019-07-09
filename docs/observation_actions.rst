=============================
Observation and Action Spaces
=============================
This is an introduction to the three standard observations and the action space of **Flatland**.

Action Space
============
Flatland is a railway simulation. Thus the actions of an agent are strongly limited to the railway network. This means that in many cases not all actions are valid.
The possible actions of an agent are

- 0 **Do Nothing**:  If the agent is moving it continues moving, if it is stopped it stays stopped
- 1 **Deviate Left**: If the agent is at a switch with a transition to its left, the agent will chose th eleft path. Otherwise the action has no effect. If the agent is stopped, this action will start agent movement again if allowed by the transitions.
- 2 **Go Forward**: This action will start the agent when stopped. This will move the agent forward and chose the go straight direction at switches.
- 3 **Deviate Right**: Exactly the same as deviate left but for right turns.
- 4 **Stop**: This action causes the agent to stop.

Observation Spaces
==================
In the **Flatland** environment we have included three basic observations to get started. The figure below illustrates the observation range of the different basic observation: Global, Local Grid and Local Tree.
.. image:: WGfFtP7.png
   :target: https://i.imgur.com/WGfFtP7.png
   
Global Observation
------------------

Local Grid Observation
----------------------

Tree Observation
----------------
