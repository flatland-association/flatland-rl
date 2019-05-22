========
Flatland
========



.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Running
.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Coverage



Flatland is a toolkit for developing and comparing multi agent reinforcement learning algorithms on grids.
The base environment is a two-dimensional grid in which many agents can be placed. Each agent must solve one or more tasks in the grid world.
In general, agents can freely navigate from cell to cell. However, cell-to-cell navigation can be restricted by transition maps.
Each cell can hold an own transition map. By default, the cell doesn't have any restriction of movement defined in its transition map. So, the agent can freely move to any neighbor cell.
With other world the agent can move to all eight neighbor cells (go up and left, go up, go up and right, go right, go down and right, go down, go down and left, go left).

The general purpose of the implementation allows to implement any kind of 2D gird based environments.
It can be used for many learning task where a two-dimensional grid could be the base of the environment.

Flatland delivers a python implementation which can be easily extended. And it provides different baselines for different environments.
Each environment enables an interesting task to solve. For example, the mutli-agent navigation task for railway train dispatching is a very exciting topic.
It can be easily extended or adapted to the airplane landing problem. This can further be the basic implementation for many other tasks in transportation and logistics.

Mapping a railway infrastructure into a grid world is an excellent example showing how the movement can of an agent can be easily restricted with the help of the cell's transition maps.
As trains can normally not run backwards and they have to follow rails the transition for one cell ot the other depends also the train's orientation.
Trains can only change the traveling path at switches. There are two variants of switches. The first kind of switch is the splitting "switch", where trains can change rails and in consequence they can change the traveling path.
The second kind of switch is the fusion switch, where train can change order. That means two rails come together. Thus, the navigation behavior of a train is very restricted.
The railway planning problem where many agents share same infrastructure is a very complex problem. If trains cannot change traveling path, the underlaying problem will be an ordering problem. Even the ordering
problem is very hard to solve.
Furthermore, trains have a departing location where they cannot depart earlier than a committed time. Then they must arrive at destination not later than the second committed time. This makes the whole planning problem
still more complicated. In such a complex environment cooperation is essential. Thus, agents must learn to cooperate in a way that all trains (agents) arrive on time.


Getting Started
===============

Online Docs
------------

The documentation for the latest code on the master branch is found at  `http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/ <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/>`_ 



Generate Docs
--------------

The docs have a lot more details about how to interact with this codebase.  

**TODO**: Mohanty to add atleast a neat outline herefor the contents to the docs here ::

    git clone git@gitlab.aicrowd.com:flatland/flatland.git
    cd flatland
    pip install -r requirements_dev.txt

* On, Linux and macOS ::

    make docs


* On, Windows ::

    python setup.py develop (or)
    python setup.py install
    python make_docs.py


Features
--------

TODO


Installation
============

Stable Release
--------------

To install flatland, run this command in your terminal ::

    pip install flatland-rl

This is the preferred method to install flatland, as it will always install the most recent stable release.

If you don’t have `pip <https://pip.pypa.io/en/stable/>`_ installed, this `Python installation guide <https://docs.python-guide.org/starting/installation/>`_ can guide you through the process.


From Sources
------------
The sources for flatland can be downloaded from the `Gitlab repo <https://gitlab.aicrowd.com/flatland/flatland>`_.

You can clone the public repository ::

    $ git clone git@gitlab.aicrowd.com:flatland/flatland.git

Once you have a copy of the source, you can install it with ::

    $ python setup.py install
    
    
Usage
=====
To use flatland in a project ::
    
    import flatland
    
flatland
========
TODO: explain the interface here

Module Dependencies
===================
.. image:: flatland.svg


Authors
--------
* Sharada Mohanty <mohanty@aicrowd.com>
* Giacomo Spigler <giacomo.spigler@gmail.com>
* Mattias Ljungström
* Jeremy Watson
* Erik Nygren <erik.nygren@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Vaibhav Agrawal <theinfamouswayne@gmail.com>
* Christian Eichenberger <christian.markus.eichenberger@sbb.ch>


<please fill yourself in>
