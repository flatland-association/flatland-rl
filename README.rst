========
Flatland
========



.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Running
.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Coverage

.. image:: https://i.imgur.com/rKGEmsk.gif
  :align: center
  :width: 400px

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


.. image:: https://i.imgur.com/pucB84T.gif
  :align: center
  :width: 600px

.. image:: https://i.imgur.com/xgWGRse.gif
  :align: center
  :width: 600px


Getting Started
===============

Online Docs
------------

The documentation for the latest code on the master branch is found at : `flatland-rl-docs <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/>`_ 

The documentation includes a few tutorials at : `Getting Started <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/gettingstarted.html>`_

Run Notebooks with Examples with one Click
------------------------------------------
Under getting_started, there are two scripts ::

    getting_started/run_notebooks.bat
    getting_started/run_notebooks.sh

They require git and Python>=3.6 installed with venv (python3-venv has to be installed under Linux).
They create a virtual environment, install Flatland and all dependencies into into and start they Jupyter notebooks in your browser.


Generate Docs
--------------

The docs have a lot more details about how to interact with this codebase. ::

    git clone git@gitlab.aicrowd.com:flatland/flatland.git
    cd flatland
    pip install -r requirements_dev.txt

* On, Linux and macOS ::

    make docs

* On, Windows ::

    python setup.py develop (or)
    python setup.py install
    python make_docs.py


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
    

Jupyter Canvas Widget
---------------------
If you work with jupyter notebook you need to install the Jupyer Canvas Widget. To install the Jupyter Canvas Widget read also
https://github.com/Who8MyLunch/Jupyter_Canvas_Widget#installation


Basic Usage
============

Basic usage of the RailEnv environment used by the Flatland Challenge

.. code-block:: python

    import numpy as np
    import time
    from flatland.envs.generators import complex_rail_generator
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.rendertools import RenderTool
    
    env = RailEnv(
                width=7,
                height=7,
                rail_generator=complex_rail_generator(
                                        nr_start_goal=10,
                                        nr_extra=1,
                                        min_dist=8,
                                        max_dist=99999,
                                        seed=0),
                number_of_agents=2)
    
    env_renderer = RenderTool(env, gl="PILSVG")
    
    for step in range(100):
        obs, all_rewards, done, _ = env.step(
                                {
                                    0:np.random.randint(0, 5),
                                    1:np.random.randint(0, 5)
                                })
        print("Rewards: {}, [done={}]".format( all_rewards, done)
        env_renderer.renderEnv(show=True, frames=False, show_observations=False)
        time.sleep(0.3)


Authors
============

* Sharada Mohanty <mohanty@aicrowd.com>
* Giacomo Spigler <giacomo.spigler@gmail.com>
* Mattias Ljungström
* Jeremy Watson
* Erik Nygren <erik.nygren@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Christian Eichenberger <christian.markus.eichenberger@sbb.ch>
* Guillaume Mollard <guillaume.mollard2@gmail.com>


Acknowledgements
====================
* Vaibhav Agrawal <theinfamouswayne@gmail.com>
* Anurag Ghosh  
