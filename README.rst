========
Flatland
========



.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Running
     
.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Coverage

'   

.. image:: https://i.imgur.com/0rnbSLY.gif
  :width: 800
  :align: center

Flatland is a opensource toolkit for developing and comparing Multi Agent Reinforcement Learning algorithms in little (or ridiculously large !) gridworlds.
The base environment is a two-dimensional grid in which many agents can be placed, and each agent must solve one or more navigational tasks in the grid world. More details about the environment and the problem statement can be found in the `official docs <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/>`_.

This library was developed by `SBB <https://www.sbb.ch/en/>`_ , `AIcrowd <https://www.aicrowd.com/>`_ and numerous contributors and AIcrowd research fellows from the AIcrowd community. 

This library was developed specifically for the `Flatland Challenge <https://www.aicrowd.com/challenges/flatland-challenge>`_ in which we strongly encourage you to take part in. 


**NOTE This document is best viewed in the official documentation site at** `Flatland-RL Docs <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/readme.html>`_

Contents
===========
* `Official Documentation <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/readme.html>`_
* `About Flatland <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/about_flatland.html>`_
* `Installation <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/installation.html>`_
* `Getting Started <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/gettingstarted.html>`_
* `Frequently Asked Questions <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/FAQ.html>`_
* `Code Docs <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/modules.html>`_
* `Contributing Guidelines <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/contributing.html>`_
* `Discussion Forum <https://discourse.aicrowd.com/c/flatland-challenge>`_
* `Issue Tracker <https://gitlab.aicrowd.com/flatland/flatland/issues/>`_

Quick Start
===========

* Install `Anaconda <https://www.anaconda.com/distribution/>`_ by following the instructions `here <https://www.anaconda.com/distribution/>`_
* Install the dependencies and the library
 
.. code-block:: console

    $ conda create python=3.6 --name flatland-rl
    $ conda activate flatland-rl
    $ conda install -c conda-forge cairosvg pycairo
    $ conda install -c anaconda tk  
    $ pip install flatland-rl

* Test that the installation works

.. code-block:: console

    $ flatland-demo


Basic Usage
============

Basic usage of the RailEnv environment used by the Flatland Challenge

.. code-block:: python

    import numpy as np
    import time
    from flatland.envs.generators import complex_rail_generator
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.rendertools import RenderTool

    NUMBER_OF_AGENTS = 10
    env = RailEnv(
                width=20,
                height=20,
                rail_generator=complex_rail_generator(
                                        nr_start_goal=10,
                                        nr_extra=1,
                                        min_dist=8,
                                        max_dist=99999,
                                        seed=0),
                number_of_agents=NUMBER_OF_AGENTS)

    env_renderer = RenderTool(env)

    def my_controller():
        """
        You are supposed to write this controller
        """
        _action = {}
        for _idx in range(NUMBER_OF_AGENTS):
            _action[_idx] = np.random.randint(0, 5)
        return _action

    for step in range(100):

        _action = my_controller()
        obs, all_rewards, done, _ = env.step(_action)
        print("Rewards: {}, [done={}]".format( all_rewards, done))
        env_renderer.render_env(show=True, frames=False, show_observations=False)
        time.sleep(0.3)

and **ideally** you should see something along the lines of 

.. image:: https://i.imgur.com/VrTQVeM.gif
  :align: center
  :width: 600px

Best of Luck !!

Contributions
=============
Flatland is an opensource project, and we very much value all and any contributions you make towards the project.
Please follow the `Contribution Guidelines <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/contributing.html>`_ for more details on how you can successfully contribute to the project. We enthusiastically look forward to your contributions.

Partners 
============
.. image:: https://i.imgur.com/OSCXtde.png
   :target: https://sbb.ch
.. image:: https://avatars1.githubusercontent.com/u/44522764?s=200&v=4
   :target: https://www.aicrowd.com


Authors
============

* Christian Eichenberger <christian.markus.eichenberger@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Mattias Ljungström
* Sharada Mohanty <mohanty@aicrowd.com>
* Guillaume Mollard <guillaume.mollard2@gmail.com>
* Erik Nygren <erik.nygren@sbb.ch>
* Giacomo Spigler <giacomo.spigler@gmail.com>
* Jeremy Watson


Acknowledgements
====================
* Vaibhav Agrawal <theinfamouswayne@gmail.com>
* Anurag Ghosh  
