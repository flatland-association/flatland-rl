Flatland
========

![Test Running](https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg)![Test Coverage](https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg "asdff")


![Flatland](https://i.imgur.com/0rnbSLY.gif)

## About Flatland

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


![Flatland](https://i.imgur.com/pucB84T.gif)
![Flatland](https://i.imgur.com/xgWGRse.gif)

**NOTE This document is best viewed in the official documentation site at** `Flatland-RL Docs <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/>`_


## Installation
### Installation Prerequistes

* Install [Anaconda](https://www.anaconda.com/distribution/) by following the instructions [here](https://www.anaconda.com/distribution/).
* Create a new conda environment:

```console
$ conda create python=3.6 --name flatland-rl
$ conda activate flatland-rl
```

* Install the necessary dependencies

```console
$ conda install -c conda-forge cairosvg pycairo
$ conda install -c anaconda tk  
```

### Stable release

To install flatland, run this command in your terminal:

```console
$ pip install flatland-rl
```

This is the preferred method to install flatland, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


### From sources

The sources for flatland can be downloaded from the `Gitlab repo`_.

You can clone the public repository:
```console
$ git clone git@gitlab.aicrowd.com:flatland/flatland.git
```

Once you have a copy of the source, you can install it with:

```console
$ python setup.py install
```

.. _Gitlab repo: https://gitlab.aicrowd.com/flatland/flatland


### Jupyter Canvas Widget
If you work with jupyter notebook you need to install the Jupyer Canvas Widget. To install the Jupyter Canvas Widget read also
https://github.com/Who8MyLunch/Jupyter_Canvas_Widget#installation

## Basic Usage

Basic usage of the RailEnv environment used by the Flatland Challenge


```python
import numpy as np
import time
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
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
            schedule_generator=complex_schedule_generator(),
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
```

and **ideally** you should see something along the lines of

![Flatland](https://i.imgur.com/VrTQVeM.gif)

Best of Luck !!

## Contributions
Please follow the [Contribution Guidelines](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/contributing.html) for more details on how you can successfully contribute to the project. We enthusiastically look forward to your contributions.

## Partners
<a href="https://sbb.ch" target="_blank"><img src="https://i.imgur.com/OSCXtde.png" alt="SBB"/></a>
<a href="https://www.aicrowd.com"  target="_blank"><img src="https://avatars1.githubusercontent.com/u/44522764?s=200&v=4" alt="AICROWD"/></a>



