Flatland
========

![Test Running](https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg)![Test Coverage](https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg "asdff")


![Flatland](https://i.imgur.com/0rnbSLY.gif)

## About Flatland

Flatland is a opensource toolkit for developing and comparing Multi Agent Reinforcement Learning algorithms in little (or ridiculously large !) gridworlds.

The base environment is a two-dimensional grid in which many agents can be placed, and each agent must solve one or more navigational tasks in the grid world. More details about the environment and the problem statement can be found in the [official docs](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/).

This library was developed by [SBB](<https://www.sbb.ch/en/>), [AIcrowd](https://www.aicrowd.com/) and numerous contributors and AIcrowd research fellows from the AIcrowd community. 

This library was developed specifically for the [Flatland Challenge](https://www.aicrowd.com/challenges/flatland-challenge) in which we strongly encourage you to take part in. 

**NOTE This document is best viewed in the official documentation site at** [Flatland-RL Docs](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/)


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

### Install Flatland
#### Stable Release

To install flatland, run this command in your terminal:

```console
$ pip install flatland-rl
```

This is the preferred method to install flatland, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


#### From sources

The sources for flatland can be downloaded from [gitlab](https://gitlab.aicrowd.com/flatland/flatland)

You can clone the public repository:
```console
$ git clone git@gitlab.aicrowd.com:flatland/flatland.git
```

Once you have a copy of the source, you can install it with:

```console
$ python setup.py install
```

### Test installation

Test that the installation works

```console
$ flatland-demo
```



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

## Communication
* [Official Documentation](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/)
* [Discussion Forum](https://discourse.aicrowd.com/c/flatland-challenge)
* [Issue Tracker](https://gitlab.aicrowd.com/flatland/flatland/issues/)


## Contributions
Please follow the [Contribution Guidelines](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/contributing.html) for more details on how you can successfully contribute to the project. We enthusiastically look forward to your contributions.

## Partners
<a href="https://sbb.ch" target="_blank"><img src="https://i.imgur.com/OSCXtde.png" alt="SBB"/></a>
<a href="https://www.aicrowd.com"  target="_blank"><img src="https://avatars1.githubusercontent.com/u/44522764?s=200&v=4" alt="AICROWD"/></a>



