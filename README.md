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
[https://github.com/Who8MyLunch/Jupyter_Canvas_Widget#installation]([https://github.com/Who8MyLunch/Jupyter_Canvas_Widget#installation)

## Basic Usage

Basic usage of the RailEnv environment used by the Flatland Challenge (also see [Example](https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/introduction_flatland_2_1.py))


```python
from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

width = 100  # With of map
height = 100  # Height of map
nr_trains = 50  # Number of trains that have an assigned task in the env
cities_in_map = 20  # Number of cities where agents can start or end
seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 6  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )

# The schedule generator can make very basic schedules with a start point, end point and a speed profile for each agent.
# The speed profiles can be adjusted directly as well as shown later on. We start by introducing a statistical
# distribution of speed profiles

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)

# We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
# during an episode.

stochastic_data = {'prop_malfunction': 0.3,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }

# Custom observation builder without predictor
observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor, uncomment line below if you want to try this one
# observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
              obs_builder_object=observation_builder,
              remove_agents_at_target=True  # Removes agents at the end of their journey to make space for others
              )

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=1080,  # Adjust these parameters to fit your resolution
                          screen_width=1920)  # Adjust these parameters to fit your resolution


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
    obs, all_rewards, done, info = env.step(_action)
    print("Rewards: {}, [done={}]".format( all_rewards, done))
    env_renderer.render_env(show=True, frames=False, show_observations=False)
    time.sleep(0.3)
```

and **ideally** you should see something along the lines of

![Flatland](https://i.imgur.com/Pc9aH4P.gif)

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



