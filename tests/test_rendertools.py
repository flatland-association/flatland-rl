#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flatland.core.env import RailEnv
#from flatland.core.transitions import GridTransitions
import numpy as np
import random

from recordtype import recordtype

import numpy as np
from numpy import array
import xarray as xr
import matplotlib.pyplot as plt

from flatland.core.transitions import RailEnvTransitions
#import flatland.core.env
from flatland.utils import rail_env_generator
from flatland.core.env import RailEnv
import flatland.utils.rendertools as rt




"""Tests for `flatland` package."""





def test_render_env():
    random.seed(100)
    oRail = rail_env_generator.generate_random_rail(10,10)
    type(oRail), len(oRail)
    oEnv = RailEnv(oRail, number_of_agents=2)
    oEnv.reset()
    oRT = rt.RenderTool(oEnv)
    plt.figure(figsize=(10,10))
    oRT.renderEnv()
