#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flatland.core.env import RailEnv
#from flatland.core.transitions import GridTransitions
import numpy as np
import random
import os

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



def checkFrozenImage(sFileImage):
    sDirRoot = "."
    sTmpFileImage = sDirRoot + "/images/test/" + sFileImage

    if os.path.exists(sTmpFileImage):
        os.remove(sTmpFileImage)

    plt.savefig(sTmpFileImage)

    bytesFrozenImage = None
    for sDir in [ "/images/", "/images/test/" ]:
        sfPath = sDirRoot + sDir + sFileImage
        bytesImage = plt.imread(sfPath)
        if bytesFrozenImage is None:
            bytesFrozenImage = bytesImage
        else:
            assert (np.sum(np.square(bytesFrozenImage-bytesImage)))


def test_render_env():
    random.seed(100)
    oRail = rail_env_generator.generate_random_rail(10,10)
    type(oRail), len(oRail)
    oEnv = RailEnv(oRail, number_of_agents=2)
    oEnv.reset()
    oRT = rt.RenderTool(oEnv)
    plt.figure(figsize=(10,10))
    oRT.renderEnv()

    checkFrozenImage("basic-env.png")

    plt.figure(figsize=(10,10))
    oRT.renderEnv()
    lVisits = oRT.plotTreeOnRail(
        oEnv.agents_position[0], 
        oEnv.agents_direction[0], 
        nDepth=17)

    checkFrozenImage("env-tree-spatial.png")
    
    plt.figure(figsize=(8,8))
    xyTarg = oRT.env.agents_target[0]
    visitDest = oRT.plotTree(lVisits, xyTarg)

    checkFrozenImage("env-tree-graph.png")


    oFig = plt.figure(figsize=(10,10))
    oRT.renderEnv()
    oRT.plotPath(visitDest)

    checkFrozenImage("env-path.png")


