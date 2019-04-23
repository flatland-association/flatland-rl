#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for `flatland` package.
"""

from flatland.envs.rail_env import RailEnv, random_rail_generator
import numpy as np
import os

import matplotlib.pyplot as plt

import flatland.utils.rendertools as rt
from flatland.core.env_observation_builder import GlobalObsForRailEnv


def checkFrozenImage(sFileImage):
    sDirRoot = "."
    sTmpFileImage = sDirRoot + "/images/test/" + sFileImage

    if os.path.exists(sTmpFileImage):
        os.remove(sTmpFileImage)

    plt.savefig(sTmpFileImage)

    bytesFrozenImage = None
    for sDir in ["/images/", "/images/test/"]:
        sfPath = sDirRoot + sDir + sFileImage
        bytesImage = plt.imread(sfPath)
        if bytesFrozenImage is None:
            bytesFrozenImage = bytesImage
        else:
            assert(bytesFrozenImage.shape == bytesImage.shape)
            assert((np.sum(np.square(bytesFrozenImage-bytesImage)) / bytesFrozenImage.size) < 1e-3)


def test_render_env():
    # random.seed(100)
    np.random.seed(100)
    oEnv = RailEnv(width=10, height=10,
                   rail_generator=random_rail_generator(),
                   number_of_agents=2,
                   obs_builder_object=GlobalObsForRailEnv())
    oEnv.reset()
    oRT = rt.RenderTool(oEnv)
    plt.figure(figsize=(10, 10))
    oRT.renderEnv()

    checkFrozenImage("basic-env.png")

    plt.figure(figsize=(10, 10))
    oRT.renderEnv()

    lVisits = oRT.getTreeFromRail(
        oEnv.agents_position[0],
        oEnv.agents_direction[0],
        nDepth=17, bPlot=True)

    checkFrozenImage("env-tree-spatial.png")

    plt.figure(figsize=(8, 8))
    xyTarg = oRT.env.agents_target[0]
    visitDest = oRT.plotTree(lVisits, xyTarg)

    checkFrozenImage("env-tree-graph.png")

    plt.figure(figsize=(10, 10))
    oRT.renderEnv()
    oRT.plotPath(visitDest)

    checkFrozenImage("env-path.png")
