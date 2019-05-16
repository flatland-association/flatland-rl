#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for `flatland` package.
"""

from flatland.envs.rail_env import RailEnv, random_rail_generator
import numpy as np
#<<<<<<< HEAD
#=======
# import os
#>>>>>>> dc2fa1ee0244b15c76d89ab768c5e1bbd2716147
import sys

import matplotlib.pyplot as plt

import flatland.utils.rendertools as rt
from flatland.envs.observations import TreeObsForRailEnv


def checkFrozenImage(oRT, sFileImage, resave=False):
    sDirRoot = "."
    sDirImages = sDirRoot + "/images/"

    img_test = oRT.getImage()

    if resave:
        np.savez_compressed(sDirImages + sFileImage, img=img_test)
        return

    # this is now just for convenience - the file is not read back
    np.savez_compressed(sDirImages + "test/" + sFileImage, img=img_test)

    image_store = np.load(sDirImages + sFileImage)
    img_expected = image_store["img"]

    assert (img_test.shape == img_expected.shape)
    assert ((np.sum(np.square(img_test - img_expected)) / img_expected.size / 256) < 1e-3), \
        "Image {} does not match".format(sFileImage)


def test_render_env(save_new_images=False):
    # random.seed(100)
    np.random.seed(100)
    oEnv = RailEnv(width=10, height=10,
                   rail_generator=random_rail_generator(),
                   number_of_agents=0,
                   # obs_builder_object=GlobalObsForRailEnv())
                   obs_builder_object=TreeObsForRailEnv(max_depth=2)
                   )
    sfTestEnv = "env-data/tests/test1.npy"
    oEnv.rail.load_transition_map(sfTestEnv)
    oRT = rt.RenderTool(oEnv)
    oRT.renderEnv()

    checkFrozenImage(oRT, "basic-env.npz", resave=save_new_images)

    oRT = rt.RenderTool(oEnv, gl="PIL")
    oRT.renderEnv()
    checkFrozenImage(oRT, "basic-env-PIL.npz", resave=save_new_images)

    # disable the tree / observation tests until env-agent save/load is available
    if False:
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


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "save":
        test_render_env(save_new_images=True)
    else:
        print("Run 'python test_rendertools.py save' to regenerate images")


if __name__ == "__main__":
    main()
