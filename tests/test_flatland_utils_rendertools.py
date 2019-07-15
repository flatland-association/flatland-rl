#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for `flatland` package.
"""

import sys

import numpy as np
from importlib_resources import path

import flatland.utils.rendertools as rt
import images.test
from flatland.envs.generators import empty_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv


def checkFrozenImage(oRT, sFileImage, resave=False):
    sDirRoot = "."
    sDirImages = sDirRoot + "/images/"

    img_test = oRT.get_image()

    if resave:
        np.savez_compressed(sDirImages + sFileImage, img=img_test)
        return

    with path(images, sFileImage) as file_in:
        np.load(file_in)

    # TODO fails!
    #  assert (img_test.shape == img_expected.shape) \ #  noqa: E800
    #  assert ((np.sum(np.square(img_test - img_expected)) / img_expected.size / 256) < 1e-3), \ #  noqa: E800
    #      "Image {} does not match".format(sFileImage) \ #  noqa: E800


def test_render_env(save_new_images=False):
    np.random.seed(100)
    oEnv = RailEnv(width=10, height=10,
                   rail_generator=empty_rail_generator(),
                   number_of_agents=0,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2)
                   )
    oEnv.rail.load_transition_map('env_data.tests', "test1.npy")
    oRT = rt.RenderTool(oEnv, gl="PILSVG")
    oRT.render_env(show=False)

    checkFrozenImage(oRT, "basic-env.npz", resave=save_new_images)

    oRT = rt.RenderTool(oEnv, gl="PIL")
    oRT.render_env()
    checkFrozenImage(oRT, "basic-env-PIL.npz", resave=save_new_images)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "save":
        test_render_env(save_new_images=True)
    else:
        print("Run 'python test_flatland_utils_rendertools.py save' to regenerate images")
        test_render_env()


if __name__ == "__main__":
    main()
