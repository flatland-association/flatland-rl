import cProfile
import runpy
import sys
from io import StringIO

import importlib_resources
import pkg_resources
from importlib_resources import path

from benchmarks.benchmark_utils import swap_attr


def profile(resource, entry):
    with path(resource, entry) as file_in:
        # TODO remove input() from examples
        print("*****************************************************************")
        print("Profiling {}".format(entry))
        print("*****************************************************************")
        with swap_attr(sys, "stdin", StringIO("q")):
            global my_func

            def my_func(): runpy.run_path(file_in, run_name="__main__")

            cProfile.run('my_func()', sort='time')


for entry in [entry for entry in importlib_resources.contents('examples') if
              not pkg_resources.resource_isdir('examples', entry)
              and entry.endswith(".py")
              and '__init__' not in entry
              and 'demo.py' not in entry
              ]:
    profile('examples', entry)
