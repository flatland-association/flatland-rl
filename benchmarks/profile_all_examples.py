import cProfile
import runpy
import sys
from io import StringIO
from test.support import swap_attr

import importlib_resources
import pkg_resources
from importlib_resources import path


def profile(resource, entry):
    with path(resource, entry) as file_in:
        # we use the test package, which is meant for internal use by Python only internal and
        # Any use of this package outside of Python’s standard library is discouraged as code (..)
        # can change or be removed without notice between releases of Python.
        # https://docs.python.org/3/library/test.html
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
