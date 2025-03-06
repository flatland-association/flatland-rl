import cProfile
import os
import runpy
import sys
import traceback
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
        profiling_output_folder = os.environ.get('PROFILING_OUTPUT_FOLDER', None)
        outfile = None
        if profiling_output_folder:
            outfile = os.path.join(profiling_output_folder, f"{entry}.prof")

        with swap_attr(sys, "stdin", StringIO("q")):
            global my_func

            def my_func():
                runpy.run_path(file_in, run_name="__main__", init_globals={
                    'argv': ['--sleep-for-animation=False', '--do_rendering=False']
                })

            try:
                cProfile.run('my_func()', sort='time', filename=outfile)
            except:
                print("cProfile failed:")
                traceback.print_exc()
            print("cProfile done.")

def profile_all_examples():
    for entry in [entry for entry in importlib_resources.contents('examples') if
                  not pkg_resources.resource_isdir('examples', entry)
                  and entry.endswith(".py")
                  and '__init__' not in entry
                  and 'DELETE' not in entry
                  ]:
        profile('examples', entry)


if __name__ == '__main__':
    profile_all_examples()
