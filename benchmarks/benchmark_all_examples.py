import os
import runpy
import sys
from io import StringIO
from time import sleep

import importlib_resources
import pkg_resources
from benchmarker import Benchmarker
from importlib_resources import path

from benchmarks.benchmark_utils import swap_attr


def benchmark_all_examples():
    for entry in [entry for entry in importlib_resources.contents('examples') if
                  not pkg_resources.resource_isdir('examples', entry)
                  and entry.endswith(".py")
                  and '__init__' not in entry
                  and 'DELETE' not in entry
                  ]:
        print("*****************************************************************")
        print("Benchmarking {}".format(entry))
        print("*****************************************************************")

        benchmarks_output_folder = os.environ.get('BENCHMARKS_OUTPUT_FOLDER', None)
        outfile = None
        if benchmarks_output_folder:
            outfile = os.path.join(benchmarks_output_folder, f"{entry}.json")
        print(f"outfile={outfile}")

        with path('examples', entry) as file_in:
            with Benchmarker(cycle=20, extra=1, outfile=outfile) as bench:
                @bench(entry)
                def _(_):
                    # prevent Benchmarker from doing "ZeroDivisionError: float division by zero:
                    #    ratio = base_time / real_time"
                    sleep(0.001)
                    # In order to pipe input into examples that have input(),
                    # we use the test package, which is meant for internal use by Python only internal and
                    # Any use of this package outside of Python’s standard library is discouraged as code (..)
                    # can change or be removed without notice between releases of Python.
                    # https://docs.python.org/3/library/test.html
                    # TODO remove input() from examples?
                    with swap_attr(sys, "stdin", StringIO("q")):
                        runpy.run_path(file_in, run_name="__main__", init_globals={
                            'argv': ['--sleep-for-animation=False', '--do_rendering=False']
                        })


if __name__ == '__main__':
    benchmark_all_examples()
