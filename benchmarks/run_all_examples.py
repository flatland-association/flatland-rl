import runpy
import sys
import traceback
from io import StringIO

import importlib_resources
from importlib_resources import path

from benchmarks.benchmark_utils import swap_attr


def run_all_examples():
    print("run_all_examples.py")
    error_log_examples = {}
    for entry in [entry for entry in importlib_resources.files('examples').iterdir() if
                  not importlib_resources.files('examples').joinpath(str(entry)).is_dir()
                  and str(entry).endswith(".py")
                  and '__init__' not in str(entry)
                  and 'DELETE' not in str(entry)
                  and 'flatland_performance_profiling' not in str(entry)
                  ]:
        with path('examples', entry.name) as file_in:
            print("")
            print("")

            print("")
            print("*****************************************************************")
            print("Running {}".format(entry))
            print("*****************************************************************")

            with swap_attr(sys, "stdin", StringIO("q")):
                try:
                    runpy.run_path(file_in, run_name="__main__", init_globals={
                        'argv': ['--sleep-for-animation=False', '--do_rendering=False']
                    })
                except Exception as e:
                    print(e)
                    error_log_examples.update({file_in: e})
                except:
                    print("runpy failed:")
                    traceback.print_exc()
                print("runpy done.")
            print("Done with {}".format(entry))
    if len(error_log_examples.keys()) > 0:
        print("*****************************************************************")
        print("Error log:")
        print("*****************************************************************")
        print(error_log_examples)
        print("*****************************************************************")
        raise Exception("Some examples failed.")
    else:
        print("*****************************************************************")
        print("All examples executed - no error.")
        print("*****************************************************************")


if __name__ == '__main__':
    run_all_examples()
