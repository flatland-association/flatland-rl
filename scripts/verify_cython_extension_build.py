#!/usr/bin/env python
"""
Verify whether the Cython-optional extensions declared in pyproject.toml's `[tool.setuptools] ext-modules`
(flatland.envs.step_utils.state_machine/states, flatland.envs.rail_env_shortest_paths) were compiled into a
built wheel, or fell back to plain Python. Also supports checking a built sdist, which - being source-only by
construction - is expected to always contain just the plain `.py` sources (never `.so`/`.pyd`); this is a
regression guard for the published-package build (see tox.ini's [testenv:build], which builds sdist-only
specifically to avoid the wheel-tag problem: a wheel's platform/ABI tag depends on whether Cython ever
attempted to cythonize a module, not on whether the final compile succeeded, so a wheel built with a merely
*missing compiler* still ends up platform-tagged rather than universal).

Used by tox.ini's verify-build-no-gcc/verify-cython-build[-no-isolation]/build envs.
"""
import argparse
import glob
import sys
import tarfile
import zipfile

MODULES = ["flatland/envs/step_utils/state_machine", "flatland/envs/step_utils/states", "flatland/envs/rail_env_shortest_paths"]


def _wheel_names(path):
    return zipfile.ZipFile(path).namelist()


def _sdist_names(path):
    # sdist entries are prefixed with "<pkg>-<version>/" - strip it so names line up with MODULES
    with tarfile.open(path) as tf:
        return [n.split("/", 1)[1] for n in tf.getnames() if "/" in n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", required=True, help="directory containing the built *.whl or *.tar.gz")
    parser.add_argument("--expect", required=True, choices=["compiled", "pure-python"])
    parser.add_argument("--artifact", default="wheel", choices=["wheel", "sdist"])
    args = parser.parse_args()

    if args.artifact == "wheel":
        paths = glob.glob(f"{args.dist_dir}/*.whl")
        assert len(paths) == 1, f"expected exactly one wheel in {args.dist_dir}, found {paths}"
        names = _wheel_names(paths[0])
    else:
        assert args.expect == "pure-python", "an sdist never contains compiled artifacts - only --expect pure-python makes sense with --artifact sdist"
        paths = glob.glob(f"{args.dist_dir}/*.tar.gz")
        assert len(paths) == 1, f"expected exactly one sdist in {args.dist_dir}, found {paths}"
        names = _sdist_names(paths[0])

    for module in MODULES:
        py_present = f"{module}.py" in names
        compiled_present = any(n.startswith(f"{module}.") and n.endswith((".so", ".pyd")) for n in names)
        assert py_present, f"{module}.py missing from {paths[0]}"
        if args.expect == "compiled":
            assert compiled_present, f"{module} was not compiled into {paths[0]} (found: {names})"
        else:
            assert not compiled_present, f"{module} was unexpectedly compiled into {paths[0]} (found: {names})"

    print(f"OK: all modules {MODULES} are '{args.expect}' in {paths[0]}")


if __name__ == "__main__":
    sys.exit(main())
