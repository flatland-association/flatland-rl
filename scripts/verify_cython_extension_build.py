#!/usr/bin/env python
"""
Verify whether the Cython-optional extensions declared in pyproject.toml's `[tool.setuptools] ext-modules`
(flatland.envs.step_utils.state_machine/states) were compiled into a wheel, or fell back to plain Python.

Used by tox.ini's verify-build-no-cython/verify-build-no-gcc/verify-cython-build envs.
"""
import argparse
import glob
import sys
import zipfile

MODULES = ["flatland/envs/step_utils/state_machine", "flatland/envs/step_utils/states"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", required=True, help="directory containing the built *.whl")
    parser.add_argument("--expect", required=True, choices=["compiled", "pure-python"])
    args = parser.parse_args()

    wheels = glob.glob(f"{args.dist_dir}/*.whl")
    assert len(wheels) == 1, f"expected exactly one wheel in {args.dist_dir}, found {wheels}"
    names = zipfile.ZipFile(wheels[0]).namelist()

    for module in MODULES:
        py_present = f"{module}.py" in names
        compiled_present = any(n.startswith(f"{module}.") and n.endswith((".so", ".pyd")) for n in names)
        assert py_present, f"{module}.py missing from wheel {wheels[0]}"
        if args.expect == "compiled":
            assert compiled_present, f"{module} was not compiled into {wheels[0]} (found: {names})"
        else:
            assert not compiled_present, f"{module} was unexpectedly compiled into {wheels[0]} (found: {names})"

    print(f"OK: all modules {MODULES} are '{args.expect}' in {wheels[0]}")


if __name__ == "__main__":
    sys.exit(main())
