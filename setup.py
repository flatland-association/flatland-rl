from Cython.Build import cythonize
from setuptools import setup

setup(
    name='Hello world app',
    ext_modules=cythonize(["flatland/envs/step_utils/state_machine.py", "flatland/envs/step_utils/states.py"]),
)
