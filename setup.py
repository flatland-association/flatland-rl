#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import platform
import sys

from setuptools import setup, find_packages

assert sys.version_info >= (3, 6)
with open('README.rst') as readme_file:
    readme = readme_file.read()

# install pycairo on Windows
if os.name == 'nt':
    p = platform.architecture()
    is64bit = p[0] == '64bit'
    if sys.version[0:3] == '3.5':
        if is64bit:
            url = 'https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/pycairo-1.18.1-cp35-cp35m-win_amd64.whl'
        else:
            url = 'https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/pycairo-1.18.1-cp35-cp35m-win32.whl'

    if sys.version[0:3] == '3.6':
        if is64bit:
            url = 'https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/pycairo-1.18.1-cp36-cp36m-win_amd64.whl'
        else:
            url = 'https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/pycairo-1.18.1-cp36-cp36m-win32.whl'

    if sys.version[0:3] == '3.7':
        if is64bit:
            url = 'https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/pycairo-1.18.1-cp37-cp37m-win_amd64.whl'
        else:
            url = 'https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/pycairo-1.18.1-cp37-cp37m-win32.whl'

    try:
        import pycairo
    except:
        call_cmd = "pip install " + url
        os.system(call_cmd)

        import site
        import ctypes.util

        default_os_path = os.environ['PATH']
        os.environ['PATH'] = ''
        for s in site.getsitepackages():
            os.environ['PATH'] = os.environ['PATH'] + ';' + s + '\\cairo'
        os.environ['PATH'] = os.environ['PATH'] + ';' + default_os_path
        print(os.environ['PATH'])
        if ctypes.util.find_library('cairo') is not None:
            print("cairo installed: OK")
else:
    try:
        import pycairo
    except:
        os.system("pip install pycairo==1.18.1")


def get_all_svg_files(directory='./svg/'):
    ret = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            ret.append(directory + f)
    return ret


def get_all_images_files(directory='./images/'):
    ret = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            ret.append(directory + f)
    return ret


# Gather requirements from requirements_dev.txt
install_reqs = []
requirements_path = 'requirements_dev.txt'
with open(requirements_path, 'r') as f:
    install_reqs += [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]
requirements = install_reqs
setup_requirements = install_reqs
test_requirements = install_reqs

setup(
    author="S.P. Mohanty",
    author_email='mohanty@aicrowd.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Multi Agent Reinforcement Learning on Trains",
    entry_points={
        'console_scripts': [
            'flatland=flatland.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='flatland',
    name='flatland-rl',
    packages=find_packages('.'),
    data_files=[('svg', get_all_svg_files()), ('images', get_all_images_files())],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.aicrowd.com/flatland/flatland',
    version='0.2.0',
    zip_safe=False,
)
