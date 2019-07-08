#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import sys

from setuptools import setup, find_packages

assert sys.version_info >= (3, 6)
with open('README.rst') as readme_file:
    readme = readme_file.read()


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
    version='0.1.2',
    zip_safe=False,
)
