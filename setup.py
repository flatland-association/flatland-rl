#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from os import walk, path
import sys
from setuptools import setup, find_packages

assert sys.version_info >= (3, 6)
with open('README.md', 'r', encoding='utf8') as readme_file:
    readme = readme_file.read()


def get_all_file_paths(directory: str):
    paths = []
    for dirpath, dirnames, filenames in walk(directory):
        for file_name in filenames:
            paths.append(path.join(dirpath, file_name))
    return paths


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
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Multi Agent Reinforcement Learning on Trains",
    entry_points={
        'console_scripts': [
            'flatland-demo=flatland.cli:demo',
            'flatland-evaluator=flatland.cli:evaluator'
        ],
    },
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='flatland',
    name='flatland-rl',
    packages=find_packages('.'),
    data_files=[('svg', get_all_file_paths('./svg/')),
                ('images', get_all_file_paths('./images/')),
                ('notebooks', get_all_file_paths('./notebooks/'))],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/flatland-association/flatland-rl',
    version='3.0.15',
    zip_safe=False,
)
