#!/bin/bash
set -e # stop on error
set -x # echo commands


FLATLAND_BASEDIR=$(dirname "$0")/..
cd ${FLATLAND_BASEDIR}

conda install -y -c conda-forge tox-conda
conda install -y tox
tox -v
tox -v -e start_jupyter &
