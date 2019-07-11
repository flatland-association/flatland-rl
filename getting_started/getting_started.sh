#!/bin/bash
set -e # stop on error
set -x # echo commands


# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
FLATLAND_BASEDIR=$(dirname "$0")/..
FLATLAND_BASEDIR=$(realpath "$FLATLAND_BASEDIR")
cd ${FLATLAND_BASEDIR}

conda install -y -c conda-forge tox-conda
conda install -y tox
tox -v -e start_jupyter &
