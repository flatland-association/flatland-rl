#!/usr/bin/env bash
set -e # stop on error
set -x # echo commands


# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
FLATLAND_BASEDIR=$(dirname "$0")/..
FLATLAND_BASEDIR=$(realpath "$FLATLAND_BASEDIR")
echo "BASEDIR=${FLATLAND_BASEDIR}"

set +x
echo "************ TESTING PREREQUISITES PYTHON3 + GIT *************************"
set -x


set +x
echo "************ SETUP VIRTUAL ENVIRONMENT FLATLAND *************************"
set -x

export WORKON_HOME=${FLATLAND_BASEDIR}/getting_started/envs
echo WORKON_HOME=$WORKON_HOME
echo PWD=$PWD
mkdir -p ${WORKON_HOME}
# cannot work with virtualenvwrapper in script
cd ${WORKON_HOME}
python3 -m venv flatland
source flatland/bin/activate

set +x
echo "************ INSTALL FLATLAND IN THE VIRTUALENV  *************************"
set -x
cd ${FLATLAND_BASEDIR}
python setup.py install
# ensure jupyter is installed in the virtualenv
pip install -r ${FLATLAND_BASEDIR}/requirements_dev.txt -r requirements_continuous_integration.txt

set +x
echo "************ INSTALL JUPYTER EXTENSION *************************"
set -x
jupyter nbextension install --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension install --py --sys-prefix jpy_canvas
jupyter nbextension enable --py --sys-prefix jpy_canvas
jupyter notebook
