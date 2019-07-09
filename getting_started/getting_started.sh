#!/bin/bash
set -e # stop on error
set -x # echo commands


# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
FLATLAND_BASEDIR=$(dirname "$0")/..
FLATLAND_BASEDIR=$(realpath "$FLATLAND_BASEDIR")
echo "BASEDIR=${FLATLAND_BASEDIR}"

set +x
echo "************ TESTING PREREQUISITES PYTHON3 + GIT *************************"
set -x

git --version
python --version
conda --version
echo $PATH


set +x
echo "************ SETUP VIRTUAL ENVIRONMENT FLATLAND *************************"
set -x
source deactivate
(conda info --envs | fgrep flatland-rl) || conda create python=3.6 -y --name flatland-rl
source activate flatland-rl


set +x
echo "************ INSTALL FLATLAND IN THE VIRTUALENV  *************************"
set -x

# TODO we should get rid of having to install these packages outside of setup.py with conda!
conda install -y -c conda-forge cairosvg pycairo
conda install -y  -c anaconda tk
python -m pip install --upgrade pip

python ${FLATLAND_BASEDIR}/setup.py install

# ensure jupyter is installed in the virtualenv
python -m pip install --upgrade -r ${FLATLAND_BASEDIR}/requirements_dev.txt -r requirements_continuous_integration.txt


set +x
echo "************ INSTALL JUPYTER EXTENSION *************************"
set -x
jupyter nbextension install --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension install --py --sys-prefix jpy_canvas
jupyter nbextension enable --py --sys-prefix jpy_canvas


set +x
echo "************ RUN JUPYTER NOTEBOOKS *************************"
set -x
jupyter notebook &
