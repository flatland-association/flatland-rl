@echo on

set FLATLAND_BASEDIR=%~dp0\..

@echo off
echo "************ TESTING PREREQUISITES PYTHON3 + GIT + GIT *************************"
@echo on

git --version || goto :error
python --version || goto :error
rem deactivate in case we're in virtualenv
call conda deactivate || call deactivate


@echo off
echo "************ SETUP VIRTUAL ENVIRONMENT FLATLAND *************************"
@echo on
(conda info --envs | findstr flatland-rl) || conda create python=3.6 -y --name flatland-rl || goto :error
(call conda activate flatland-rl || call activate flatland-rl) || goto :error


@echo off
echo "************ INSTALL FLATLAND AND DEPENDENCIES IN THE VIRTUALENV  *************************"
@echo on
rem TODO we should get rid of having to install these packages outside of setup.py with conda!
call conda install -y -c conda-forge cairosvg pycairo || goto :error
call conda install -y -c anaconda tk || goto :error
call python -m pip install --upgrade pip || goto :error
python setup.py install || goto :error

# ensure jupyter is installed in the virtualenv
python -m pip install --upgrade -r %FLATLAND_BASEDIR%/requirements_dev.txt -r %FLATLAND_BASEDIR%/requirements_continuous_integration.txt || goto :error


@echo off
echo "************ INSTALL JUPYTER EXTENSION *************************"
@echo on
jupyter nbextension install --py --sys-prefix widgetsnbextension || goto :error
jupyter nbextension enable --py --sys-prefix widgetsnbextension || goto :error
jupyter nbextension install --py --sys-prefix jpy_canvas || goto :error
jupyter nbextension enable --py --sys-prefix jpy_canvas || goto :error


@echo off
echo "************ RUN JUPYTER NOTEBOOKS *************************"
@echo on
jupyter notebook || goto :error


goto :EOF


:error
echo Failed with error #%errorlevel%.
pause
