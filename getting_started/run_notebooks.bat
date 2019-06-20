@echo on

set PWD_BEFORE=%cd%

@echo off
echo "************ TESTING PREREQUISITES PYTHON3 + GIT *************************"
@echo on

git --version || goto :error
python3 --version || goto :error


@echo off
echo "************ SETUP VIRTUAL ENVIRONMENT FLATLAND *************************"
@echo on

set FLATLAND_BASEDIR=%~dp0\..
set WORKON_HOME=%FLATLAND_BASEDIR%\getting_started\envs_win
if not exist "%WORKON_HOME%" md "%WORKON_HOME%" || goto :error
cd "%WORKON_HOME%"
rem use venv instead of virtualenv/virtualenv-wrapper because of https://github.com/pypa/virtualenv/issues/355
python3 -m venv flatland || goto :error
rem ignore activation error: https://stackoverflow.com/questions/51358202/python-3-7-activate-venv-error-parameter-format-not-correct-65001-windows
call "%WORKON_HOME%\flatland\Scripts\activate.bat" || true


@echo off
echo "************ INSTALL FLATLAND IN THE VIRTUALENV  *************************"
@echo on
python -m pip install --upgrade pip || goto :error
cd %FLATLAND_BASEDIR% || goto :error
python setup.py install || goto :error
REM ensure jupyter is installed in the virtualenv
pip install -r "%FLATLAND_BASEDIR%/requirements_dev.txt" -r "%FLATLAND_BASEDIR%\requirements_continuous_integration.txt" || goto :error

@echo off
echo "************ INSTALL JUPYTER EXTENSION *************************"
@echo on
jupyter nbextension install --py --sys-prefix widgetsnbextension || goto :error
jupyter nbextension enable --py --sys-prefix widgetsnbextension || goto :error
jupyter nbextension install --py --sys-prefix jpy_canvas || goto :error
jupyter nbextension enable --py --sys-prefix jpy_canvas || goto :error
jupyter notebook || goto :error


goto :EOF


:error
echo Failed with error #%errorlevel%.
cd "%PWD_BEFORE%" || true
deactivate || true
pause
