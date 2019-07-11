@echo on

set FLATLAND_BASEDIR=%~dp0\..

cd %FLATLAND_BASEDIR%

conda install -c conda-forge tox-conda || goto :error
conda install tox || goto :error
tox -v -e start_jupyter --recreate || goto :error


goto :EOF


:error
echo Failed with error #%errorlevel%.
pause
