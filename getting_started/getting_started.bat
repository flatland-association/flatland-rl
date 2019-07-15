@echo on

set FLATLAND_BASEDIR=%~dp0\..

cd %FLATLAND_BASEDIR%

call conda install -y -c conda-forge tox-conda || goto :error
call conda install -y tox || goto :error
call tox -v  --recreate || goto :error
call tox -v -e start_jupyter --recreate || goto :error

goto :EOF


:error
echo Failed with error #%errorlevel%.
pause
