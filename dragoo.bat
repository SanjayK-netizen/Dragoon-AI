@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python virtual environment not found:
    echo "%PYTHON%"
    echo Create it and install dependencies before starting Dragoon.
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Dragoon exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
 