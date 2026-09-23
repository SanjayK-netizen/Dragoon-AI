@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python virtual environment not found:
    echo "%PYTHON%"
    pause
    exit /b 1
)

"%PYTHON%" -m streamlit run "%~dp0streamlit_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Streamlit exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
