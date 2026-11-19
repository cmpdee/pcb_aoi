@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title PCB AOI

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

echo ==========================================
echo PCB AOI - setup and run
echo ==========================================
echo.

if not exist "%PY%" (
    echo [INFO] Creating local virtual environment...

    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV%"
    )

    if not exist "%PY%" (
        where python >nul 2>nul
        if not errorlevel 1 (
            python -m venv "%VENV%"
        )
    )
)

if not exist "%PY%" (
    echo.
    echo [ERROR] Python 3 was not found or the virtual environment could not be created.
    echo Install Python 3 from https://www.python.org/ and enable "Add Python to PATH".
    echo Then run this BAT again.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python:
"%PY%" --version
if errorlevel 1 goto :fail

echo.
if not exist "%~dp0requirements.txt" (
    echo [ERROR] Missing requirements.txt next to run_windows.bat.
    echo Please extract the full ZIP, including requirements.txt, to one folder.
    pause
    exit /b 1
)

echo [INFO] Installing/updating dependencies...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail

echo.
echo [INFO] Starting PCB AOI...
"%PY%" pcb_aoi.py
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] PCB AOI exited with code %APP_EXIT%.
    echo Send the text from this window if the problem persists.
    echo.
    pause
)

exit /b %APP_EXIT%

:fail
echo.
echo [ERROR] Setup failed. See the messages above.
echo.
pause
exit /b 1
