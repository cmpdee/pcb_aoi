@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title PCB AOI - build Windows EXE

set "VENV=.buildvenv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo [INFO] Creating build environment...

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
    echo [ERROR] Python 3 was not found or the build environment could not be created.
    pause
    exit /b 1
)

if not exist "%~dp0requirements.txt" (
    echo [ERROR] Missing requirements.txt next to build_windows_exe.bat.
    echo Please extract the full ZIP, including requirements.txt, to one folder.
    pause
    exit /b 1
)

echo [INFO] Installing build dependencies...
"%PY%" -m pip install --upgrade pip wheel setuptools pyinstaller
if errorlevel 1 goto :fail
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail

echo [INFO] Building PCB_AOI.exe...
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name PCB_AOI --collect-all pymupdf --hidden-import fitz pcb_aoi.py
if errorlevel 1 goto :fail

echo.
echo [OK] EXE: %CD%\dist\PCB_AOI.exe
pause
exit /b 0

:fail
echo.
echo [ERROR] BUILD FAILED
pause
exit /b 1
