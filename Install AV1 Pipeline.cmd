@echo off
rem ====================================================================
rem Install AV1 Pipeline.cmd
rem Double-click this. It runs install_av1_pipeline.ps1 from whatever
rem folder this file happens to be sitting in, so the two files just
rem have to stay together. The working directory does not matter.
rem ====================================================================

title AV1 Pipeline installer
cd /d "%~dp0"

if not exist "%~dp0install_av1_pipeline.ps1" (
    echo.
    echo install_av1_pipeline.ps1 was not found next to this file.
    echo Put both files in the same folder and run this again.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_av1_pipeline.ps1"

echo.
echo ====================================================================
echo Installer finished. Read the lines above for anything marked NOT
echo found or failed. Press any key to close this window.
echo ====================================================================
pause >nul
