@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo  SMDP Edge Caching: Full Research Paper Experiment Pipeline
echo ======================================================================

pushd "%~dp0.."
set ROOT_DIR=%CD%

:: 1. Ensure Go Server is built
if not exist "%ROOT_DIR%\bin\server.exe" (
    echo [*] Building Go SMDP Server...
    call "%ROOT_DIR%\scripts\build_go_windows.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo [!] Failed to build Go server.
        popd
        exit /b 1
    )
)

:: 2. Start Go gRPC Environment Server in background
echo [*] Starting Go SMDP Caching Server on port 50051...
start "" /b "%ROOT_DIR%\bin\server.exe" -port 50051

:: Wait 2 seconds for server to bind port
timeout /t 2 /nobreak >nul

:: 3. Run Python RL/TL Agent Pipeline
echo [*] Launching Python DDQL and Transfer Learning Agent...
pushd "%ROOT_DIR%\agent-python"
python main.py --mode full_experiment --source-steps 5000 --target-steps 6000 --eval-requests 1000
set PY_EXIT_CODE=%ERRORLEVEL%
popd

:: 4. Terminate background server
echo [*] Stopping background Go server...
taskkill /f /im server.exe >nul 2>&1

popd
echo ======================================================================
if %PY_EXIT_CODE% EQU 0 (
    echo [*] Full Experiment Pipeline Completed Successfully!
) else (
    echo [!] Experiment encountered errors (Exit Code: %PY_EXIT_CODE%).
)
echo ======================================================================
exit /b %PY_EXIT_CODE%

