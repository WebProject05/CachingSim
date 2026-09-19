@echo off
setlocal enabledelayedexpansion

:: Optional command-line overrides with sensible defaults
set EVAL_REQS=%~1
if "%EVAL_REQS%"=="" set EVAL_REQS=2000

set SOURCE_STEPS=%~2
if "%SOURCE_STEPS%"=="" set SOURCE_STEPS=5000

set TARGET_STEPS=%~3
if "%TARGET_STEPS%"=="" set TARGET_STEPS=6000

echo ======================================================================
echo  SMDP Edge Caching: 1-Click Complete Research Experiment Pipeline
echo  Test Requests: %EVAL_REQS% ^| Source Steps: %SOURCE_STEPS% ^| Target Steps: %TARGET_STEPS%
echo ======================================================================

pushd "%~dp0.."
set ROOT_DIR=%CD%

:: 1. Ensure Go Server and Baseline binaries are built
if not exist "%ROOT_DIR%\bin\server.exe" (
    echo [*] [1/5] Building Go Simulator Binaries...
    call "%ROOT_DIR%\scripts\build_go_windows.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo [!] Failed to build Go binaries.
        popd
        exit /b 1
    )
) else (
    echo [*] [1/5] Go binaries verified in bin\.
)

:: 2. Run Go Baseline Benchmarks, Parameter Sweeps & Table III MDP Evaluation
echo.
echo ======================================================================
echo [*] [2/5] Running Caching Benchmark: Proposed (SMDP-DDQL) vs CTD...
echo           (Baselines and Table III MDP metrics saved to data\results\)
echo ======================================================================
pushd "%ROOT_DIR%\simulator-go"
"%ROOT_DIR%\bin\baseline.exe" -requests 5000 -files 50 -capacity 10000 -g -mdp-table
popd

:: 3. Start Go gRPC Environment Server in background
echo.
echo ======================================================================
echo [*] [3/5] Starting Go SMDP Caching Server on port 50051...
echo ======================================================================
start "" /b "%ROOT_DIR%\bin\server.exe" -port 50051
timeout /t 2 /nobreak >nul

:: 4. Run Python Deep Reinforcement Learning & Transfer Learning Pipeline
echo.
echo ======================================================================
echo [*] [4/5] Executing Python SMDP-DDQL Agent and Transfer Learning Suite...
echo           (Source Domain DDQL -^> Policy Eval -^> Target Domain TL)
echo ======================================================================
pushd "%ROOT_DIR%\agent-python"
python main.py --mode full_experiment --source-steps %SOURCE_STEPS% --target-steps %TARGET_STEPS% --eval-requests %EVAL_REQS%
set PY_EXIT_CODE=!ERRORLEVEL!
popd

:: Stop background server
echo [*] Stopping background Go server...
taskkill /f /im server.exe >nul 2>&1

:: 5. Generate All Publication-Quality Research Paper Figures
echo.
echo ======================================================================
echo [*] [5/5] Rendering All 25 Research Paper Figures (PNG 300 DPI + SVG)...
echo ======================================================================
python "%ROOT_DIR%\scripts\generate_paper_graphs.py"
set GRAPH_EXIT_CODE=!ERRORLEVEL!

popd
echo.
echo ======================================================================
if !PY_EXIT_CODE! EQU 0 (
    echo [✓] Complete Research Pipeline Executed Successfully!
    echo.
    echo Artifacts Generated:
    echo   • All 25 Experimental Figures   : graphs\  (PNG + SVG)
    echo   • Graphs Analysis Documentation : graphs\README.md
    echo   • Recorded JSON Datasets        : data\results\*.json
    echo   • Trained PyTorch Checkpoints   : agent-python\checkpoints\
    echo   • System Data Flow Guide        : DATA_FLOW.md
) else (
    echo [!] Experiment encountered an error (Python Exit Code: !PY_EXIT_CODE!).
)
echo ======================================================================
exit /b !PY_EXIT_CODE!
