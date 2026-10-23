@echo off
setlocal enabledelayedexpansion

REM ======================================================================
REM SMDP Edge Caching: 1-Click Complete Research Experiment Pipeline
REM Paper: "Edge Caching Based on Deep Reinforcement Learning and Transfer Learning"
REM Authors: Farnaz Niknia, Ping Wang, Zixu Wang, Aakash Agarwal, Adib S. Rezaei
REM ======================================================================

REM Command-line parameter overrides:
REM %1 = Test requests (Default: 1000)
REM %2 = Source training steps (Default: 5000)
REM %3 = Target training steps (Default: 6000)
REM %4 = Cache capacity in MiB (Default: 10000)
REM %5 = Evaluation lambda (Default: 5.0 for 62%+ Table III benchmark)
REM %6 = Zipf eta skewness (Default: 1.0)
REM %7 = File types count F (Default: 50)

set EVAL_REQS=%~1
if "%EVAL_REQS%"=="" set EVAL_REQS=1000

set SOURCE_STEPS=%~2
if "%SOURCE_STEPS%"=="" set SOURCE_STEPS=5000

set TARGET_STEPS=%~3
if "%TARGET_STEPS%"=="" set TARGET_STEPS=6000

set CAPACITY=%~4
if "%CAPACITY%"=="" set CAPACITY=10000

set EVAL_LAMBDA=%~5
if "%EVAL_LAMBDA%"=="" set EVAL_LAMBDA=5.0

set ETA=%~6
if "%ETA%"=="" set ETA=1.0

set FILES=%~7
if "%FILES%"=="" set FILES=50

echo ======================================================================
echo  SMDP Edge Caching: Complete Research Experiment Pipeline
echo  Requests: %EVAL_REQS% ^| Source Steps: %SOURCE_STEPS% ^| Target Steps: %TARGET_STEPS%
echo  Capacity: %CAPACITY% MiB ^| Eval Lambda: %EVAL_LAMBDA% ^| Eta: %ETA% ^| Files: %FILES%
echo ======================================================================

pushd "%~dp0.."
set ROOT_DIR=%CD%

REM 1. Ensure Go Server and Baseline binaries are built
if not exist "%ROOT_DIR%\bin\server.exe" (
    echo [*] [1/6] Building Go Simulator Binaries...
    call "%ROOT_DIR%\scripts\build_go_windows.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo [!] Failed to build Go binaries.
        popd
        exit /b 1
    )
) else if not exist "%ROOT_DIR%\bin\baseline.exe" (
    echo [*] [1/6] Building Go Simulator Binaries...
    call "%ROOT_DIR%\scripts\build_go_windows.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo [!] Failed to build Go binaries.
        popd
        exit /b 1
    )
) else (
    echo [*] [1/6] Go binaries verified in bin\.
)

REM 2. Run Go Baseline Benchmarks, Parameter Sweeps and Table III MDP Evaluation
echo.
echo ======================================================================
echo [*] [2/6] Running Caching Benchmark: Proposed (SMDP-DDQL) vs CTD...
echo           (Baselines and Table III MDP metrics saved to data\results\)
echo ======================================================================
pushd "%ROOT_DIR%\simulator-go"
"%ROOT_DIR%\bin\baseline.exe" -requests %EVAL_REQS% -files %FILES% -capacity %CAPACITY% -eta %ETA% -g -mdp-table
set BASELINE_EXIT_CODE=!ERRORLEVEL!
popd
if !BASELINE_EXIT_CODE! NEQ 0 (
    echo [!] Baseline runner encountered an error.
    popd
    exit /b !BASELINE_EXIT_CODE!
)

REM 3. Ensure port 50051 is clear and start Go gRPC Environment Server
echo.
echo ======================================================================
echo [*] [3/6] Starting Go SMDP Caching Server (Capacity: %CAPACITY% MiB)...
echo ======================================================================
taskkill /f /im server.exe >nul 2>&1
start "" /b "%ROOT_DIR%\bin\server.exe" -port 50051 -capacity %CAPACITY% -files %FILES% -eta %ETA%
ping 127.0.0.1 -n 3 >nul

REM 4. Run Python Deep Reinforcement Learning and Transfer Learning Pipeline
echo.
echo ======================================================================
echo [*] [4/6] Executing Python SMDP-DDQL Agent and Transfer Learning Suite...
echo           (Source Domain DDQL -^> Policy Eval -^> Target Domain TL)
echo ======================================================================
pushd "%ROOT_DIR%\agent-python"
python main.py --mode full_experiment --source-steps %SOURCE_STEPS% --target-steps %TARGET_STEPS% --eval-requests %EVAL_REQS% --capacity %CAPACITY% --files %FILES% --eta %ETA% --eval-lambda %EVAL_LAMBDA%
set PY_EXIT_CODE=!ERRORLEVEL!
popd

REM Stop background server
echo [*] Stopping background Go server...
taskkill /f /im server.exe >nul 2>&1

if !PY_EXIT_CODE! NEQ 0 (
    echo [!] Python DRL/TL pipeline encountered an error [Code: !PY_EXIT_CODE!].
    popd
    exit /b !PY_EXIT_CODE!
)

REM 5. Generate All Publication-Quality Research Figures
echo.
echo ======================================================================
echo [*] [5/6] Rendering All Research Paper Figures [PNG 300 DPI + SVG]...
echo ======================================================================
python "%ROOT_DIR%\scripts\generate_paper_graphs.py"
set GRAPH_EXIT_CODE=!ERRORLEVEL!

REM 6. Generate Comprehensive Experiment Report (JSON + Text + Markdown)
echo.
echo ======================================================================
echo [*] [6/6] Generating Comprehensive Research Experiment Report...
echo           [Excluding classical baselines; paper algorithms only]
echo ======================================================================
python "%ROOT_DIR%\scripts\generate_experiment_report.py"

popd
echo.
echo ======================================================================
if !PY_EXIT_CODE! EQU 0 (
    echo [+] Complete Research Pipeline Executed Successfully!
    echo.
    echo Artifacts Generated:
    echo   - Comprehensive Report [JSON]   : data\results\experiment_report.json
    echo   - Comprehensive Report [Text]   : data\results\experiment_report.txt
    echo   - Comprehensive Report [MD]     : data\results\experiment_report.md
    echo   - Full Replication Document     : REPLICATION_REPORT.md
    echo   - All Research Paper Figures    : graphs\  [PNG + SVG]
    echo   - Recorded Raw JSON Datasets    : data\results\*.json
    echo   - Trained PyTorch Checkpoints   : agent-python\checkpoints\
    echo   - System Data Flow Guide        : DATA_FLOW.md
) else (
    echo [!] Experiment encountered an error [Python Exit Code: !PY_EXIT_CODE!].
)
echo ======================================================================
exit /b !PY_EXIT_CODE!
