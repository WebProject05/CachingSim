<#
.SYNOPSIS
    Complete Research Experiment Pipeline for SMDP Edge Caching Framework.
    Paper: "Edge Caching Based on Deep Reinforcement Learning and Transfer Learning"
    (Niknia et al., IEEE / arXiv:2402.14576v2, 2024)

.DESCRIPTION
    Executes the entire research experiment with fully configurable parameters:
    - Number of requests / evaluation steps
    - Training steps for source and target domains
    - Cache capacity in MiB
    - Evaluation lambda request rate
    - Zipf popularity exponent eta
    - Total file types count F

.PARAMETER EvalRequests
    Number of evaluation requests (Default: 1000)
.PARAMETER SourceSteps
    Source domain training steps (Default: 5000)
.PARAMETER TargetSteps
    Target domain training steps (Default: 6000)
.PARAMETER Capacity
    Cache capacity in MiB (Default: 10000, e.g. 5000, 15000, 20000)
.PARAMETER EvalLambda
    Evaluation lambda request rate (Default: 5.0 for 62%+ Table III benchmark)
.PARAMETER ZipfEta
    Zipf popularity skewness parameter eta (Default: 1.0)
.PARAMETER Files
    Total number of file types F (Default: 50)
#>

param(
    [int]$EvalRequests = 1000,
    [int]$SourceSteps = 5000,
    [int]$TargetSteps = 6000,
    [double]$Capacity = 10000,
    [double]$EvalLambda = 5.0,
    [double]$ZipfEta = 1.0,
    [int]$Files = 50
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " SMDP Edge Caching: Complete Research Experiment Pipeline" -ForegroundColor Cyan
Write-Host " Requests: $EvalRequests | Source Steps: $SourceSteps | Target Steps: $TargetSteps" -ForegroundColor Cyan
Write-Host " Capacity: $Capacity MiB | Eval Lambda: $EvalLambda | Eta: $ZipfEta | Files: $Files" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify / Build Go Simulator Binaries
$ServerBin = Join-Path $RootDir "bin\server.exe"
$BaselineBin = Join-Path $RootDir "bin\baseline.exe"

if (-not (Test-Path $ServerBin) -or -not (Test-Path $BaselineBin)) {
    Write-Host "[*] [1/6] Building Go Simulator Binaries..." -ForegroundColor Yellow
    & "$ScriptDir\build_go_windows.bat"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[!] Failed to build Go simulator binaries."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "[*] [1/6] Go simulator binaries verified in bin\." -ForegroundColor Green
}

# 2. Run Go Baseline Sweeps and Table III MDP Evaluation
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] [2/6] Running Caching Benchmark: Proposed (SMDP-DDQL) vs CTD..." -ForegroundColor Yellow
Write-Host "          (Baselines and Table III MDP metrics saved to data\results\)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

Push-Location (Join-Path $RootDir "simulator-go")
try {
    & $BaselineBin -requests $EvalRequests -files $Files -capacity $Capacity -eta $ZipfEta -g -mdp-table
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[!] Baseline runner failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

# 3. Ensure port 50051 is free and start Go gRPC Server in background
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] [3/6] Starting Go SMDP Caching Server (Capacity: $Capacity MiB)..." -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

Get-Process -Name "server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

$ServerProc = Start-Process -FilePath $ServerBin -ArgumentList "-port 50051 -capacity $Capacity -files $Files -eta $ZipfEta" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

# 4. Run Python DRL and Transfer Learning Pipeline
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] [4/6] Executing Python SMDP-DDQL Agent and Transfer Learning Suite..." -ForegroundColor Yellow
Write-Host "          (Source Domain DDQL -> Policy Eval -> Target Domain TL)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

$PyExitCode = 0
Push-Location (Join-Path $RootDir "agent-python")
try {
    & python main.py --mode full_experiment --source-steps $SourceSteps --target-steps $TargetSteps --eval-requests $EvalRequests --capacity $Capacity --files $Files --eta $ZipfEta --eval-lambda $EvalLambda
    $PyExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    # Always clean up background server process
    Write-Host "[*] Stopping background Go server..." -ForegroundColor Yellow
    if ($ServerProc -and -not $ServerProc.HasExited) {
        Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
    }
    Get-Process -Name "server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

if ($PyExitCode -ne 0) {
    Write-Error "[!] Python pipeline failed with exit code $PyExitCode"
    exit $PyExitCode
}

# 5. Generate All Publication-Quality Figures (PNG + SVG)
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] [5/6] Rendering All Research Paper Figures (PNG 300 DPI + SVG)..." -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

& python (Join-Path $RootDir "scripts\generate_paper_graphs.py")

# 6. Generate Comprehensive Reports
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] [6/6] Generating Comprehensive Research Experiment Report..." -ForegroundColor Yellow
Write-Host "          (Excluding classical baselines; paper algorithms only)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

& python (Join-Path $RootDir "scripts\generate_experiment_report.py")

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "[+] Complete Research Pipeline Executed Successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Artifacts Generated:"
Write-Host "  - Comprehensive Report (JSON)   : data\results\experiment_report.json"
Write-Host "  - Comprehensive Report (Text)   : data\results\experiment_report.txt"
Write-Host "  - Comprehensive Report (MD)     : data\results\experiment_report.md"
Write-Host "  - Full Replication Document     : REPLICATION_REPORT.md"
Write-Host "  - All Research Paper Figures    : graphs\  (PNG + SVG)"
Write-Host "  - Recorded Raw JSON Datasets    : data\results\*.json"
Write-Host "  - Trained PyTorch Checkpoints   : agent-python\checkpoints\"
Write-Host "  - System Data Flow Guide        : DATA_FLOW.md"
Write-Host "======================================================================" -ForegroundColor Green
