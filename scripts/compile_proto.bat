@echo off
setlocal enabledelayedexpansion

:: Get the parent directory of the script to set as PROJECT_ROOT
pushd "%~dp0.."
set PROJECT_ROOT=%CD%
popd

echo ==^> Compiling Protocol Buffers for Go and Python...

:: Ensure target directories exist
if not exist "%PROJECT_ROOT%\pkg\pb" (
    mkdir "%PROJECT_ROOT%\pkg\pb"
)
if not exist "%PROJECT_ROOT%\..\agent-python\pb" (
    mkdir "%PROJECT_ROOT%\..\agent-python\pb"
)

:: 1. Compile for Go
protoc --proto_path="%PROJECT_ROOT%\proto" ^
       --go_out="%PROJECT_ROOT%\pkg\pb" --go_opt=paths=source_relative ^
       --go-grpc_out="%PROJECT_ROOT%\pkg\pb" --go-grpc_opt=paths=source_relative ^
       "%PROJECT_ROOT%\proto\cache_env.proto"

if %ERRORLEVEL% NEQ 0 (
    echo Go protobuf compilation failed.
    exit /b %ERRORLEVEL%
)

:: 2. Compile for Python (if Python is available)
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python -m grpc_tools.protoc ^
        -I="%PROJECT_ROOT%\proto" ^
        --python_out="%PROJECT_ROOT%\..\agent-python\pb" ^
        --grpc_python_out="%PROJECT_ROOT%\..\agent-python\pb" ^
        "%PROJECT_ROOT%\proto\cache_env.proto"

    if !ERRORLEVEL! NEQ 0 (
        echo Python protobuf compilation failed.
    )
) else (
    echo Python not found. Skipping Python protobuf compilation.
)

echo ==^> Protobuf compilation complete.