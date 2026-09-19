@echo off
setlocal

pushd "%~dp0..\simulator-go"
if errorlevel 1 (
	echo Could not enter the simulator directory.
	exit /b 1
)

echo [*] Running Go test suite...
go test ./...
if errorlevel 1 (
	echo Go tests failed.
	echo [!] Go tests failed.
	popd
	exit /b 1
)

if not exist "..\bin" mkdir "..\bin"

echo [*] Building bin\server.exe...
go build -o "..\bin\server.exe" .\cmd\server
if errorlevel 1 (
	echo Go server build failed.
	echo [!] Go server build failed.
	popd
	exit /b 1
)

echo [*] Building bin\baseline.exe...
go build -o "..\bin\baseline.exe" .\cmd\baseline
if errorlevel 1 (
	echo [!] Go baseline build failed.
	popd
	exit /b 1
)

popd
echo Go simulator build completed successfully.
echo [*] Go simulator binaries built successfully.
