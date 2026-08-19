@echo off
setlocal

pushd "%~dp0..\simulator-go"
if errorlevel 1 (
	echo Could not enter the simulator directory.
	exit /b 1
)

go test ./...
if errorlevel 1 (
	echo Go tests failed.
	popd
	exit /b 1
)

if not exist "..\bin" mkdir "..\bin"
go build -o "..\bin\server.exe" .\cmd\server
if errorlevel 1 (
	echo Go server build failed.
	popd
	exit /b 1
)

popd
echo Go simulator build completed successfully.
