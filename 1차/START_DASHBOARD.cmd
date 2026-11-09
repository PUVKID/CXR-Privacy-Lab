@echo off
setlocal
set "NODE_EXE=node"
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" set "NODE_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
cd /d "%~dp0"
echo CXR Privacy Lab: http://127.0.0.1:8786
"%NODE_EXE%" "%~dp0server.mjs"
if errorlevel 1 pause
