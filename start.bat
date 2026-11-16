@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [VisionCare AI] Iniciando servicio con entorno optimizado...
"%PYTHON_EXE%" run.py %*
