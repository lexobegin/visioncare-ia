# VisionCare AI - Script de Arranque PowerShell (Windows)
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

Write-Host "Iniciando VisionCare AI con entorno optimizado..." -ForegroundColor Cyan
& $venvPython run.py @args
