# start.ps1 — Start backend and Streamlit frontend on Windows (PowerShell)
param(
    [switch]$CreateVenv
)

$ErrorActionPreference = 'Stop'

# Optionally create venv
if ($CreateVenv) {
    python -m venv .\venv
    Write-Host "Created virtual environment .\venv"
}

# Activate venv if present
$venvActivate = Join-Path -Path $PWD -ChildPath "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..."
    & $venvActivate
} else {
    Write-Host "No virtual environment found at .\venv. Continuing with system Python." -ForegroundColor Yellow
}

# Start backend (uvicorn)
Write-Host "Starting FastAPI backend on http://localhost:8000..."
$backendProc = Start-Process -FilePath "python" -ArgumentList "-m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 2

# Start Streamlit
Write-Host "Starting Streamlit dashboard on http://localhost:8501..."
$streamlitProc = Start-Process -FilePath "streamlit" -ArgumentList "run dashboard\app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true" -WindowStyle Normal -PassThru

Write-Host "Started backend (PID: $($backendProc.Id)) and Streamlit (PID: $($streamlitProc.Id)). Use Stop-Process -Id <PID> to stop them."