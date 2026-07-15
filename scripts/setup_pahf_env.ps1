param(
    [switch]$WithMemoryDeps
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\\python.exe"

function Invoke-Checked {
    param(
        [string]$Label,
        [string[]]$CommandArgs
    )

    & $python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed during $Label (exit code $LASTEXITCODE): $python $($CommandArgs -join ' ')"
    }
}

if (-not (Test-Path $python)) {
    py -3 -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment at: $venv"
    }
}

Invoke-Checked -Label "pip upgrade" -CommandArgs @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Label "native dependency install" -CommandArgs @("-m", "pip", "install", "-r", (Join-Path $root "requirements.native.txt"))

if ($WithMemoryDeps) {
    Invoke-Checked -Label "embedding dependency install" -CommandArgs @("-m", "pip", "install", "-r", (Join-Path $root "requirements.memory.txt"))
}

Write-Host "PAHF virtual environment ready at: $venv"
Write-Host "Fill API config in: $(Join-Path $root '.env.pahf')"
Write-Host "You can copy from: $(Join-Path $root '.env.pahf.example')"
