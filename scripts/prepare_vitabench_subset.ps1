param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\\Scripts\\python.exe"

if (-not (Test-Path $python)) {
    throw "PAHF virtual environment is missing: $python"
}

& $python (Join-Path $root "scripts\\prepare_vitabench_subset.py")
