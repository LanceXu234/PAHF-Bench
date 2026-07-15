param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("shopping", "embodied")]
    [string]$Agent,
    [ValidateSet("sql", "faiss")]
    [string]$MemStyle = "sql",
    [string]$Model = "",
    [string]$HumanModel = "",
    [switch]$NoMemory
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env.pahf"
$python = Join-Path $root ".venv\\Scripts\\python.exe"

if (-not (Test-Path $python)) {
    throw "PAHF virtual environment is missing: $python"
}

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
        $parts = $_.Split('=', 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

$args = @(
    "run_agent.py",
    "--agent", $Agent,
    "--mem_style", $MemStyle
)

if ($NoMemory) {
    $args += "--no-memory"
}
if ($Model) {
    $args += @("--model", $Model)
}
if ($HumanModel) {
    $args += @("--human_model", $HumanModel)
}

Push-Location $root
try {
    & $python @args
}
finally {
    Pop-Location
}
