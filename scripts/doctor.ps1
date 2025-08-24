# scripts/doctor.ps1
param(
  [switch]$NoVenv
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..")
Set-Location $root

if (-not $NoVenv) {
  $venv = Join-Path $root ".venv\Scripts\Activate.ps1"
  if (Test-Path $venv) { . $venv }
}

python scripts\doctor.py
