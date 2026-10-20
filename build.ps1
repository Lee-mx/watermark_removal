param(
    [string]$Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)

$ErrorActionPreference = 'Stop'
$desktopRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Push-Location -LiteralPath $desktopRoot
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'React build failed.' }

    & $Python (Join-Path $desktopRoot 'build_app.py')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
} finally {
    Pop-Location
}
