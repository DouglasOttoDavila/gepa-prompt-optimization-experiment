$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousCache = $env:UV_CACHE_DIR
$previousKey = $env:OPENAI_API_KEY
try {
    $env:UV_CACHE_DIR = Join-Path $PWD 'security/cache'
    $env:TEMP = Join-Path $PWD 'security/tmp'
    $env:TMP = $env:TEMP
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path 'runtime','security/tmp' -Force | Out-Null
    $archive = Join-Path $PWD 'runtime/python-3.12.16.tar.gz'
    $python = Join-Path $PWD 'runtime/python/python.exe'
    $url = 'https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
    if (!(Test-Path -LiteralPath $archive)) {
        $partial = "$archive.partial"
        curl.exe --fail --silent --show-error --location --retry 2 --continue-at - --output $partial $url
        if ($LASTEXITCODE -ne 0) { throw 'CPython download failed' }
        Move-Item -LiteralPath $partial -Destination $archive
    }
    if ((Get-FileHash -LiteralPath $archive).Hash.ToLowerInvariant() -ne '355ee61b8207b91574491b1c6dc9b0ebb4408fc56bfafe498f3ef439314b52dc') {
        throw 'CPython archive hash mismatch'
    }
    if (!(Test-Path -LiteralPath $python)) {
        tar.exe -xzf $archive -C runtime
        if ($LASTEXITCODE -ne 0) { throw 'CPython extraction failed' }
    }
    if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv venv --python $python .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment setup failed' }
    }
    if ((.venv/Scripts/python.exe --version) -ne 'Python 3.12.16') {
        throw 'The existing virtual environment uses an unreviewed Python version'
    }
    uv --no-config pip sync --python .venv/Scripts/python.exe --require-hashes --no-build --default-index https://pypi.org/simple requirements.lock
    if ($LASTEXITCODE -ne 0) { throw 'Hashed dependency install failed' }
    uv pip check --python .venv/Scripts/python.exe
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
    .venv/Scripts/python.exe --version
}
finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:UV_CACHE_DIR = $previousCache
    $env:OPENAI_API_KEY = $previousKey
    Pop-Location
}
