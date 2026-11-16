$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path 'runtime/llama','runtime/models' -Force | Out-Null

    function Get-VerifiedFile($Url, $Destination, $Sha256) {
        if (!(Test-Path -LiteralPath $Destination)) {
            $partial = "$Destination.partial"
            curl.exe --fail --silent --show-error --location --retry 2 --continue-at - --output $partial $Url
            if ($LASTEXITCODE -ne 0) { throw "Download failed for $Destination" }
            if ((Get-FileHash -LiteralPath $partial).Hash.ToLowerInvariant() -ne $Sha256) {
                throw "Downloaded file hash mismatch for $Destination"
            }
            Move-Item -LiteralPath $partial -Destination $Destination
            return
        }
        if ((Get-FileHash -LiteralPath $Destination).Hash.ToLowerInvariant() -ne $Sha256) {
            throw "Existing file hash mismatch for $Destination"
        }
    }

    Get-VerifiedFile 'https://github.com/ggml-org/llama.cpp/releases/download/b12407/llama-b12407-bin-win-vulkan-x64.zip' 'runtime/llama-b12407.zip' 'fa487fce566c4143bcfd28cb0f48bc98c55ead18376b48eb67d06f4a8b17e233'
    Expand-Archive -LiteralPath 'runtime/llama-b12407.zip' -DestinationPath 'runtime/llama' -Force
    Get-VerifiedFile 'https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF/resolve/8d43d974efc48399cb58957ad49c8b26bf5a109c/Qwen3.5-4B-Q4_K_M.gguf' 'runtime/models/Qwen3.5-4B-Q4_K_M.gguf' '217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14'
    Write-Output 'llama.cpp and Qwen GGUF verified'



}
finally {
    Pop-Location
}
