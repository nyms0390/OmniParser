<#
.SYNOPSIS
    Batch Chandra OCR inference on PDF files using a remote vLLM server.

.DESCRIPTION
    Recursively finds all PDF files under InputDir, runs Chandra OCR on each
    via the specified vLLM server, and writes output (Markdown, HTML, metadata)
    into OutputDir mirroring the input directory tree.

    Requires chandra-ocr to be installed:
        pip install chandra-ocr

.PARAMETER InputDir
    Root folder containing PDF files (searched recursively).

.PARAMETER OutputDir
    Root output folder. Directory tree mirrors InputDir.

.PARAMETER BaseUrl
    Base URL of the vLLM server, e.g. http://host:8000/v1

.PARAMETER MaxOutputTokens
    Maximum output tokens per page. Lower this if the server returns a
    "max_tokens too large" error. Default: 4096.

.EXAMPLE
    .\run_chandra.ps1 -InputDir .\docs -OutputDir .\results -BaseUrl http://localhost:8000/v1

.EXAMPLE
    .\run_chandra.ps1 -InputDir .\docs -OutputDir .\results -BaseUrl http://10.0.0.5:8000/v1 -MaxOutputTokens 8192
#>

param (
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $false)]
    [int]$MaxOutputTokens = 4096
)

# ---- Resolve paths ----
$InputDir  = (Resolve-Path $InputDir).Path
$OutputDir = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName

# ---- Set vLLM base URL for the chandra CLI ----
$env:VLLM_API_BASE = $BaseUrl
Write-Host "VLLM_API_BASE = $env:VLLM_API_BASE"
Write-Host "Input:  $InputDir"
Write-Host "Output: $OutputDir"
Write-Host "Max output tokens: $MaxOutputTokens"
Write-Host ""

# ---- Discover PDFs ----
$pdfs = Get-ChildItem -Path $InputDir -Recurse -Filter "*.pdf"
if ($pdfs.Count -eq 0) {
    Write-Warning "No PDF files found under $InputDir"
    exit 0
}
Write-Host "Found $($pdfs.Count) PDF file(s)"
Write-Host ""

# ---- Process each PDF ----
$success = 0
$failed  = 0
$start   = Get-Date

foreach ($pdf in $pdfs) {
    # Mirror the subdirectory structure under OutputDir
    $rel     = $pdf.DirectoryName.Substring($InputDir.Length).TrimStart('\', '/')
    $outDir  = if ($rel) { Join-Path $OutputDir $rel } else { $OutputDir }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    Write-Host "Processing: $($pdf.FullName.Substring($InputDir.Length).TrimStart('\', '/'))"

    $t0 = Get-Date
    chandra $pdf.FullName $outDir --method vllm --max-output-tokens $MaxOutputTokens
    $elapsed = ((Get-Date) - $t0).TotalSeconds

    if ($LASTEXITCODE -eq 0) {
        $success++
        Write-Host "  OK ($([math]::Round($elapsed, 1))s)"
    } else {
        $failed++
        Write-Warning "  FAILED (exit code $LASTEXITCODE)"
    }
    Write-Host ""
}

# ---- Summary ----
$totalElapsed = ((Get-Date) - $start).TotalSeconds
Write-Host "Done — $success/$($pdfs.Count) succeeded, $failed failed, $([math]::Round($totalElapsed, 1))s total"
