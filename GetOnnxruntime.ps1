[CmdletBinding()]
param(
    # Version to download, e.g. "1.18.1" or "v1.18.1". If omitted, downloads the latest release.
    [string]$Version,

    # Output directory where the package will be extracted
    [string]$OutDir = (Join-Path -Path $PSScriptRoot -ChildPath 'onnxruntime')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure TLS1.2 for GitHub
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$repo = 'microsoft/onnxruntime'

function Get-GitHubHeaders {
    # No token required; releases are public
    return @{
        'User-Agent' = 'onnxruntime-downloader'
        'Accept'     = 'application/vnd.github+json'
    }
}

function Get-AssetName {
    param(
        [Parameter(Mandatory=$true)][string]$VersionNumber
    )
    return "onnxruntime-win-x64-$VersionNumber.zip"
}

function Get-ReleaseInfo {
    param([string]$Version)

    $url = if ($Version) {
        $tag = if ($Version -match '^[vV]') { $Version } else { "v$Version" }
        "https://api.github.com/repos/$repo/releases/tags/$tag"
    } else {
        "https://api.github.com/repos/$repo/releases/latest"
    }

    try {
        return Invoke-RestMethod -Uri $url -Headers (Get-GitHubHeaders) -ErrorAction Stop
    }
    catch {
        if ($Version) {
            # Fall back with minimal info; we can still construct the asset URL directly
            $tag = if ($Version -match '^[vV]') { $Version } else { "v$Version" }
            return [pscustomobject]@{
                tag_name = $tag
                assets   = @()
            }
        }
        throw
    }
}

function Get-AllReleases {
    param(
        [int]$MaxReleases = 50
    )

    $url = "https://api.github.com/repos/$repo/releases?per_page=$MaxReleases"
    
    try {
        return Invoke-RestMethod -Uri $url -Headers (Get-GitHubHeaders) -ErrorAction Stop
    }
    catch {
        throw "Failed to get releases list: $_"
    }
}

function Find-LatestReleaseWithWindows {
    param(
        [string]$SpecificVersion
    )

    if ($SpecificVersion) {
        # If a specific version is requested, use the original logic
        return Get-ReleaseInfo -Version $SpecificVersion
    }

    Write-Host "Searching for the latest release with Windows package..."
    
    $releases = Get-AllReleases
    $foundRelease = $null
    $skippedReleases = @()

    foreach ($release in $releases) {
        $tag = $release.tag_name
        $versionNumber = $tag.TrimStart('v','V')
        $assetName = Get-AssetName -VersionNumber $versionNumber
        
        $asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
        
        if ($null -ne $asset) {
            $foundRelease = $release
            break
        } else {
            $skippedReleases += $tag
        }
    }

    # Print warnings for skipped releases
    foreach ($skippedTag in $skippedReleases) {
        Write-Warning "Release $skippedTag does not have Windows x64 package, skipping..."
    }

    if ($null -eq $foundRelease) {
        throw "No recent releases found with Windows x64 package. Please check the repository manually."
    }

    if ($skippedReleases.Count -gt 0) {
        Write-Host "Found release $($foundRelease.tag_name) with Windows package (skipped $($skippedReleases.Count) newer release$(if ($skippedReleases.Count -ne 1) {'s'}))."
    } else {
        Write-Host "Using latest release $($foundRelease.tag_name)."
    }

    return $foundRelease
}

# Resolve release and asset
$release = Find-LatestReleaseWithWindows -SpecificVersion $Version
$tag = $release.tag_name
$versionNumber = $tag.TrimStart('v','V')
$assetName = Get-AssetName -VersionNumber $versionNumber

# Try to locate asset via API first
$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
if ($null -ne $asset) {
    $downloadUrl = $asset.browser_download_url
} else {
    # Construct direct URL for the asset (fallback for specific versions)
    $downloadUrl = "https://github.com/$repo/releases/download/$tag/$assetName"
    
    # For specific versions, we still try the direct URL, but warn about potential issues
    if ($Version) {
        Write-Warning "Asset not found in release metadata for version $Version. Attempting direct download - this may fail if the asset doesn't exist."
    }
}

# Prepare paths
if (Test-Path $OutDir) {
    Write-Host "Cleaning output directory: $OutDir"
    Remove-Item -Recurse -Force -LiteralPath $OutDir
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Use a dedicated temp working directory for downloads and extraction
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("onnxruntime-dl-" + [guid]::NewGuid().ToString('N'))
if (Test-Path $tempRoot) { Remove-Item -Recurse -Force -LiteralPath $tempRoot -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

$zipPath = Join-Path $tempRoot $assetName
$tmpExtract = Join-Path $tempRoot "__extract__"

try {
    # Download fresh copy to temp
    Write-Host "Downloading $assetName from $downloadUrl ..."
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -ErrorAction Stop
    }
    catch {
        $errorMessage = "Failed to download '$downloadUrl'. $_"
        if ($Version -and $null -eq $asset) {
            $errorMessage += "`nNote: The specified version '$Version' may not have a Windows x64 package available."
        }
        throw $errorMessage
    }

    # Extract and flatten so files are directly under $OutDir
    Write-Host "Extracting to $OutDir ..."
    if (Test-Path $tmpExtract) { Remove-Item -Recurse -Force -LiteralPath $tmpExtract }
    New-Item -ItemType Directory -Path $tmpExtract -Force | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $tmpExtract -Force

    # Move inner content up if the zip has a single root folder
    $top = @(Get-ChildItem -LiteralPath $tmpExtract -Force)
    if ($top.Count -eq 1 -and $top[0].PSIsContainer) {
        $innerRoot = $top[0].FullName
        Get-ChildItem -LiteralPath $innerRoot -Force | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination $OutDir -Force
        }
    } else {
        foreach ($item in $top) {
            Move-Item -LiteralPath $item.FullName -Destination $OutDir -Force
        }
    }

    Write-Host "ONNX Runtime downloaded and extracted to: $OutDir"
    Write-Output $OutDir
}
finally {
    # Always clean up temporary working directory (zip and temp extract)
    if (Test-Path $tempRoot) {
        try { Remove-Item -Recurse -Force -LiteralPath $tempRoot -ErrorAction SilentlyContinue } catch { }
    }
}
