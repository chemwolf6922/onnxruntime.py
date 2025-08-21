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
        [Parameter(Mandatory=$true)][string]$VersionNumber,
        [Parameter(Mandatory=$false)][string]$Architecture = "x64"
    )
    return "onnxruntime-win-$Architecture-$VersionNumber.zip"
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

    Write-Host "Searching for the latest release with Windows packages (x64 and arm64)..."
    
    $releases = Get-AllReleases
    $foundRelease = $null
    $skippedReleases = @()

    foreach ($release in $releases) {
        $tag = $release.tag_name
        $versionNumber = $tag.TrimStart('v','V')
        $x64AssetName = Get-AssetName -VersionNumber $versionNumber -Architecture "x64"
        $arm64AssetName = Get-AssetName -VersionNumber $versionNumber -Architecture "arm64"
        
        $x64Asset = $release.assets | Where-Object { $_.name -eq $x64AssetName } | Select-Object -First 1
        $arm64Asset = $release.assets | Where-Object { $_.name -eq $arm64AssetName } | Select-Object -First 1
        
        if ($null -ne $x64Asset -and $null -ne $arm64Asset) {
            $foundRelease = $release
            break
        } else {
            $skippedReleases += $tag
        }
    }

    # Print warnings for skipped releases
    foreach ($skippedTag in $skippedReleases) {
        Write-Warning "Release $skippedTag does not have both Windows x64 and arm64 packages, skipping..."
    }

    if ($null -eq $foundRelease) {
        throw "No recent releases found with both Windows x64 and arm64 packages. Please check the repository manually."
    }

    if ($skippedReleases.Count -gt 0) {
        Write-Host "Found release $($foundRelease.tag_name) with both Windows packages (skipped $($skippedReleases.Count) newer release$(if ($skippedReleases.Count -ne 1) {'s'}))."
    } else {
        Write-Host "Using latest release $($foundRelease.tag_name)."
    }

    return $foundRelease
}

# Resolve release and asset
$release = Find-LatestReleaseWithWindows -SpecificVersion $Version
$tag = $release.tag_name
$versionNumber = $tag.TrimStart('v','V')
$x64AssetName = Get-AssetName -VersionNumber $versionNumber -Architecture "x64"
$arm64AssetName = Get-AssetName -VersionNumber $versionNumber -Architecture "arm64"

# Try to locate assets via API first
$x64Asset = $release.assets | Where-Object { $_.name -eq $x64AssetName } | Select-Object -First 1
$arm64Asset = $release.assets | Where-Object { $_.name -eq $arm64AssetName } | Select-Object -First 1

$downloadUrls = @{}

if ($null -ne $x64Asset) {
    $downloadUrls["x64"] = $x64Asset.browser_download_url
} else {
    # Construct direct URL for the asset (fallback for specific versions)
    $downloadUrls["x64"] = "https://github.com/$repo/releases/download/$tag/$x64AssetName"
    
    # For specific versions, we still try the direct URL, but warn about potential issues
    if ($Version) {
        Write-Warning "x64 asset not found in release metadata for version $Version. Attempting direct download - this may fail if the asset doesn't exist."
    }
}

if ($null -ne $arm64Asset) {
    $downloadUrls["arm64"] = $arm64Asset.browser_download_url
} else {
    # Construct direct URL for the asset (fallback for specific versions)
    $downloadUrls["arm64"] = "https://github.com/$repo/releases/download/$tag/$arm64AssetName"
    
    # For specific versions, we still try the direct URL, but warn about potential issues
    if ($Version) {
        Write-Warning "arm64 asset not found in release metadata for version $Version. Attempting direct download - this may fail if the asset doesn't exist."
    }
}

# Prepare paths - x64 goes to the original location, arm64 to a new location
$x64OutDir = $OutDir
$arm64OutDir = Join-Path -Path (Split-Path $OutDir -Parent) -ChildPath ((Split-Path $OutDir -Leaf) + '-arm64')

if (Test-Path $x64OutDir) {
    Write-Host "Cleaning x64 output directory: $x64OutDir"
    Remove-Item -Recurse -Force -LiteralPath $x64OutDir
}
New-Item -ItemType Directory -Path $x64OutDir -Force | Out-Null

if (Test-Path $arm64OutDir) {
    Write-Host "Cleaning arm64 output directory: $arm64OutDir"
    Remove-Item -Recurse -Force -LiteralPath $arm64OutDir
}
New-Item -ItemType Directory -Path $arm64OutDir -Force | Out-Null

# Use a dedicated temp working directory for downloads and extraction
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("onnxruntime-dl-" + [guid]::NewGuid().ToString('N'))
if (Test-Path $tempRoot) { Remove-Item -Recurse -Force -LiteralPath $tempRoot -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

$x64ZipPath = Join-Path $tempRoot $x64AssetName
$arm64ZipPath = Join-Path $tempRoot $arm64AssetName
$x64TmpExtract = Join-Path $tempRoot "__extract_x64__"
$arm64TmpExtract = Join-Path $tempRoot "__extract_arm64__"

function Download-And-Extract {
    param(
        [string]$Architecture,
        [string]$DownloadUrl,
        [string]$ZipPath,
        [string]$TmpExtract,
        [string]$OutputDir,
        [string]$AssetName
    )

    # Download fresh copy to temp
    Write-Host "Downloading $Architecture package: $AssetName from $DownloadUrl ..."
    try {
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath -ErrorAction Stop
    }
    catch {
        $errorMessage = "Failed to download '$DownloadUrl'. $_"
        if ($Version -and (($Architecture -eq "x64" -and $null -eq $x64Asset) -or ($Architecture -eq "arm64" -and $null -eq $arm64Asset))) {
            $errorMessage += "`nNote: The specified version '$Version' may not have a Windows $Architecture package available."
        }
        throw $errorMessage
    }

    # Extract and flatten so files are directly under $OutputDir
    Write-Host "Extracting $Architecture package to $OutputDir ..."
    if (Test-Path $TmpExtract) { Remove-Item -Recurse -Force -LiteralPath $TmpExtract }
    New-Item -ItemType Directory -Path $TmpExtract -Force | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $TmpExtract -Force

    # Move inner content up if the zip has a single root folder
    $top = @(Get-ChildItem -LiteralPath $TmpExtract -Force)
    if ($top.Count -eq 1 -and $top[0].PSIsContainer) {
        $innerRoot = $top[0].FullName
        Get-ChildItem -LiteralPath $innerRoot -Force | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination $OutputDir -Force
        }
    } else {
        foreach ($item in $top) {
            Move-Item -LiteralPath $item.FullName -Destination $OutputDir -Force
        }
    }

    Write-Host "ONNX Runtime $Architecture downloaded and extracted to: $OutputDir"
}

try {
    # Download and extract x64 version
    Download-And-Extract -Architecture "x64" -DownloadUrl $downloadUrls["x64"] -ZipPath $x64ZipPath -TmpExtract $x64TmpExtract -OutputDir $x64OutDir -AssetName $x64AssetName

    # Download and extract arm64 version
    Download-And-Extract -Architecture "arm64" -DownloadUrl $downloadUrls["arm64"] -ZipPath $arm64ZipPath -TmpExtract $arm64TmpExtract -OutputDir $arm64OutDir -AssetName $arm64AssetName

    Write-Host ""
    Write-Host "Both packages downloaded successfully:"
    Write-Host "  x64 package: $x64OutDir"
    Write-Host "  arm64 package: $arm64OutDir"
    Write-Output @($x64OutDir, $arm64OutDir)
}
finally {
    # Always clean up temporary working directory (zip and temp extract)
    if (Test-Path $tempRoot) {
        try { Remove-Item -Recurse -Force -LiteralPath $tempRoot -ErrorAction SilentlyContinue } catch { }
    }
}
