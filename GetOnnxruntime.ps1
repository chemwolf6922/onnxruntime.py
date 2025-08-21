[CmdletBinding()]
param(
    # Version hint to download, e.g. "1.18" or "1.22". Downloads the latest release matching the major.minor version. If omitted, downloads the latest release.
    [string]$Version,

    # Output directory where the package will be extracted
    [string]$OutDir = (Join-Path -Path $PSScriptRoot -ChildPath 'onnxruntime')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# If Version is not provided, check if current git HEAD is a tag and use its major.minor version
if (-not $Version) {
    try {
        Push-Location $PSScriptRoot
        $gitTag = git describe --exact-match --tags HEAD 2>$null
        if ($gitTag -and $LASTEXITCODE -eq 0) {
            # Extract major.minor version from tag (format: major.minor[.suffix])
            $cleanTag = $gitTag.TrimStart('v','V')
            if ($cleanTag -match '^(\d+\.\d+)') {
                $Version = $matches[1]
                Write-Host "Using version hint '$Version' from current git tag '$gitTag'"
            }
        }
    }
    catch {
        # Ignore any git errors and continue without version hint
    }
    finally {
        Pop-Location
    }
}

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
        [string]$VersionHint
    )

    Write-Host "Searching for the latest release with Windows packages (x64 and arm64)..."
    
    $releases = Get-AllReleases
    $foundRelease = $null
    $skippedReleases = @()

    # If version hint is provided, validate and normalize it
    $majorMinorPattern = $null
    if ($VersionHint) {
        # Remove 'v' prefix if present and validate format
        $cleanVersion = $VersionHint.TrimStart('v','V')
        if ($cleanVersion -match '^(\d+\.\d+)(?:\.\d+)*$') {
            $majorMinorPattern = $matches[1]
            Write-Host "Looking for the latest release matching version pattern: $majorMinorPattern.*"
        } else {
            throw "Version hint '$VersionHint' is not valid. Please provide a version in format 'major.minor' (e.g., '1.22')."
        }
    }

    foreach ($release in $releases) {
        $tag = $release.tag_name
        $versionNumber = $tag.TrimStart('v','V')
        
        # If version hint is specified, check if this release matches
        if ($majorMinorPattern) {
            if (-not $versionNumber.StartsWith($majorMinorPattern)) {
                continue  # Skip releases that don't match the version pattern
            }
        }
        
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
        if ($majorMinorPattern) {
            throw "No releases found matching version pattern '$majorMinorPattern.*' with both Windows x64 and arm64 packages. Please check the repository manually or try a different version pattern."
        } else {
            throw "No recent releases found with both Windows x64 and arm64 packages. Please check the repository manually."
        }
    }

    if ($skippedReleases.Count -gt 0) {
        if ($majorMinorPattern) {
            Write-Host "Found release $($foundRelease.tag_name) matching pattern '$majorMinorPattern.*' with both Windows packages (skipped $($skippedReleases.Count) release$(if ($skippedReleases.Count -ne 1) {'s'}))."
        } else {
            Write-Host "Found release $($foundRelease.tag_name) with both Windows packages (skipped $($skippedReleases.Count) newer release$(if ($skippedReleases.Count -ne 1) {'s'}))."
        }
    } else {
        if ($majorMinorPattern) {
            Write-Host "Using latest release $($foundRelease.tag_name) matching pattern '$majorMinorPattern.*'."
        } else {
            Write-Host "Using latest release $($foundRelease.tag_name)."
        }
    }

    return $foundRelease
}

# Resolve release and asset
$release = Find-LatestReleaseWithWindows -VersionHint $Version
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
    # This should not happen since Find-LatestReleaseWithWindows ensures both assets exist
    throw "x64 asset '$x64AssetName' not found in release $tag. This is unexpected - please report this issue."
}

if ($null -ne $arm64Asset) {
    $downloadUrls["arm64"] = $arm64Asset.browser_download_url
} else {
    # This should not happen since Find-LatestReleaseWithWindows ensures both assets exist
    throw "arm64 asset '$arm64AssetName' not found in release $tag. This is unexpected - please report this issue."
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
        if ($Version) {
            $errorMessage += "`nNote: The specified version hint '$Version' resolved to a release that may have issues with the Windows $Architecture package."
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
