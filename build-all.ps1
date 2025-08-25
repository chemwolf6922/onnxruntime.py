function BuildWheels {
    [CmdletBinding()]
    param(
        # Python library version, e.g. "3.11", "3.12", or semantic version tag
        [Parameter(Mandatory=$true, Position=0)]
        [ValidatePattern('^\d+(\.\d+){1,2}$')]
        [string]$PythonVersion,

        # Target platform architecture
        [Parameter(Mandatory=$true, Position=1)]
        [ValidateSet('x64','ARM64')]
        [string]$Platform
    )

    $installPath = Join-Path -Path $PSScriptRoot -ChildPath 'nuget-packages'
    if (-not (Test-Path -LiteralPath $installPath)) {
        New-Item -ItemType Directory -Path $installPath -Force | Out-Null
    }

    $nugetPackageName = if ($Platform -eq 'x64') { 'python' } else { 'pythonarm64' }
    & nuget install $nugetPackageName -Version $PythonVersion -DependencyVersion Ignore -OutputDirectory $installPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install Python $PythonVersion for $Platform."
        exit $LASTEXITCODE
    }

    $pythonPath = Join-Path -Path $installPath -ChildPath "$nugetPackageName.$PythonVersion"
    $pythonToolsPath = Join-Path -Path $pythonPath -ChildPath 'tools'
    $pythonIncludePath = Join-Path -Path $pythonToolsPath -ChildPath 'include'
    $pythonLibsPath = Join-Path -Path $pythonToolsPath -ChildPath 'libs'
    # Parse version number for library name (e.g., "3.13.7" -> "313")
    $versionParts = $PythonVersion.Split('.')
    $libVersion = $versionParts[0] + $versionParts[1]
    $pythonLibPath = Join-Path -Path $pythonLibsPath -ChildPath "python$libVersion.lib"

    Write-Host "Using Python include path: $pythonIncludePath"
    Write-Host "Using Python library path: $pythonLibPath"

    $buildDir = Join-Path -Path $PSScriptRoot -ChildPath 'build'
    if (Test-Path -LiteralPath $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
    }

    & cmake -S $PSScriptRoot -B $buildDir `
        "-DPYTHON_INCLUDE_DIR=$pythonIncludePath" `
        "-DPYTHON_LIBRARY=$pythonLibPath" `
        "-A $Platform"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed for Python $PythonVersion on $Platform."
        exit $LASTEXITCODE
    }

    & cmake --build $buildDir --config Release

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed for Python $PythonVersion on $Platform."
        exit $LASTEXITCODE
    }
}

$pythonVersions = @(
    '3.10.11',
    '3.11.9',
    '3.12.10',
    '3.13.7'
)

$platforms = @(
    'x64',
    'ARM64'
)

$distPath = Join-Path -Path $PSScriptRoot -ChildPath 'dist'
if (Test-Path -LiteralPath $distPath) {
    Remove-Item -LiteralPath $distPath -Recurse -Force
}
New-Item -ItemType Directory -Path $distPath -Force | Out-Null

foreach ($pythonVersion in $pythonVersions) {
    foreach ($platform in $platforms) {
        BuildWheels -PythonVersion $pythonVersion -Platform $platform
    }
}
