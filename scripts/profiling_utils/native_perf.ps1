[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'BuildSdk', 'BuildCandidate', 'Verify', 'Frames', 'Viewport')]
    [string]$Stage = 'Preflight',
    [string]$BlenderRoot = 'E:/repos/blender',
    [string]$OpenRigLogicRoot = 'E:/repos/OpenRigLogic',
    [string]$ReportRoot = '',
    [string]$BuildRoot = 'E:/repos/build_riglogic_native',
    [string]$BlenderExe = 'E:/repos/build_riglogic_native/candidate-install/blender.exe',
    [string]$Fixture = '',
    [string]$NativeFixture = '',
    [ValidateRange(1, 100)]
    [int]$Trials = 5,
    [ValidateRange(1, 100000)]
    [int]$Frames = 1200,
    [ValidateRange(0, 100000)]
    [int]$Warmup = 120,
    [ValidateRange(1, 3600)]
    [int]$Seconds = 60,
    [ValidateSet('SOLID', 'MATERIAL')]
    [string]$Shading = 'MATERIAL'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$AddonRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $ReportRoot) {
    $ReportRoot = Join-Path $AddonRoot ('reports/profiling/native/' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$ReportRoot = [System.IO.Path]::GetFullPath($ReportRoot)
$BuildRoot = [System.IO.Path]::GetFullPath($BuildRoot)

function Get-RepositorySnapshot {
    param([string]$Root)
    $revision = & git -C $Root rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw "Cannot read repository: $Root" }
    $branch = & git -C $Root branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Cannot read branch: $Root" }
    $status = @(& git -C $Root status --porcelain=v1)
    if ($LASTEXITCODE -ne 0) { throw "Cannot read status: $Root" }
    $submodules = @(& git -C $Root submodule status)
    if ($LASTEXITCODE -ne 0) { throw "Cannot read submodules: $Root" }
    return [ordered]@{
        root = $Root
        revision = $revision
        branch = $branch
        status = $status
        submodules = $submodules
    }
}

function Get-AssetSnapshot {
    param([string]$Path)
    $exists = Test-Path -LiteralPath $Path -PathType Leaf
    $result = [ordered]@{ path = $Path; exists = $exists }
    if ($exists) {
        $result.bytes = (Get-Item -LiteralPath $Path).Length
        $result.sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
    return $result
}

function Invoke-Preflight {
    $manifestPath = Join-Path $ReportRoot 'preflight.json'
    if (Test-Path -LiteralPath $manifestPath) {
        throw "Refusing to overwrite an existing manifest: $manifestPath"
    }
    $issues = [System.Collections.Generic.List[string]]::new()
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    $vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
    $visualStudio = @()
    if (Test-Path -LiteralPath $vswhere) {
        $installations = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json) | ConvertFrom-Json
        $visualStudio = @($installations | ForEach-Object { $_ })
    }
    if ($visualStudio.Count -eq 0) { $issues.Add('Visual Studio x64 C++ tools not found.') }
    $cmakeVersion = @()
    if ($null -ne $cmake) {
        $cmakeVersion = @(& $cmake.Source --version)
    }
    else { $issues.Add('CMake not found.') }

    $libraries = [ordered]@{}
    foreach ($relativePath in @('python/313/include/Python.h', 'python/313/libs/python313.lib', 'tbb/lib/tbb12.lib', 'zlib/include/zlib.h')) {
        $path = Join-Path (Join-Path $BlenderRoot 'lib/windows_x64') $relativePath
        $libraries[$relativePath] = Test-Path -LiteralPath $path -PathType Leaf
        if (-not $libraries[$relativePath]) { $issues.Add("Missing Blender library sentinel: $path") }
    }
    $assets = @(
        Get-AssetSnapshot (Join-Path $BlenderRoot 'release/datafiles/startup.blend')
        Get-AssetSnapshot (Join-Path $AddonRoot 'tests/test_files/dna/ada/head.dna')
        Get-AssetSnapshot (Join-Path $AddonRoot 'tests/test_files/dna/ada/body.dna')
        Get-AssetSnapshot (Join-Path $OpenRigLogicRoot 'include/riglogic/version/Version.h')
        Get-AssetSnapshot (Join-Path $OpenRigLogicRoot 'include/riglogic/Defs.h')
        Get-AssetSnapshot (Join-Path $AddonRoot 'src/addons/character_dna/editors/shape_key_editor/utilities.py')
    )
    foreach ($asset in $assets) {
        if (-not $asset.exists) { $issues.Add("Missing required asset: $($asset.path)") }
        elseif ($asset.path.EndsWith('startup.blend') -and $asset.bytes -lt 1024) {
            $issues.Add('Blender startup.blend is an LFS pointer; fetch source assets from the canonical Blender LFS endpoint.')
        }
    }
    $bindingsRoot = Join-Path $AddonRoot 'src/addons/character_dna/bindings/windows/x64/py313'
    $bindings = @(Get-ChildItem -LiteralPath $bindingsRoot -Recurse -Filter '*.pyd' -ErrorAction SilentlyContinue | ForEach-Object {
        Get-AssetSnapshot $_.FullName
    })
    if ($bindings.Count -eq 0) { $issues.Add('No Python 3.13 native add-on bindings found.') }

    $manifest = [ordered]@{
        schema_version = 1
        stage = 'preflight'
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        report_root = $ReportRoot
        build_root = $BuildRoot
        repositories = [ordered]@{
            blender = Get-RepositorySnapshot $BlenderRoot
            addon = Get-RepositorySnapshot $AddonRoot
            riglogic = Get-RepositorySnapshot $OpenRigLogicRoot
        }
        tools = [ordered]@{
            cmake_path = $(if ($null -ne $cmake) { $cmake.Source } else { $null })
            cmake_version = $cmakeVersion
            visual_studio = $visualStudio
            generator = 'Ninja'
            toolchain_requirement = 'MSVC >=19.44.35216; initialize VS DevShell with -vcvars_ver=14.44'
            architecture = 'x64'
            configuration = 'Release'
        }
        hardware = [ordered]@{
            cpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed)
            system = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
            os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture
            gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,CurrentHorizontalResolution,CurrentVerticalResolution,CurrentRefreshRate)
            disks = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace)
            power_scheme = @(& powercfg /getactivescheme)
        }
        libraries = $libraries
        assets = $assets
        bindings = $bindings
        issues = @($issues.ToArray())
        ready = ($issues.Count -eq 0)
    }
    New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Output "Manifest: $manifestPath"
    Write-Output "Preflight ready: $($manifest.ready)"
    foreach ($issue in $issues) { Write-Output "BLOCKER: $issue" }
    if (-not $manifest.ready) { throw 'Native build preflight failed; see preflight.json.' }
}

function Initialize-NativeToolchain {
    $vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installation) { throw 'Visual Studio C++ tools not found.' }
    Import-Module (Join-Path $installation 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64 -vcvars_ver=14.44'
}

function Invoke-LoggedCommand {
    param([string]$Executable, [string[]]$CommandArguments, [string]$Name)
    New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
    $log = Join-Path $ReportRoot "$Name.log"
    if (Test-Path -LiteralPath $log) { throw "Refusing to overwrite $log" }
    Write-Output "Running $Name"
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @CommandArguments *> $log
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    if ($exitCode -ne 0) { throw "$Name failed ($exitCode); see $log" }
    Write-Output "Passed $Name"
}

function Invoke-SdkBuild {
    Invoke-Preflight
    Initialize-NativeToolchain
    $build = Join-Path $BuildRoot 'sdk-ninja'
    $install = Join-Path $BuildRoot 'sdk-install'
    Invoke-LoggedCommand cmake @('-S', $OpenRigLogicRoot, '-B', $build, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL',
        "-DCMAKE_INSTALL_PREFIX=$install", '-DRL_LIBRARY_TYPE=STATIC', '-DRL_BUILD_TESTS=OFF',
        '-DRL_BUILD_BENCHMARKS=OFF', '-DRL_BUILD_EXAMPLES=OFF', '-DRL_BUILD_PYTHON_WRAPPER=',
        '-DREAD_ONLY_SOURCE_TREE=ON', '-DRL_BUILD_WITH_SSE=ON', '-DRL_BUILD_WITH_AVX=ON',
        '-DRL_BUILD_WITH_HALF_FLOATS=OFF') 'sdk-configure'
    Invoke-LoggedCommand cmake @('--build', $build, '--target', 'install', '--parallel', '16') 'sdk-build'
}

function Invoke-CandidateBuild {
    Invoke-Preflight
    Initialize-NativeToolchain
    $build = Join-Path $BuildRoot 'candidate-ninja'
    $install = Join-Path $BuildRoot 'candidate-install'
    $sdk = Join-Path $BuildRoot 'sdk-install'
    Invoke-LoggedCommand cmake @('-S', $BlenderRoot, '-B', $build, '-G', 'Ninja',
        '-C', (Join-Path $BlenderRoot 'build_files/cmake/config/blender_full.cmake'),
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DWITH_RIGLOGIC=ON', "-DCMAKE_PREFIX_PATH=$sdk", "-DCMAKE_INSTALL_PREFIX=$install") 'candidate-configure'
    Invoke-LoggedCommand cmake @('--build', $build, '--target', 'install', '--parallel', '16') 'candidate-build'
}

function Invoke-NativeVerification {
    if (-not (Test-Path -LiteralPath $Fixture -PathType Leaf) -or
        -not (Test-Path -LiteralPath $NativeFixture -PathType Leaf)) {
        throw 'Verify requires -Fixture (reference Ada) and -NativeFixture (saved native scene).'
    }
    Invoke-Preflight
    $prefix = @('--background', '--factory-startup', '--python-exit-code', '1', '--python')
    $sdkScript = Join-Path $PSScriptRoot 'native_benchmark.py'
    Invoke-LoggedCommand $BlenderExe ($prefix + @($sdkScript, '--', 'verify-sdk', '--output',
        (Join-Path $ReportRoot 'sdk.json'))) 'verify-sdk'
    foreach ($name in @('body', 'head', 'settings', 'binding', 'isolation', 'undo', 'render')) {
        $script = Join-Path $PSScriptRoot "native_${name}_test.py"
        $input = $NativeFixture
        if ($name -eq 'head' -or $name -eq 'body') { $input = $Fixture }
        Invoke-LoggedCommand $BlenderExe ($prefix + @($script, '--', '--fixture', $input, '--output',
            (Join-Path $ReportRoot "$name.json"))) "verify-$name"
    }
    Invoke-LoggedCommand $BlenderExe ($prefix + @((Join-Path $PSScriptRoot 'native_head_test.py'), '--',
        '--fixture', $Fixture, '--output', (Join-Path $ReportRoot 'eye-aim.json'), '--eye-aim')) 'verify-eye-aim'
}

function Invoke-FrameTrials {
    if (-not (Test-Path -LiteralPath $Fixture -PathType Leaf)) { throw 'An existing -Fixture is required.' }
    if (-not (Test-Path -LiteralPath $BlenderExe -PathType Leaf)) { throw 'The built Blender executable is missing.' }
    Invoke-Preflight
    $runner = Join-Path $PSScriptRoot 'native_frame_benchmark.py'
    $binaryHash = (Get-FileHash -LiteralPath $BlenderExe -Algorithm SHA256).Hash
    $fixtureHash = (Get-FileHash -LiteralPath $Fixture -Algorithm SHA256).Hash
    for ($trial = 1; $trial -le $Trials; $trial++) {
        $backends = @('python', 'native')
        if ($trial % 2 -eq 0) { $backends = @('native', 'python') }
        foreach ($backend in $backends) {
            if ((Get-FileHash -LiteralPath $BlenderExe -Algorithm SHA256).Hash -ne $binaryHash) {
                throw 'Blender binary changed during the trials.'
            }
            if ((Get-FileHash -LiteralPath $Fixture -Algorithm SHA256).Hash -ne $fixtureHash) {
                throw 'Benchmark fixture changed during the trials.'
            }
            $result = Join-Path $ReportRoot ("frames-{0}-{1:D2}.json" -f $backend, $trial)
            $log = [System.IO.Path]::ChangeExtension($result, '.log')
            Write-Output "Starting trial $trial/$Trials ($backend)"
            & $BlenderExe --background --factory-startup --python-exit-code 1 --python $runner -- frames --fixture $Fixture --output $result --backend $backend --warmup $Warmup --frames $Frames *> $log
            $exitCode = $LASTEXITCODE
            if ($exitCode -ne 0) { throw "Trial failed ($exitCode); see $log" }
            $metrics = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
            Write-Output ("Trial {0} {1}: {2:N3} ms, {3:N2} evaluated frames/s" -f $trial, $backend, $metrics.mean_ms, $metrics.evaluated_frames_per_second)
        }
    }
}

function Invoke-ViewportTrials {
    if (-not (Test-Path -LiteralPath $Fixture -PathType Leaf)) { throw 'An existing -Fixture is required.' }
    if (-not (Test-Path -LiteralPath $BlenderExe -PathType Leaf)) { throw 'The built Blender executable is missing.' }
    Invoke-Preflight
    $runner = Join-Path $PSScriptRoot 'native_viewport_benchmark.py'
    $binaryHash = (Get-FileHash -LiteralPath $BlenderExe -Algorithm SHA256).Hash
    $fixtureHash = (Get-FileHash -LiteralPath $Fixture -Algorithm SHA256).Hash
    for ($trial = 1; $trial -le $Trials; $trial++) {
        $backends = @('python', 'native')
        if ($trial % 2 -eq 0) { $backends = @('native', 'python') }
        foreach ($backend in $backends) {
            if ((Get-FileHash -LiteralPath $BlenderExe -Algorithm SHA256).Hash -ne $binaryHash -or
                (Get-FileHash -LiteralPath $Fixture -Algorithm SHA256).Hash -ne $fixtureHash) {
                throw 'Viewport executable or fixture changed during the trials.'
            }
            $result = Join-Path $ReportRoot ("viewport-{0}-{1:D2}.json" -f $backend, $trial)
            $log = [System.IO.Path]::ChangeExtension($result, '.log')
            Write-Output "Starting viewport trial $trial/$Trials ($backend/$Shading)"
            & $BlenderExe --factory-startup --window-geometry 0 0 1920 1080 --python-exit-code 1 --python $runner -- --fixture $Fixture --output $result --backend $backend --shading $Shading --warmup-seconds 10 --seconds $Seconds *> $log
            if ($LASTEXITCODE -ne 0) { throw "Viewport trial failed; see $log" }
            $metrics = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
            if (-not $metrics.passed) { throw "No valid viewport capture: $result" }
            Write-Output ("Viewport {0} {1}: {2:N2} drawn frames/s" -f $trial, $backend, $metrics.drawn_animation_frames_per_second)
        }
    }
}

switch ($Stage) {
    'Preflight' { Invoke-Preflight }
    'BuildSdk' { Invoke-SdkBuild }
    'BuildCandidate' { Invoke-CandidateBuild }
    'Verify' { Invoke-NativeVerification }
    'Frames' { Invoke-FrameTrials }
    'Viewport' { Invoke-ViewportTrials }
}
