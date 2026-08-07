param(
    [string]$UnityEditorRoot = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1",
    [string]$DeviceSerial = "340YC10G7Y0X0N",
    [ValidateSet("base_gpu", "base_mobile")]
    [string]$Variant = "base_mobile",
    [string]$ModelPath = "",
    [int]$Size = 140,
    [int]$SourceWidth = 1920,
    [int]$SourceHeight = 1080,
    [int]$Warmup = 2,
    [int]$Iterations = 10
)

$ErrorActionPreference = "Stop"
$nativeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $filename = if ($Variant -eq "base_mobile") {
        "zipdepth_base_mobile.zipd"
    } else {
        "zipdepth_base.zipd"
    }
    $ModelPath = Join-Path $nativeRoot "out\models\$filename"
}
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
    throw "Set VULKAN_SDK to a host Vulkan SDK containing glslc"
}

$androidRoot = Join-Path $UnityEditorRoot `
    "Editor\Data\PlaybackEngines\AndroidPlayer"
$cmake = Join-Path $androidRoot "SDK\cmake\3.22.1\bin\cmake.exe"
$adb = Join-Path $androidRoot "SDK\platform-tools\adb.exe"
$ndk = Join-Path $androidRoot "NDK"
$buildRoot = Join-Path $nativeRoot ".BuildAndroid"
$ndkLink = Join-Path $buildRoot "unity-ndk"
$buildDirectory = Join-Path $buildRoot "quest-arm64"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
if (-not (Test-Path -LiteralPath $ndkLink)) {
    New-Item -ItemType Junction -Path $ndkLink -Target $ndk | Out-Null
}

& $adb -s $DeviceSerial get-state | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Quest device $DeviceSerial is not available"
}
& $cmake -S $nativeRoot -B $buildDirectory -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$ndkLink\build\cmake\android.toolchain.cmake" `
    "-DANDROID_NDK=$ndkLink" `
    -DANDROID_ABI=arm64-v8a `
    -DANDROID_PLATFORM=android-29 `
    -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release `
    -DZIPDEPTH_WITH_VULKAN=ON `
    -DBUILD_TESTING=OFF
if ($LASTEXITCODE -ne 0) {
    throw "ZipDepth Android configure failed"
}
& $cmake --build $buildDirectory `
    --target zipdepth_native zipdepth_harness_benchmark --parallel 8
if ($LASTEXITCODE -ne 0) {
    throw "ZipDepth Android build failed"
}

$remoteRoot = "/data/local/tmp/zipdepth-android-benchmark"
& $adb -s $DeviceSerial shell "rm -rf $remoteRoot && mkdir -p $remoteRoot"
& $adb -s $DeviceSerial push `
    (Join-Path $buildDirectory "zipdepth_harness_benchmark") `
    "$remoteRoot/benchmark" | Out-Null
& $adb -s $DeviceSerial push `
    (Join-Path $buildDirectory "libzipdepth_native.so") `
    "$remoteRoot/libzipdepth_native.so" | Out-Null
& $adb -s $DeviceSerial push $ModelPath "$remoteRoot/model.zipd" | Out-Null
& $adb -s $DeviceSerial shell `
    "chmod 755 $remoteRoot/benchmark && cd $remoteRoot && export LD_LIBRARY_PATH=. && ./benchmark ./libzipdepth_native.so model.zipd $Size $SourceWidth $SourceHeight $Warmup $Iterations"
if ($LASTEXITCODE -ne 0) {
    throw "ZipDepth Android benchmark failed"
}
