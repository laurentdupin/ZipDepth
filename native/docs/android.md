# Android and Meta Quest native inference

The dependency-free ZipDepth runtime builds as an Android ARM64 shared
library and exports the canonical InferBridge harness ABI 2 entry point.
Android uses a persistent model worker: `submit` validates and admits the
borrowed HOST bindings, queues Vulkan work, and returns without performing
inference on the caller thread. Up to three jobs may be admitted. Cancellation
does not become terminal until any running Vulkan inference has finished, so
InferBridge may safely retain and recycle its input/output storage.

Run the reproducible Quest benchmark with Unity's Android SDK/NDK and a host
Vulkan SDK containing `glslc`:

```powershell
$env:VULKAN_SDK = "<host Vulkan SDK>"
native/tools/android/run_quest_benchmark.ps1 -Variant base_mobile
native/tools/android/run_quest_benchmark.ps1 -Variant base_gpu
```

The script creates only an ignored `native/.BuildAndroid` directory and the
probe-owned `/data/local/tmp/zipdepth-android-benchmark` device directory.
The resulting library is
`native/.BuildAndroid/quest-arm64/libzipdepth_native.so`.

## Quest 3S result

On the Adreno 740 in a Quest 3S, persistent 256x128 tensor inference including
HOST tensor upload and depth readback measured about 60 ms (16.5 FPS) for both
official heads. Through the public InferBridge ABI, a 1920x1080 BGRA frame
with `Size=140` measured 50.246 ms (19.902 FPS) for the Quest-optimized
`base_gpu` path. The earlier untuned `base_mobile` path measured 69.947 ms
(14.297 FPS). Those figures include CPU preprocessing,
Vulkan inference, readback, and source-sized float output publication.

The `base_gpu` Vulkan executor uses Adreno-friendly shared-weight 1x1
convolution, tiled 3x3 convolution, a smaller high-channel feature-map tile,
and the specialized depthwise path. Numerical parity against the official
PyTorch graph passes with maximum absolute error below `9e-8`.

The dynamic harness probe also passed correlated repeated output,
sub-0.02-ms submit return, cancellation, model/runtime destruction, and
`dlclose`. Exact samples are recorded in
`android_quest3s_benchmark_2026-08-07.json`.

The current Android capability is HOST memory. Direct InferBridge-owned
AHardwareBuffer/Vulkan input and output remains a separate capability gate;
the harness does not advertise it or silently substitute HOST transfers.
