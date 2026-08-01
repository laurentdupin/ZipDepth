# ZipDepth native runtime

This directory contains a dependency-free C++17/Vulkan inference runtime for
the official ZipDepth base checkpoints. One weight-neutral InferBridge ABI 2
harness serves both derived artifact variants:

- `base_gpu`: the server/GPU convex-unfold head;
- `base_mobile`: the NPU/mobile unfold-free head.

The `.pth` checkpoint remains the canonical artifact. `export_model.py`
derives a deterministic, pickle-free `ZIPDMOD1` (`.zipd`) container whose
embedded receipt records the canonical SHA-256, converter ID, format version,
and model kind.

## Build

```powershell
cmake -S native -B native/out/build -DZIPDEPTH_WITH_VULKAN=ON
cmake --build native/out/build --config Release --target zipdepth_native
```

The Vulkan SDK is supplied by the build environment; no machine-specific SDK
path is stored in this repository. The resulting Windows harness is
`native/out/build/Release/zipdepth_native.dll`.

## Convert

```powershell
python native/tools/export_model.py --checkpoint checkpoints/zipdepth_base.pth --variant base_gpu --output native/out/models/zipdepth_base.zipd
python native/tools/export_model.py --checkpoint checkpoints/zipdepth_base_npu.pth --variant base_mobile --output native/out/models/zipdepth_base_mobile.zipd
```

## Validate

```powershell
python native/tests/numerical_parity.py --dll native/out/build/Release/zipdepth_native.dll --checkpoint checkpoints/zipdepth_base.pth --model native/out/models/zipdepth_base.zipd
python native/tests/numerical_parity.py --dll native/out/build/Release/zipdepth_native.dll --checkpoint checkpoints/zipdepth_base_npu.pth --model native/out/models/zipdepth_base_mobile.zipd --mobile
$env:ZIPDEPTH_MODEL=(Resolve-Path native/out/models/zipdepth_base.zipd)
native/out/build/Release/zipdepth_d3d12_full_graph.exe
```

The external-resource canary supplies Core-owned shared D3D12 BGRA and R32
textures plus shared fences through the public InferBridge ABI 2 bindings. It
requires zero per-frame tensor upload/download deltas and a submit return below
5 ms. Android can use the same Vulkan graph and mobile artifact; importing an
InferBridge-owned `AHardwareBuffer` is the remaining platform transfer adapter,
not a different model graph or public model card.
