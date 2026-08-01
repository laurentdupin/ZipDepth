"""Compare deterministic ZipDepth native artifacts with the official graph."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from zipdepth.model.architecture import create_model
    from zipdepth.utils.model_utils import strip_state_dict_prefixes

    official = create_model(
        variant="base", global_mode="balanced",
        upsample_unfold=not args.mobile).eval()
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = strip_state_dict_prefixes(loaded.get("model_state_dict", loaded))
    missing, unexpected = official.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    rng = np.random.default_rng(428)
    rgb = rng.random((3, args.size, args.size), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    normalized = np.ascontiguousarray((rgb - mean) / std)
    with torch.inference_mode():
        reference = official(torch.from_numpy(rgb[None])).numpy().reshape(
            args.size, args.size)

    library = ctypes.CDLL(str(args.dll.resolve()))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.zipdepth_create_vulkan.argtypes = [
        ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    library.zipdepth_create_vulkan.restype = ctypes.c_int
    library.zipdepth_destroy.argtypes = [ctypes.c_void_p]
    for name in ("zipdepth_infer_rgb_f32", "zipdepth_infer_tensor_vulkan_f32"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p, pointer, ctypes.c_uint32,
                             ctypes.c_uint32, pointer, ctypes.c_uint64]
        function.restype = ctypes.c_int
    context = ctypes.c_void_p()
    status = library.zipdepth_create_vulkan(
        str(args.model.resolve()).encode(), 0, ctypes.byref(context))
    if status:
        raise RuntimeError(f"native model load failed: {status}")
    try:
        outputs = {}
        for name, value in (
                ("cpu", rgb), ("vulkan", normalized)):
            output = np.empty((args.size, args.size), np.float32)
            function = (library.zipdepth_infer_rgb_f32 if name == "cpu" else
                        library.zipdepth_infer_tensor_vulkan_f32)
            status = function(
                context, value.ctypes.data_as(pointer), args.size, args.size,
                output.ctypes.data_as(pointer), output.size)
            if status:
                raise RuntimeError(f"{name} inference failed: {status}")
            outputs[name] = output
    finally:
        library.zipdepth_destroy(context)
    for name, output in outputs.items():
        difference = np.abs(reference - output)
        print(f"variant={'mobile' if args.mobile else 'server'} backend={name} "
              f"max_abs={difference.max():.9g} mean_abs={difference.mean():.9g}")
        if float(difference.max()) >= 1.0e-5:
            raise RuntimeError(f"{name} numerical parity gate failed")


if __name__ == "__main__":
    main()
