#!/usr/bin/env python3
"""Prevent Android performance shortcuts from changing ZipDepth semantics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ZIPDEPTH_ANDROID_PARITY_CONTRACT_FAILED: {message}")


def section(text: str, start: str, end: str) -> str:
    begin = text.find(start)
    require(begin >= 0, f"missing section start: {start}")
    finish = text.find(end, begin)
    require(finish >= 0, f"missing section end: {end}")
    return text[begin:finish]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("harness", type=Path)
    parser.add_argument("executor", type=Path)
    args = parser.parse_args()

    harness = args.harness.read_text(encoding="utf-8")
    executor = args.executor.read_text(encoding="utf-8")
    native = (args.harness.parent / "zipdepth_native.cpp").read_text(encoding="utf-8")
    android_input = section(native, "zipdepth_infer_android_hardware_buffer_vulkan_f32(",
                            "zipdepth_infer_dma_buf_vulkan_f32(")
    require("gpu_io->preprocess_capture(" in android_input and
            "gpu_io->preprocess(" not in android_input,
            "Android hardware buffers must use raw RGB/nearest capture preprocessing; "
            "the encoder weights already fold normalization")

    dimensions = section(harness, "void network_dimensions(", "ibrh_result status_result")
    require("__ANDROID__" not in dimensions,
            "Android must use the same aspect-preserving network dimensions")
    require("input_width * scale" in dimensions and "input_height * scale" in dimensions,
            "network dimensions must scale both source axes")

    for forbidden in (
        "temporal_warp", "read_android_luma", "previous_depth_",
        "previous_luma_", "android_frame_count_",
    ):
        require(forbidden not in harness,
                f"Android must infer each submitted frame; found {forbidden}")

    graph = section(executor, "VulkanExecutor::Tensor VulkanExecutor::infer_device(",
                    "        output = {")
    require("__ANDROID__" not in graph,
            "Android and desktop must execute the same model topology")

    required_patterns = {
        "both stage 1 blocks": r'rep\(rep\(q,"encoder\.stage1\.0"\),"encoder\.stage1\.1"\)',
        "both stage 2 blocks": r'rep\(rep\(s2,"encoder\.stage2\.0"\),"encoder\.stage2\.1"\)',
        "all six stage 3 blocks": r'for\s*\([^)]*i\s*<\s*6',
        "stage 3 channel attention": r'encoder\.stage3\.6\.fc\.0\.weight',
        "stage 3 context transform": r'encoder\.stage3\.7\.context_weight\.weight',
        "both stage 4 blocks": r'rep\(rep\(s4,"encoder\.stage4\.0"\),"encoder\.stage4\.1"\)',
    }
    for label, pattern in required_patterns.items():
        require(re.search(pattern, graph) is not None, f"full graph is missing {label}")

    print("ZIPDEPTH_ANDROID_PARITY_CONTRACT_OK")


if __name__ == "__main__":
    main()
