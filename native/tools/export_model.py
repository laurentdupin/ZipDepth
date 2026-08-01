"""Derive a deterministic, pickle-free ZipDepth native model container."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

MAGIC = b"ZIPDMOD1"
META_MAGIC = b"ZIPMETA1"
VERSION = 1
ENDIAN = 0x01020304
FLOAT32 = 1
ALIGNMENT = 64
HEADER = struct.Struct("<8sIIIIQQQQQ")
RECORD = struct.Struct("<112sII4QQQQIIQ")
METADATA = struct.Struct("<8sIIIIII32s64s")
KINDS = {"base_gpu": 0, "base_mobile": 1}
CONVERTER = "zipdepth-export-pytorch-v1"


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=KINDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    canonical = sha256_file(args.checkpoint)
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = loaded.get("model_state_dict", loaded)
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint is not a non-empty state dictionary")
    tensors = []
    for name in sorted(state):
        value = state[name]
        if name.endswith(".num_batches_tracked"):
            continue
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise TypeError(f"unsupported state entry: {name}")
        if not 1 <= value.ndim <= 4:
            raise ValueError(f"unsupported tensor rank: {name}")
        payload = value.detach().cpu().contiguous().numpy().tobytes(order="C")
        tensors.append((name, tuple(value.shape), payload, zlib.crc32(payload)))
    directory_offset = HEADER.size
    directory_bytes = len(tensors) * RECORD.size
    metadata_offset = directory_offset + directory_bytes
    data_offset = align(metadata_offset + METADATA.size)
    cursor = data_offset
    records = []
    for name, shape, payload, checksum in tensors:
        encoded = name.encode("utf-8")
        if not encoded or len(encoded) >= 112:
            raise ValueError(f"invalid tensor name: {name}")
        cursor = align(cursor)
        dimensions = list(shape) + [0] * (4 - len(shape))
        records.append(RECORD.pack(
            encoded, FLOAT32, len(shape), *dimensions, cursor, len(payload),
            len(payload) // 4, checksum, 0, 0))
        cursor += len(payload)
    kind = KINDS[args.variant]
    metadata = METADATA.pack(
        META_MAGIC, 1, METADATA.size, VERSION, kind, 0, 0, canonical,
        CONVERTER.encode("ascii"))
    header = HEADER.pack(
        MAGIC, VERSION, ENDIAN, kind, len(tensors), directory_offset,
        directory_bytes, data_offset, cursor, metadata_offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(header)
        for record in records:
            output.write(record)
        output.write(metadata)
        output.write(b"\0" * (data_offset - output.tell()))
        for (_, _, payload, _), record in zip(tensors, records):
            offset = RECORD.unpack(record)[7]
            output.write(b"\0" * (offset - output.tell()))
            output.write(payload)
    receipt = {
        "format": "ZIPDMOD1", "version": VERSION, "variant": args.variant,
        "output": str(args.output.resolve()), "bytes": cursor,
        "tensor_count": len(tensors),
        "derivation": {
            "canonical_sha256": canonical.hex(), "converter": CONVERTER,
            "format_version": VERSION,
            "cache_key": f"zipdepth:{canonical.hex()}:{CONVERTER}:{VERSION}:{args.variant}",
        },
    }
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
