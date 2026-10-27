"""Bounded, checksummed structured metadata over the SDK's string-only API."""

import base64
import hashlib
import json
import zlib

from typing import Any


PREFIX = "character_dna.partitions"
CHUNK_BYTES = 256 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024**2
MAX_DOCUMENT_BYTES = 2 * 1024**3


def encode(payload: dict, *, prefix: str = PREFIX) -> dict[str, str]:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ValueError("Component ownership metadata exceeds the supported memory budget.")
    packed = base64.b64encode(zlib.compress(raw)).decode("ascii")
    chunks = [packed[offset : offset + CHUNK_BYTES] for offset in range(0, len(packed), CHUNK_BYTES)]
    manifest = {
        "version": 2,
        "encoding": "zlib+base64",
        "chunks": len(chunks),
        "bytes": len(raw),
        "encoded_bytes": len(packed),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return {prefix: json.dumps(manifest), **{f"{prefix}.{i:06d}": value for i, value in enumerate(chunks)}}


def decode(reader: Any, *, prefix: str = PREFIX) -> dict:
    try:
        header = json.loads(reader.getMetaDataValue(prefix))
        if header["version"] != 2 or header["encoding"] != "zlib+base64":
            raise ValueError("Unsupported component metadata version or encoding.")
        if (
            not 0 < header["bytes"] <= MAX_PAYLOAD_BYTES
            or not 0 < header["chunks"] <= MAX_PAYLOAD_BYTES // CHUNK_BYTES + 1
        ):
            raise ValueError("Invalid component metadata size.")
        encoded = "".join(reader.getMetaDataValue(f"{prefix}.{i:06d}") for i in range(header["chunks"]))
        if len(encoded) != header["encoded_bytes"] or len(encoded) > MAX_PAYLOAD_BYTES:
            raise ValueError("Truncated or oversized component metadata.")
        inflater = zlib.decompressobj()
        raw = inflater.decompress(base64.b64decode(encoded, validate=True), header["bytes"] + 1)
        if len(raw) != header["bytes"] or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError("Invalid compressed component metadata.")
        if hashlib.sha256(raw).hexdigest() != header["sha256"]:
            raise ValueError("Component metadata failed its integrity check.")
        return json.loads(raw)
    except (KeyError, TypeError, json.JSONDecodeError, zlib.error, UnicodeError) as error:
        raise ValueError("Invalid component ownership metadata.") from error


def write(writer: Any, reader: Any, payload: dict, *, prefix: str = PREFIX) -> None:
    """Replace only our namespace; never pass nullable/non-string inputs to SWIG."""
    values = {}
    for index in range(reader.getMetaDataCount()):
        key = reader.getMetaDataKey(index)
        if key != prefix and not key.startswith(prefix + "."):
            values[key] = reader.getMetaDataValue(key)
    values.update(encode(payload, prefix=prefix))
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str) or "\0" in key or "\0" in value:
            raise ValueError("DNA metadata requires strings without literal NUL characters.")
    writer.clearMetaData()
    for key, value in values.items():
        writer.setMetaData(key, value)
