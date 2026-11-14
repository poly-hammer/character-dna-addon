"""Bounded, checksummed structured metadata over the SDK's string-only API."""

import base64
import hashlib
import json
import zlib

from typing import Any


PREFIX = "character_dna.partitions"
CHUNK_BYTES = 256 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024**2
MAX_ENCODED_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 512 * 1024**2
MAX_METADATA_ENTRIES = 4096
UINT32_MAX = 2**32 - 1
MAX_DOCUMENT_BYTES = 2 * 1024**3


def encode(payload: dict, *, prefix: str = PREFIX) -> dict[str, str]:
    raw = bytearray()
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    for fragment in encoder.iterencode(payload):
        if len(raw) + len(fragment) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"DNA metadata {prefix} exceeds the uncompressed payload budget.")
        raw.extend(fragment.encode("ascii"))
    packed = base64.b64encode(zlib.compress(raw)).decode("ascii")
    if len(packed) > MAX_ENCODED_BYTES:
        raise ValueError(f"DNA metadata {prefix} exceeds the encoded payload budget.")
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
        if any(type(header[field]) is not int for field in ("bytes", "encoded_bytes", "chunks")) or (
            not 0 < header["bytes"] <= MAX_PAYLOAD_BYTES
            or not 0 < header["encoded_bytes"] <= MAX_ENCODED_BYTES
            or not 0 < header["chunks"] <= min(MAX_METADATA_ENTRIES - 1, MAX_ENCODED_BYTES // CHUNK_BYTES + 1)
        ):
            raise ValueError("Invalid component metadata size.")
        chunks = []
        size = 0
        for index in range(header["chunks"]):
            chunk = reader.getMetaDataValue(f"{prefix}.{index:06d}")
            size += len(chunk)
            if not 0 < len(chunk) <= CHUNK_BYTES or size > header["encoded_bytes"]:
                raise ValueError("Truncated or oversized component metadata chunk.")
            chunks.append(chunk)
        encoded = "".join(chunks)
        if len(encoded) != header["encoded_bytes"]:
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
    write_many(writer, reader, {prefix: payload})


def write_many(writer: Any, reader: Any, payloads: dict[str, dict], strings: dict[str, str] | None = None) -> None:
    """Replace related namespaces in one transaction, preserving foreign keys."""
    values = {}
    for index in range(reader.getMetaDataCount()):
        key = reader.getMetaDataKey(index)
        if not any(key == prefix or key.startswith(prefix + ".") for prefix in payloads):
            values[key] = reader.getMetaDataValue(key)
    for prefix, payload in payloads.items():
        values.update(encode(payload, prefix=prefix))
    values.update(strings or {})
    validate_fields(values)
    writer.clearMetaData()
    for key, value in values.items():
        writer.setMetaData(key, value)


def validate_fields(values: dict[str, str]) -> None:
    """Bound the complete descriptor before calling the string-only native API.

    DNA serializes string lengths, metadata counts and offsets as uint32. Keep
    the application budget much smaller, including preserved foreign namespaces.
    """
    if len(values) > min(MAX_METADATA_ENTRIES, UINT32_MAX):
        raise ValueError(f"DNA metadata exceeds the {MAX_METADATA_ENTRIES}-entry budget.")
    size = 4
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str) or "\0" in key or "\0" in value:
            raise ValueError("DNA metadata requires strings without literal NUL characters.")
        lengths = (len(key.encode("utf-8")), len(value.encode("utf-8")))
        if max(lengths) > UINT32_MAX:
            raise ValueError("DNA metadata string exceeds its uint32 length limit.")
        size += sum(lengths) + 8
        if size > MAX_METADATA_BYTES:
            raise ValueError("DNA metadata exceeds the aggregate storage budget (512 MiB by default).")
