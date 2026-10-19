"""Native DNA metadata feasibility checks for a reversible merged rig document.

These use SDK binary streams, not the addon's partition archive. Large tests
exercise bounded sizes; the binary format's uint32 limits require source audit.
"""

import base64
import hashlib
import json
import zlib

from contextlib import contextmanager

import pytest

from character_dna.bindings import dna
from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
from character_dna.dna_io.coordinates import configuration
from constants import TEST_DNA_FOLDER


@contextmanager
def descriptor_writer():
    source = get_dna_reader(TEST_DNA_FOLDER / "ada" / "head.dna", data_layer="Descriptor")
    stream = dna.MemoryStream()
    writer = dna.BinaryStreamWriter(stream)
    try:
        writer.setFrom(source, dna.DataLayer_Descriptor, dna.UnknownLayerPolicy_Preserve, None)
        writer.clearMetaData()
        yield writer, stream
    finally:
        release_dna_handle(writer)
        release_dna_handle(stream)
        release_dna_handle(source)


@contextmanager
def read_written(writer, stream):
    writer.write()
    assert dna.Status.isOk(), dna.Status.get().message
    stream.seek(0)
    reader = dna.BinaryStreamReader(stream, configuration("Descriptor", blender=False), None)
    try:
        reader.read()
        assert dna.Status.isOk(), dna.Status.get().message
        yield reader
    finally:
        release_dna_handle(reader)


@pytest.mark.parametrize("size", [255, 256, 65535, 65536, 1024**2, 16 * 1024**2, 64 * 1024**2])
def test_native_metadata_value_size(size):
    pattern = "0123456789abcdef"
    value = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
    with descriptor_writer() as (writer, stream):
        writer.setMetaData("character_dna.probe", value)
        with read_written(writer, stream) as reader:
            actual = reader.getMetaDataValue("character_dna.probe")
            assert isinstance(actual, str)
            assert len(actual) == size
            assert actual == value


@pytest.mark.parametrize("size", [255, 256, 65535, 65536, 1024**2])
def test_native_metadata_key_size(size):
    key = "k" * size
    with descriptor_writer() as (writer, stream):
        writer.setMetaData(key, "complete value")
        with read_written(writer, stream) as reader:
            assert reader.getMetaDataKey(0) == key
            assert reader.getMetaDataValue(key) == "complete value"


def test_native_metadata_string_semantics():
    values = {
        "unicode_颈部_🧬": "tête / cuerpo / 身体 / 🧬",
        "escaped": json.dumps({"nul": "\0", "newline": "\n", "quote": '"'}, ensure_ascii=True),
        "": "empty key",
        "empty": "",
        "replace": "second",
        "nul_value": "before",
        "nul_key": "truncated key",
    }
    with descriptor_writer() as (writer, stream):
        for key, value in values.items():
            writer.setMetaData(key, value)
        writer.setMetaData("replace", "first")
        writer.setMetaData("replace", "second")
        # Python None currently crashes the installed native binding, despite
        # the C++ API documenting nullptr deletion. Probe unsafe types only in
        # isolated subprocesses (scratches/probe_dna_metadata_types.py).
        writer.setMetaData("nul_value", "before\0after")
        writer.setMetaData("nul_key\0discarded", "truncated key")
        with read_written(writer, stream) as reader:
            assert reader.getMetaDataCount() == len(values)
            assert {reader.getMetaDataKey(i) for i in range(reader.getMetaDataCount())} == set(values)
            for key, value in values.items():
                assert reader.getMetaDataValue(key) == value
            assert reader.getMetaDataValue("absent") == ""
            assert json.loads(reader.getMetaDataValue("escaped"))["nul"] == "\0"


def test_native_metadata_clear_and_rebuild_preserves_other_consumers():
    original = {"external.vendor": "keep", "character_dna.old_chunk": "remove"}
    with descriptor_writer() as (writer, stream):
        for key, value in original.items():
            writer.setMetaData(key, value)
        with read_written(writer, stream) as reader:
            retained = {
                reader.getMetaDataKey(index): reader.getMetaDataValue(reader.getMetaDataKey(index))
                for index in range(reader.getMetaDataCount())
                if not reader.getMetaDataKey(index).startswith("character_dna.")
            }
    with descriptor_writer() as (writer, stream):
        for key, value in original.items():
            writer.setMetaData(key, value)
        writer.clearMetaData()
        for key, value in (retained | {"character_dna.new_chunk": "new"}).items():
            writer.setMetaData(key, value)
        with read_written(writer, stream) as reader:
            assert reader.getMetaDataCount() == 2
            assert reader.getMetaDataValue("external.vendor") == "keep"
            assert reader.getMetaDataValue("character_dna.old_chunk") == ""
            assert reader.getMetaDataValue("character_dna.new_chunk") == "new"


def test_native_metadata_many_entries():
    # The native setter searches existing entries linearly. Avoid an unnecessary
    # quadratic 65K-entry stress run: inspect uint32 count/serialization in C++.
    with descriptor_writer() as (writer, stream):
        for index in range(4096):
            writer.setMetaData(f"chunk.{index:06d}", str(index))
        with read_written(writer, stream) as reader:
            assert reader.getMetaDataCount() == 4096
            for index in range(4096):
                key = f"chunk.{index:06d}"
                assert reader.getMetaDataKey(index) == key
                assert reader.getMetaDataValue(key) == str(index)


@pytest.mark.parametrize("source_kind", ["head", "body"])
def test_partition_payload_survives_canonical_writes(tmp_path, source_kind):
    """All real Ada meshes/vertices, Unicode, escaped NUL, and >1 chunk.

    This probes transport feasibility only, not a finished ownership schema or
    a consolidated behavior graph. Originals are never modified.
    """
    payload = {"schema": 1, "label": "Bruce_颈部", "escaped_nul": "\0", "components": {}}
    for kind in ("head", "body"):
        reader = get_dna_reader(TEST_DNA_FOLDER / "ada" / f"{kind}.dna")
        try:
            payload["components"][kind] = {
                "joints": [reader.getJointName(index) for index in range(reader.getJointCount())],
                "meshes": [
                    {
                        "name": reader.getMeshName(index),
                        "original_vertices": list(range(reader.getVertexPositionCount(index))),
                        "original_positions": list(
                            zip(
                                reader.getVertexPositionXs(index),
                                reader.getVertexPositionYs(index),
                                reader.getVertexPositionZs(index),
                                strict=True,
                            )
                        ),
                    }
                    for index in range(reader.getMeshCount())
                ],
            }
        finally:
            release_dna_handle(reader)
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    encoded = base64.b64encode(zlib.compress(raw)).decode("ascii")
    chunk_size = 256 * 1024
    chunks = [encoded[offset : offset + chunk_size] for offset in range(0, len(encoded), chunk_size)]
    assert len(chunks) > 1
    manifest = {"schema": 1, "encoding": "zlib+base64", "count": len(chunks), "sha256": hashlib.sha256(raw).hexdigest()}
    metadata = {"character_dna.partitions": json.dumps(manifest), "unrelated.consumer": "keep me"}
    metadata.update({f"character_dna.partitions.{index:06d}": chunk for index, chunk in enumerate(chunks)})
    # Also exercise a large single value through the production coordinate writer.
    metadata["character_dna.large_probe"] = "v" * (16 * 1024**2)
    original_source = TEST_DNA_FOLDER / "ada" / f"{source_kind}.dna"
    source = original_source
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_original_metadata = None
    for revision in range(2):
        destination = tmp_path / f"revision-{revision}.dna"
        reader = get_dna_reader(source)
        writer = get_dna_writer(destination)
        try:
            if expected_original_metadata is None:
                expected_original_metadata = {
                    reader.getMetaDataKey(index): reader.getMetaDataValue(reader.getMetaDataKey(index))
                    for index in range(reader.getMetaDataCount())
                }
            mesh_count = reader.getMeshCount()
            joint_count = reader.getJointCount()
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            writer.setName(f"Metadata probe revision {revision}")
            if revision == 0:
                for key, value in metadata.items():
                    writer.setMetaData(key, value)
            writer.write()
            assert dna.Status.isOk(), dna.Status.get().message
        finally:
            release_dna_handle(writer)
            release_dna_handle(reader)
        assert destination.read_bytes()[:3] == b"DNA"
        for layer in ("Descriptor", "All"):
            result = get_dna_reader(destination, data_layer=layer)
            try:
                assert result.getName() == f"Metadata probe revision {revision}"
                for key, value in (expected_original_metadata | metadata).items():
                    assert result.getMetaDataValue(key) == value
                joined = "".join(
                    result.getMetaDataValue(f"character_dna.partitions.{index:06d}") for index in range(len(chunks))
                )
                decoded = zlib.decompress(base64.b64decode(joined))
                assert hashlib.sha256(decoded).hexdigest() == manifest["sha256"]
                assert decoded == raw
                if layer == "All":
                    assert result.getMeshCount() == mesh_count
                    assert result.getJointCount() == joint_count
            finally:
                release_dna_handle(result)
        source = destination
    assert hashlib.sha256(original_source.read_bytes()).hexdigest() == original_hash
    print(f"\nOwnership transport probe: JSON={len(raw)}, encoded={len(encoded)}, chunks={len(chunks)}")
