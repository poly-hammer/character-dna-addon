"""Ownership metadata fails closed before constructing any native graph."""

import json

from unittest.mock import Mock

import pytest

from character_dna.dna_io import metadata


class Reader:
    def __init__(self, values):
        self.values = values

    def getMetaDataValue(self, key):
        return self.values.get(key, "")


def test_metadata_namespaces_and_unicode():
    payload = {"name": "角色", "nested": {"value": "literal\0escaped"}, "vertices": list(range(10000))}
    values = metadata.encode(payload, prefix="test.owner")
    assert all(isinstance(value, str) and "\0" not in value for value in values.values())
    assert metadata.decode(Reader(values), prefix="test.owner") == payload


@pytest.mark.parametrize("damage", ["missing", "checksum", "size", "version", "encoding", "compressed"])
def test_metadata_rejects_corruption(damage):
    values = metadata.encode({"ownership": list(range(1000))})
    header = json.loads(values[metadata.PREFIX])
    if damage == "missing":
        del values[metadata.PREFIX + ".000000"]
    elif damage == "checksum":
        header["sha256"] = "wrong"
    elif damage == "size":
        header["bytes"] = metadata.MAX_PAYLOAD_BYTES + 1
    elif damage == "version":
        header["version"] = 99
    elif damage == "encoding":
        header["encoding"] = "unknown"
    else:
        values[metadata.PREFIX + ".000000"] = "!" * header["encoded_bytes"]
    values[metadata.PREFIX] = json.dumps(header)
    with pytest.raises(ValueError):
        metadata.decode(Reader(values))


def test_metadata_checks_encoded_size_before_native_write(monkeypatch):
    monkeypatch.setattr(metadata, "MAX_ENCODED_BYTES", 8)
    with pytest.raises(ValueError, match="encoded payload budget"):
        metadata.encode({"color": [0, 1, 0, 1]})


@pytest.mark.parametrize("limit", ["MAX_METADATA_BYTES", "MAX_METADATA_ENTRIES", "UINT32_MAX"])
def test_metadata_limits_include_foreign_fields_without_mutating_writer(monkeypatch, limit):
    monkeypatch.setattr(metadata, limit, 1)
    reader = Mock()
    reader.getMetaDataCount.return_value = 1
    reader.getMetaDataKey.return_value = "foreign"
    reader.getMetaDataValue.return_value = "preserve me"
    writer = Mock()
    with pytest.raises(ValueError):
        metadata.write_many(writer, reader, {}, {"new": "value"})
    writer.clearMetaData.assert_not_called()
    writer.setMetaData.assert_not_called()


def test_metadata_rejects_oversized_chunks(monkeypatch):
    values = metadata.encode({"colors": list(range(100))})
    monkeypatch.setattr(metadata, "CHUNK_BYTES", 8)
    with pytest.raises(ValueError, match="chunk"):
        metadata.decode(Reader(values))


def test_file_size_budget_does_not_overwrite_destination(tmp_path, monkeypatch):
    from character_dna.dna_io import get_dna_writer, release_dna_handle

    path = tmp_path / "head.dna"
    path.write_bytes(b"existing export")
    writer = get_dna_writer(path)
    try:
        writer.setName("Metadata limit check")
        monkeypatch.setattr(metadata, "MAX_DOCUMENT_BYTES", 1)
        with pytest.raises(ValueError, match="file-size budget"):
            writer.write()
        assert path.read_bytes() == b"existing export"
    finally:
        release_dna_handle(writer)


def test_native_graph_preserves_quoted_foreign_metadata(tmp_path):
    from character_dna.bindings import dna
    from character_dna.dna_io import merged_graph, release_dna_handle

    path = tmp_path / "quoted.dna"
    stream = dna.FileStream(str(path), dna.AccessMode_Write, dna.OpenMode_Binary, None)
    writer = dna.BinaryStreamWriter(stream)
    value = '{"owner":"角色", "text":"quote \\" and newline\\n"}'
    try:
        writer.setMetaData("foreign", value)
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(stream)
    graph = merged_graph.from_bytes(path.read_bytes())
    assert ["foreign", value] in graph["desc1.1"]["metadata"]
    reread = merged_graph.from_bytes(merged_graph.to_bytes(graph))
    assert ["foreign", value] in reread["desc1.1"]["metadata"]
    plain = tmp_path / "plain.dna"
    plain.write_bytes(merged_graph.to_bytes(graph))
    owned = tmp_path / "owned.dna"
    owned.write_bytes(merged_graph.to_bytes(graph, {"schema": 2, "owner": "head"}))
    assert merged_graph.binary_digest(plain) == merged_graph.binary_digest(owned)
