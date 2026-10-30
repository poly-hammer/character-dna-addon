"""Document caches observe new bytes and each successful write publishes once."""

from collections import OrderedDict
from pathlib import Path
from unittest.mock import create_autospec

from character_dna.dna_io import document, get_dna_reader, native_projection, release_dna_handle


def test_document_read_rechecks_file_revision(tmp_path, monkeypatch):
    path = tmp_path / "revision.dna"
    path.write_bytes(b"first")
    monkeypatch.setattr(document, "_read_cache", OrderedDict())

    def payload(source):
        return {"manifest": {"id": "revision-test", "name": source.read_bytes().decode()}}

    def project(content, _payload):
        return {"head": b"projected:" + content}, []

    projection = create_autospec(native_projection.project, side_effect=project)
    monkeypatch.setattr(document, "_payload", create_autospec(document._payload, side_effect=payload))
    monkeypatch.setattr(native_projection, "project", projection)
    manifest, parts = document.read_document(path)
    assert manifest["name"] == "first" and parts == {"head": b"projected:first"}
    # Equal-length replacement at the same path must invalidate the old view.
    path.write_bytes(b"later")
    manifest, parts = document.read_document(path)
    assert manifest["name"] == "later" and parts == {"head": b"projected:later"}
    assert projection.call_count == 2
    manifest["name"] = "caller mutation"
    parts["head"] = b"caller mutation"
    assert document.read_document(path) == (
        {"id": "revision-test", "name": "later", "scene_bridges": []},
        {"head": b"projected:later"},
    )
    assert projection.call_count == 2


def test_native_document_read_refreshes_after_publication(compact_dna, tmp_path):
    path = tmp_path / "revision.dna"
    document.merge_files({kind: compact_dna / f"{kind}.dna" for kind in document.PARTITIONS}, path, "Original")
    manifest, parts = document.read_document(path)
    before = path.read_bytes()
    manifest["name"] = "Revised"
    document.write_document(path, manifest, parts)
    assert path.read_bytes() != before
    # The independently opened SDK reader and the cached component view must agree.
    reader = get_dna_reader(path)
    try:
        assert reader.getName() == "Revised"
        current, current_parts = document.read_document(path)
        assert current["name"] == reader.getName()
        assert current_parts == parts
    finally:
        release_dna_handle(reader)


def test_native_document_write_publishes_once(compact_dna, tmp_path, monkeypatch):
    path = tmp_path / "publication.dna"
    sources = {kind: compact_dna / f"{kind}.dna" for kind in document.PARTITIONS}
    original = {kind: source.read_bytes() for kind, source in sources.items()}
    publish = create_autospec(document._atomic_write, side_effect=document._atomic_write)
    monkeypatch.setattr(document, "_atomic_write", publish)
    document.merge_files(sources, path, "Published")
    assert [Path(call.args[0]) for call in publish.call_args_list] == [path]
    assert {kind: source.read_bytes() for kind, source in sources.items()} == original
    manifest, parts = document.read_document(path)
    assert manifest["name"] == "Published"
    assert parts == original
