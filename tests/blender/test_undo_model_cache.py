"""Undo may retain DNA data, but never Blender wrappers or graph sessions."""

import json
import os

from pathlib import Path
from types import SimpleNamespace

import pytest

from character_dna.runtime import controller, engine


@pytest.fixture
def model_cache(monkeypatch, tmp_path):
    loads = []
    hashes = []

    def load(path, body, _canonical, _blender):
        model = SimpleNamespace(data=Path(path).read_bytes(), body=body)
        loads.append(model)
        return model

    original_digest = engine.hashlib.file_digest

    def digest(stream, algorithm):
        hashes.append(stream.name)
        return original_digest(stream, algorithm)

    for name in ("_models", "_model_files", "_undo_models", "_records", "_errors", "_warnings"):
        monkeypatch.setattr(engine, name, {})
    monkeypatch.setattr(engine, "_restoring_undo", False)
    monkeypatch.setattr(engine, "_module", SimpleNamespace(load_model=load))
    monkeypatch.setattr(engine.hashlib, "file_digest", digest)
    monkeypatch.setattr(controller, "_undoing", False)
    path = tmp_path / "head.dna"
    path.write_bytes(b"first DNA")
    return path, loads, hashes


def restore(monkeypatch, paths):
    models = []

    def hydrate():
        for index, (path, body) in enumerate(paths):
            model = engine._model(str(path), body)
            engine._records[index] = {"model": model}
            models.append(model)

    monkeypatch.setattr(engine, "hydrate", hydrate)
    controller.after_undo()
    assert not controller._undoing
    assert not engine._restoring_undo
    assert not engine._undo_models
    return models


def test_undo_reuses_model_without_retaining_scene_records(model_cache, monkeypatch):
    path, loads, hashes = model_cache
    model = engine._model(str(path), False)
    engine._records[1] = {"carrier": object(), "contexts": {"old graph": object()}, "model": model}
    controller.before_undo()
    assert engine._records == {}
    # Evaluations during Blender's remap must not discard the recovery snapshot.
    assert engine._live_records() == []
    assert restore(monkeypatch, [(path, False)]) == [model]
    assert len(loads) == len(hashes) == 1
    controller.before_undo()
    assert restore(monkeypatch, [(path, False)]) == [model]
    assert len(loads) == len(hashes) == 1


@pytest.mark.parametrize("change", ["touch", "replace", "rewrite", "missing", "invalidate"])
def test_file_changes_do_not_return_an_unchecked_model(model_cache, monkeypatch, change):
    path, loads, hashes = model_cache
    model = engine._model(str(path), False)
    controller.before_undo()
    stamp = path.stat()
    if change == "touch":
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
    elif change == "replace":
        replacement = path.with_suffix(".new")
        replacement.write_bytes(b"other DNA")
        os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        replacement.replace(path)
    elif change in {"rewrite", "invalidate"}:
        path.write_bytes(b"other DNA")
        if change == "invalidate":
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            engine.invalidate_model_file(path)
        else:
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
    else:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            restore(monkeypatch, [(path, False)])
        assert not engine._restoring_undo and not engine._undo_models and not controller._undoing
        return
    restored = restore(monkeypatch, [(path, False)])[0]
    assert len(hashes) == 2
    assert restored.data == path.read_bytes()
    # Touching unchanged content may share the immutable model by its digest.
    assert restored is model if change == "touch" else restored is not model
    assert len(loads) == (1 if change == "touch" else 2)


def test_explicit_rebuild_rehashes_even_when_metadata_matches(model_cache):
    path, loads, hashes = model_cache
    before = engine._model(str(path), False)
    stamp = path.stat()
    path.write_bytes(b"other DNA")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = engine._model(str(path), False)
    assert after is not before and after.data == b"other DNA"
    assert len(loads) == len(hashes) == 2


def test_freshness_changes_during_load_do_not_populate_cache(model_cache, monkeypatch):
    path, _loads, _hashes = model_cache

    def changed_during_load(*_args):
        path.write_bytes(b"changed while loading")
        return object()

    monkeypatch.setattr(engine._module, "load_model", changed_during_load)
    with pytest.raises(RuntimeError, match="changed while loading"):
        engine._model(str(path), False)
    assert not engine._model_files and not engine._models


def test_undo_keeps_modes_separate_and_shares_identical_dna(model_cache, monkeypatch):
    path, loads, hashes = model_cache
    copy = path.with_name("copy.dna")
    copy.write_bytes(path.read_bytes())
    head = engine._model(str(path), False)
    body = engine._model(str(path), True)
    assert engine._model(str(copy), False) is head
    controller.before_undo()
    assert restore(monkeypatch, [(path, False), (path, True), (copy, False)]) == [head, body, head]
    assert len(loads) == 2 and len(hashes) == 3


@pytest.mark.parametrize("survivors", [0, 1])
def test_undo_prunes_unreferenced_models(model_cache, monkeypatch, survivors):
    path, _loads, _hashes = model_cache
    engine._model(str(path), False)
    engine._model(str(path), True)
    controller.before_undo()
    restore(monkeypatch, [(path, False)] if survivors else [])
    assert len(engine._models) == len(engine._model_files) == survivors


def test_restore_failure_releases_recovery_state(model_cache, monkeypatch):
    path, _loads, _hashes = model_cache
    engine._model(str(path), False)
    controller.before_undo()

    def fail():
        engine._model(str(path), False)
        raise RuntimeError("restore failed")

    monkeypatch.setattr(engine, "hydrate", fail)
    with pytest.raises(RuntimeError, match="restore failed"):
        controller.after_undo()
    assert not controller._undoing and not engine._restoring_undo
    assert not engine._undo_models and not engine._models and not engine._model_files


def test_full_invalidation_cancels_pending_reuse(model_cache, monkeypatch):
    path, loads, hashes = model_cache
    old = engine._model(str(path), False)
    controller.before_undo()
    engine.invalidate()
    assert not engine._undo_models and not engine._model_files
    assert restore(monkeypatch, [(path, False)])[0] is not old
    assert len(loads) == len(hashes) == 2


def test_replacing_native_module_disables_reuse(model_cache, monkeypatch):
    path, _loads, hashes = model_cache
    old = engine._model(str(path), False)
    controller.before_undo()
    new = object()
    monkeypatch.setattr(engine, "_module", SimpleNamespace(load_model=lambda *_args: new))
    assert restore(monkeypatch, [(path, False)]) == [new]
    assert new is not old and len(hashes) == 2


def test_canonical_writer_invalidates_pending_undo_model(model_cache):
    from character_dna.bindings import dna
    from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
    from constants import TEST_DNA_FOLDER

    path, _loads, _hashes = model_cache
    engine._model(str(path), False)
    controller.before_undo()
    assert engine._undo_models
    reader = get_dna_reader(TEST_DNA_FOLDER / "ada/body.dna")
    writer = get_dna_writer(path)
    try:
        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        writer.write()
        assert not engine._undo_models and not engine._model_files
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)


def test_backup_restore_invalidates_even_with_preserved_timestamps(model_cache, monkeypatch):
    core = pytest.importorskip("character_dna.editors.backup_manager.core")
    surfaces = pytest.importorskip("character_dna.editors.converter.surface_session")
    path, _loads, _hashes = model_cache
    stamp = path.stat()
    old = engine._model(str(path), False)
    controller.before_undo()
    folder = path.parent / "backups" / "saved"
    folder.mkdir(parents=True)
    saved = folder / path.name
    saved.write_bytes(b"other DNA")
    os.utime(saved, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    (folder / "metadata.json").write_text(json.dumps({"head_dna_path": str(path)}))
    monkeypatch.setattr(core, "get_backup_folder", lambda **_kwargs: folder.parent)
    # Verify the write's invalidation independently of the later UI refresh.
    monkeypatch.setattr(core, "get_addon_ops_module", lambda: SimpleNamespace(force_evaluate=lambda: None))
    monkeypatch.setattr(surfaces, "refresh", lambda _instance: None)
    assert core.restore_backup(SimpleNamespace(name="UndoProbe", get=lambda _key, default=None: default), "saved")
    assert not engine._undo_models and not engine._model_files
    assert restore(monkeypatch, [(path, False)])[0] is not old
