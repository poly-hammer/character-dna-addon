"""A concurrent Windows preview reader must not abort a successful solve."""

from pathlib import Path

import pytest

from character_dna.editors.converter import pose_worker


def test_worker_retries_atomic_publication_with_an_open_reader(tmp_path, monkeypatch):
    temporary, target = tmp_path / "preview.tmp", tmp_path / "preview.json"
    temporary.write_text("new")
    target.write_text("old")
    replace, attempts = Path.replace, []

    def with_reader(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            assert target.read_text() == "old"
            raise PermissionError("reader denies delete sharing")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", with_reader)
    monkeypatch.setattr(pose_worker.time, "sleep", lambda _seconds: None)
    pose_worker._publish(temporary, target)
    assert len(attempts) == 3
    assert target.read_text() == "new"
    assert not temporary.exists()


def test_worker_publication_does_not_hide_a_permanent_failure(tmp_path, monkeypatch):
    temporary, target = tmp_path / "preview.tmp", tmp_path / "preview.json"
    temporary.write_text("new")
    target.write_text("old")

    def denied(_source, _destination):
        raise PermissionError("permanently denied")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(pose_worker.time, "sleep", lambda _seconds: None)
    with pytest.raises(PermissionError, match="permanently denied"):
        pose_worker._publish(temporary, target)
    assert target.read_text() == "old"
    assert temporary.read_text() == "new"
