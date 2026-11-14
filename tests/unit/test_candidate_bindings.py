"""The test bootstrap must not replace a staged candidate with released Core bytes."""

import importlib.metadata

from pathlib import Path
from types import SimpleNamespace

from utilities import bindings


def test_candidate_binary_and_stub_take_precedence_without_a_stale_intermediate_copy(tmp_path, monkeypatch):
    source, destination, wheel = (tmp_path / name for name in ("released", "staged", "wheel"))
    for folder in (source, destination, wheel):
        folder.mkdir()
    binary = "character_dna_core.pyd"
    stub = "character_dna_core.pyi"
    (source / binary).write_bytes(b"old binary")
    (source / stub).write_bytes(b"old API")
    (source / "riglogic.pyd").write_bytes(b"SDK")
    (destination / binary).write_bytes(b"candidate binary")
    (destination / "character_dna_core.json").write_bytes(b"verified candidate manifest")
    (destination / "character_dna_core_NOTICES.txt").write_bytes(b"legacy notices")
    payload = {binary: b"candidate binary", stub: b"candidate API", "CHARACTER_DNA.md": b"license"}
    for name, data in payload.items():
        (wheel / name).write_bytes(data)
    candidate = SimpleNamespace(files=list(payload), locate_file=lambda name: wheel / name)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: candidate)
    copy = bindings._copy_changed

    def checked_copy(src, dst):
        assert Path(src) != source / binary, "A loaded candidate must never be overwritten with the released DLL"
        return copy(src, dst)

    monkeypatch.setattr(bindings, "_copy_changed", checked_copy)
    bindings.install_test_bindings(source, destination)
    assert (destination / binary).read_bytes() == b"candidate binary"
    assert (destination / stub).read_bytes() == b"candidate API"
    assert (destination / "riglogic.pyd").read_bytes() == b"SDK"
    assert (destination / "character_dna_core.json").read_bytes() == b"verified candidate manifest"
    assert (destination / "license/CHARACTER_DNA.md").read_bytes() == b"license"
    assert not (destination / "character_dna_core_NOTICES.txt").exists()


def test_without_a_candidate_the_released_bindings_are_refreshed(tmp_path, monkeypatch):
    source, destination = tmp_path / "released", tmp_path / "staged"
    source.mkdir()
    (source / "character_dna_core.pyi").write_bytes(b"released API")

    def missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    bindings.install_test_bindings(source, destination)
    assert (destination / "character_dna_core.pyi").read_bytes() == b"released API"


def test_legacy_license_layout_is_replaced_by_the_source_license_folder(tmp_path, monkeypatch):
    source, destination = tmp_path / "released", tmp_path / "staged"
    (source / "license").mkdir(parents=True)
    (source / "license/OPEN_RIGLOGIC.md").write_bytes(b"OpenRigLogic")
    (source / "license/CHARACTER_DNA.md").write_bytes(b"Character DNA")
    (destination / "native-runtime-notices").mkdir(parents=True)
    (destination / "native-runtime-notices/LICENSE").write_bytes(b"legacy")
    (destination / "character_dna_core_NOTICES.txt").write_bytes(b"legacy")

    def missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    bindings.install_test_bindings(source, destination)
    assert sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*")) == [
        "license",
        "license/CHARACTER_DNA.md",
        "license/OPEN_RIGLOGIC.md",
    ]
