"""Local tests must use the current bindings, including in an existing checkout."""

import shutil

import pytest

from utilities.bindings import install_test_bindings


@pytest.mark.parametrize("existing", [None, b"older", b"fresh"])
def test_binding_installation_refreshes_existing_files(tmp_path, monkeypatch, existing):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "runtime.pyd").write_bytes(b"fresh")
    if existing is not None:
        destination.mkdir()
        (destination / "runtime.pyd").write_bytes(existing)
        shutil.copystat(source / "runtime.pyd", destination / "runtime.pyd")
    install_test_bindings(source, destination)
    assert (destination / "runtime.pyd").read_bytes() == b"fresh"

    def unexpected_copy(*args, **kwargs):
        pytest.fail("Unchanged native libraries must not be overwritten while potentially loaded")

    monkeypatch.setattr(shutil, "copy2", unexpected_copy)
    install_test_bindings(source, destination)


def test_binding_installation_allows_bundled_bindings(tmp_path):
    destination = tmp_path / "bundled"
    destination.mkdir()
    binary = destination / "runtime.pyd"
    binary.write_bytes(b"bundled")
    install_test_bindings(tmp_path / "missing", destination)
    assert binary.read_bytes() == b"bundled"


def test_binding_installation_requires_bindings(tmp_path):
    with pytest.raises(FileNotFoundError, match="Please add them"):
        install_test_bindings(tmp_path / "missing", tmp_path / "destination")
