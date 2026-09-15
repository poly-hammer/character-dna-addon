"""The native runtime is an isolated platform binding, not a shared wheel."""

import sys

import pytest

from character_dna import bindings


def test_missing_native_runtime(monkeypatch, tmp_path):
    """Absence of the optional binary does not fall back to an installed wheel."""
    monkeypatch.delitem(sys.modules, f"{bindings.__name__}._riglogic_blender", raising=False)
    monkeypatch.setattr(bindings, "combo_folder", tmp_path)
    with pytest.raises(ModuleNotFoundError, match="Native runtime is not installed"):
        bindings.load_native_runtime()


def test_native_loader_isolated():
    """Load only from the selected bindings folder and reuse the native module."""
    if not list(bindings.combo_folder.glob("_riglogic_blender.*")):
        pytest.skip("Native binding is not built for this test interpreter")
    before = list(sys.path)
    module = bindings.load_native_runtime()
    assert bindings.load_native_runtime() is module
    assert sys.path == before
    assert module.__name__ == f"{bindings.__name__}._riglogic_blender"
    assert "_riglogic_blender" not in sys.modules
    assert "riglogic_blender" not in sys.modules
