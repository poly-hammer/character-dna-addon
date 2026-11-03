"""File-browser options follow real DNA mesh LOD mappings, not template slots."""

from types import SimpleNamespace

import bpy
import pytest

from test_vertex_color_roundtrip import write_surface

from character_dna.dna_io import get_dna_reader, release_dna_handle
from character_dna.dna_io.misc import get_mesh_lods
from character_dna.ui import importer as ui


@pytest.mark.parametrize(
    ("count", "lods"), [(1, (0,)), (8, (0,)), (8, tuple(range(8))), (4, (0, 1, 2, 3)), (8, (0, 3)), (8, ())]
)
def test_available_lods_and_panel(tmp_path, monkeypatch, count, lods):
    path = write_surface(tmp_path / "head.dna", lod_count=count, lods=lods)
    reader = get_dna_reader(path, data_layer="Definition")
    try:
        assert reader.getLODCount() == count
        assert get_mesh_lods(reader) == list(lods)
        drawn = []
        layout = SimpleNamespace()
        layout.row = lambda: layout
        layout.label = lambda **_: None
        layout.prop = lambda _operator, prop: drawn.append(prop)
        context = SimpleNamespace(space_data=SimpleNamespace(active_operator=SimpleNamespace(filepath=str(path))))
        monkeypatch.setattr(ui, "get_import_dna_reader", lambda *_: reader)
        ui.CHARACTER_DNA_LODS_PT_panel.draw(SimpleNamespace(layout=layout), context)
        assert drawn == [f"import_lod{lod}" for lod in lods]
    finally:
        release_dna_handle(reader)


def test_panels_share_definition_reader_and_refresh_changed_file(tmp_path, monkeypatch):
    path = write_surface(tmp_path / "head.dna")
    cache = {}
    monkeypatch.setattr(ui.utilities, "get_addon_window_manager_properties", lambda *_: SimpleNamespace(dna_info=cache))
    opened = []
    original = ui.get_dna_reader

    def read(*args, **kwargs):
        opened.append(kwargs["data_layer"])
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "get_dna_reader", read)
    first = ui.get_import_dna_reader(bpy.context, str(path))
    assert get_mesh_lods(first) == [0]
    assert ui.get_import_dna_reader(bpy.context, str(path)) is first
    assert opened == ["Definition"]
    # Release the Windows read handle before replacing this asset in place.
    release_dna_handle(first)
    write_surface(path, lods=(0, 3))
    second = ui.get_import_dna_reader(bpy.context, str(path))
    assert get_mesh_lods(second) == [0, 3]
    assert opened == ["Definition", "Definition"]
    assert ui.get_import_dna_reader(bpy.context, str(tmp_path)) is None
    assert ui.get_import_dna_reader(bpy.context, "") is None
    assert cache["_dna_reader"] is None


def test_unreadable_file_does_not_keep_previous_lods(tmp_path, monkeypatch):
    cache = {}
    monkeypatch.setattr(ui.utilities, "get_addon_window_manager_properties", lambda *_: SimpleNamespace(dna_info=cache))
    valid = write_surface(tmp_path / "head.dna")
    assert ui.get_import_dna_reader(bpy.context, str(valid)) is not None
    bad = tmp_path / "invalid.dna"
    bad.write_bytes(b"not DNA")
    assert ui.get_import_dna_reader(bpy.context, str(bad)) is None
    assert ui.get_import_dna_reader(bpy.context, str(bad)) is None
